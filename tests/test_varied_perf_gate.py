from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/varied_perf_gate.py")


def _module():
    spec = importlib.util.spec_from_file_location("varied_perf_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Dataclasses resolves postponed annotations through sys.modules while the
    # class decorator executes. Register the dynamically loaded module first,
    # matching normal import semantics.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def test_query_corpus_is_large_unique_and_diverse():
    module = _module()
    queries = module.default_queries()
    assert len(queries) >= 100
    assert len(set(queries)) == len(queries)
    assert any(any("一" <= char <= "鿿" for char in query) for query in queries)
    assert any(query.startswith("/") for query in queries)
    assert any("127.0.0.1" in query for query in queries)
    assert any("no result" in query or "absent" in query for query in queries)


def test_targets_cover_all_modes_and_concurrency_is_in_runner():
    module = _module()
    assert module.TARGETS["keyword"]["http_p95_ms"] == 300.0
    assert module.TARGETS["keyword"]["server_p95_ms"] == 150.0
    assert module.TARGETS["dense"]["http_p95_ms"] == 800.0
    assert module.TARGETS["hybrid"]["http_p95_ms"] == 1000.0
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for concurrency in (1, 5)" in source
    assert "repeated" in source and "varied" in source
    assert "fallback_http" in source and "degraded_http" in source


def test_percentile_and_gate_logic():
    module = _module()
    assert module.percentile([1, 2, 3, 4, 100], 95) == 100
    result = {
        "varied": {"http_200": 200, "http": {"p95_ms": 250}, "server": {"p95_ms": 100}},
        "repeated": {"http_200": 200},
    }
    passed, failures = module.check_result("keyword", result, 200)
    assert passed is True
    assert failures == []


def test_self_test_and_shell_syntax():
    proc = subprocess.run(["python", str(SCRIPT), "--self-test"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert '"query_count"' in proc.stdout
    shell = subprocess.run(["bash", "-n", "scripts/varied_perf_gate.sh"], capture_output=True, text=True)
    assert shell.returncode == 0, shell.stderr


def test_gate_is_explicitly_host_only():
    wrapper = Path("scripts/varied_perf_gate.sh").read_text(encoding="utf-8")
    assert "Host-only" in wrapper
    assert "GitHub-hosted CI" in wrapper


# ---------------------------------------------------------------------------
# D-version contract tests: persistent client, trust_env=False, real
# concurrency model, correct URL / pool, correct percentile math, non-zero
# exit on failure.
# ---------------------------------------------------------------------------


def test_gate_uses_single_persistent_async_client():
    """The gate must build ONE httpx.AsyncClient per gate run (not per request)
    so HTTP/1.1 keep-alive can amortize the 3-way handshake over many requests."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "httpx.AsyncClient" in source, "gate must use httpx.AsyncClient"
    assert "trust_env=False" in source, "httpx client must set trust_env=False"
    assert "max_keepalive_connections" in source, (
        "pool must declare max_keepalive_connections"
    )
    # Once per gate run, not per probe.
    assert "_build_async_probe_factory" in source, (
        "factory must be called once per run so the client is reused"
    )


def test_gate_disables_system_proxy_env():
    """trust_env=False on the httpx client must be present in the factory so
    HTTP_PROXY / HTTPS_PROXY / ALL_PROXY are NOT honored during benchmarks."""
    source = SCRIPT.read_text(encoding="utf-8")
    factory_idx = source.find("_build_async_probe_factory")
    assert factory_idx >= 0
    # Look at the factory body: the AsyncClient kwargs must include trust_env=False.
    factory_block = source[factory_idx:factory_idx + 1500]
    assert "trust_env=False" in factory_block


def test_gate_uses_asyncio_gather_with_semaphore_for_concurrency():
    """Concurrent batches must drive the gate via asyncio.gather and a
    Semaphore sized to the configured concurrency. ThreadPoolExecutor is
    forbidden because it forces a new thread per concurrent probe and the
    connection pool it would use (the stdlib http.client) does not survive
    across threads in a stable, keep-alive-aware way."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source, (
        "ThreadPoolExecutor must NOT appear in the gate — it builds a new "
        "connection per probe and cannot amortize the 3-way handshake over "
        "an HTTP/1.1 keep-alive pool."
    )
    assert "asyncio.gather" in source, "asyncio.gather must drive concurrency"
    assert "asyncio.Semaphore(concurrency)" in source, (
        "concurrency must be enforced via asyncio.Semaphore"
    )


def test_gate_default_url_points_at_loopback():
    """App performance gate must default to 127.0.0.1:7788 so nginx/TLS are
    not part of the in-process benchmark. A separate proxy_https_p95 SLO
    can sit on top if needed."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'DEFAULT_URL = "http://127.0.0.1:7788/api/recall-test"' in source, (
        "DEFAULT_URL must default to loopback, not the public domain"
    )


def test_gate_concurrency_values_are_one_and_five():
    """Per D-version the gate sweeps concurrency 1 (smoke) and 5 (load)."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for concurrency in (1, 5)" in source, (
        "gate must sweep concurrency 1 and 5 only"
    )


def test_gate_percentile_returns_expected_p95_for_known_sample():
    """Percentile math must match the p95 threshold semantics used by the
    gate's pass/fail check (HTTP p95 target 300ms, server p95 target 150ms)."""
    module = _module()
    # p95 of [1..100] with inclusive linear interpolation should land at 95.05.
    sample = list(range(1, 101))
    p95 = module.percentile(sample, 95)
    assert 95.0 <= p95 <= 95.1, p95
    # p95 of [1..20] must be 19.05 (linear interpolation).
    sample = list(range(1, 21))
    p95 = module.percentile(sample, 95)
    assert 19.0 <= p95 <= 19.1, p95


def test_gate_returns_nonzero_exit_on_failure(tmp_path):
    """If the gate is invoked against a URL that always returns errors, the
    process exit code must be non-zero (i.e. CI fails the job)."""
    import os
    proc = subprocess.run(
        [
            "python",
            str(SCRIPT),
            "--url", "http://127.0.0.1:1/api/recall-test",  # connection refused
            "--warmup", "1",
            "--measured", "1",
            "--out", str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MEMORY_OS_TOKEN": "***"},
        timeout=120,
    )
    assert proc.returncode != 0, (
        f"gate returned 0 even though the endpoint is unreachable\n"
        f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
    )


def test_run_batch_returns_count_samples_for_each_concurrency():
    """Sanity check on the new asyncio-driven run_batch: it must produce one
    Probe per request (no duplicates, no drops) regardless of concurrency.
    The ``client`` and ``one_coro`` arguments are passed in (not constructed)
    so the gate must share a single client across every run_batch call."""
    module = _module()

    async def fake_one(query: str, mode: str, limit: int) -> module.Probe:
        return module.Probe(1.0, 200, 0.5, False, False, None)

    # build the same shape the gate does so we can pass it through.
    client, one_coro = module._build_async_probe_factory(
        url="http://127.0.0.1:1/api/recall-test", token="dummy",
        max_keepalive=5, max_connections=5,
    )

    async def main():
        return await module._run_async_batch(
            client=client, one_coro=fake_one, queries=["q1", "q2", "q3"],
            mode="keyword", concurrency=2, count=5, limit=3,
        )

    result = asyncio.run(main())
    assert len(result) == 5
    assert all(p.status == 200 for p in result)


def test_run_gate_builds_exactly_one_client_for_the_whole_run():
    """The gate must construct ONE httpx.AsyncClient per gate run (not per
    run_batch) so the keep-alive pool survives the whole sweep.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    run_gate_idx = source.find("def run_gate")
    assert run_gate_idx >= 0
    run_gate_body = source[run_gate_idx:]
    # The single client is constructed once near the top of run_gate.
    assert run_gate_body.count("_build_async_probe_factory(") == 1, (
        "_build_async_probe_factory must be called exactly once in run_gate"
    )
    # And the gate drives every batch through ``_run_async_batch`` with the
    # shared client inside one persistent event loop.
    assert "_run_async_batch(" in run_gate_body, (
        "run_gate must drive batches through _run_async_batch with shared client"
    )
    assert "asyncio.run(drive_gate" in run_gate_body, (
        "run_gate must run all batches inside one asyncio.run() so the loop "
        "is not torn down between batches (httpx raises 'Event loop is closed' "
        "if its transport outlives its loop)."
    )


def test_run_gate_does_not_torn_down_event_loop_between_batches():
    """Calling asyncio.run() per run_batch tears down the loop and breaks the
    httpx transport. Assert that run_gate has exactly one asyncio.run call
    that wraps the entire gate drive."""
    source = SCRIPT.read_text(encoding="utf-8")
    run_gate_idx = source.find("def run_gate")
    assert run_gate_idx >= 0
    # Stop at the next top-level def to avoid counting main().
    next_def = source.find("\ndef ", run_gate_idx + 1)
    run_gate_body = source[run_gate_idx:next_def if next_def > 0 else None]
    # Only count real calls, not docstring/comment occurrences.
    real_calls = [
        line for line in run_gate_body.splitlines()
        if "asyncio.run(" in line and not line.strip().startswith(("* ", "# ", '"""'))
    ]
    assert len(real_calls) == 1, (
        f"run_gate must call asyncio.run exactly once (to wrap drive_gate), "
        f"not once per batch. Found: {real_calls}"
    )
    assert "asyncio.run(drive_gate" in run_gate_body
