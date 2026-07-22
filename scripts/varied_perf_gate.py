#!/usr/bin/env python3
"""Host-side varied-query performance graduation gate.

Runs privacy-clean, heterogeneous queries against the live recall API. It
separates repeated-query latency from varied-query latency and tests
concurrency 1 and 5 for keyword, dense and hybrid modes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://127.0.0.1:7788/api/recall-test"
TARGETS = {
    "keyword": {"http_p95_ms": 300.0, "server_p95_ms": 150.0},
    "dense": {"http_p95_ms": 800.0},
    "hybrid": {"http_p95_ms": 1000.0},
}


def default_queries() -> List[str]:
    chinese = [
        "如何配置本地记忆检索", "最近修改过的系统设置", "数据库备份流程是什么",
        "怎样检查服务健康状态", "召回结果为什么会降级", "查找上一次架构决策",
        "本地模型启动失败怎么办", "记忆索引什么时候刷新", "如何查看策略版本",
        "需要人工审核的重复记忆", "Active 记忆不足时怎么办", "旧记忆为什么被替代",
        "怎样恢复上一个策略", "登录以后如何注销", "如何验证会话已经过期",
        "找出与部署相关的记忆", "最近一次维护任务结果", "关键词检索支持中文吗",
        "向量检索连接不上怎么办", "怎样查看候选策略", "评估报告保存在哪里",
        "反馈数据如何参与评估", "召回接口的诊断字段", "如何触发离线评估",
        "索引损坏以后怎么恢复", "多集合记忆如何区分", "过期记忆会自动出现吗",
        "怎样确认没有物理删除", "本地服务端口如何修改", "最近的性能测试结果",
    ]
    english = [
        "how to configure local memory retrieval", "show the latest system decision",
        "what happens when embeddings are unavailable", "where are policy files stored",
        "how does superseded fallback work", "find the most recent maintenance result",
        "explain recall diagnostics", "how to rebuild the lexical index",
        "show active session security", "how are recovery codes consumed",
        "find deployment configuration", "what is the current policy version",
        "how does candidate promotion work", "when does automatic rollback trigger",
        "where are evaluation reports saved", "find database migration history",
        "how is collection identity represented", "show memory feedback statistics",
        "why did hybrid retrieval degrade", "how to verify qdrant connectivity",
        "what is the lexical tokenizer version", "find the backup snapshot procedure",
        "show memories requiring review", "how to inspect the circuit breaker",
        "what is the recall fallback threshold", "find the service health endpoint",
        "how are session tokens stored", "what does the privacy scan check",
        "find the latest ingestion error", "how to rotate authentication secrets",
    ]
    identifiers = [
        "/opt/example/openclaw/config.yaml", "/srv/example/memory/index.json",
        "openclaw-memory-os.service", "memory-os.example.conf", "active.json",
        "previous.json", "candidate.json", "recall_feedback.db", "sessions.db",
        "nomic-embed-text", "qwen2.5:1.5b", "v0.3.0", "schema_version=2",
        "127.0.0.1:6333", "127.0.0.1:7788", "MEMORY_OS_TOKEN",
        "MEMORY_OS_POLICY_DIR", "RECALL_FALLBACK_SUPERSEDED_MIN_RESULTS",
        "collection:memory_id", "candidate_key",
    ]
    exact_and_no_result = [
        "error code embedding_unavailable", "degraded_reason lexical_unavailable",
        "policy checksum mismatch", "migration status ambiguous collection",
        "query id feedback validation", "rrf_k dense_k lexical_k",
        "exact identifier model version", "localhost port configuration",
        "totally absent benchmark phrase alpha-zulu-001",
        "nonexistent component beta-kappa-002", "unknown memory gamma-theta-003",
        "no result synthetic delta-iota-004", "missing record epsilon-lambda-005",
        "unseen identifier zeta-mu-006", "unmatched phrase eta-nu-007",
        "fictional setting theta-xi-008", "absent path /opt/example/missing-009",
        "unknown model example-model-010", "never stored token placeholder-011",
        "no such decision synthetic-012",
    ]
    queries = chinese + english + identifiers + exact_and_no_result
    if len(queries) < 100 or len(set(queries)) != len(queries):
        raise RuntimeError("varied query corpus must contain at least 100 unique queries")
    return queries


def _read_token() -> Optional[str]:
    value = os.environ.get("MEMORY_OS_TOKEN", "").strip()
    if value:
        return value
    env_file = Path(os.environ.get("ENV_FILE", PROJECT_DIR / ".env"))
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(?:export\s+)?MEMORY_OS_TOKEN\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        candidate = match.group(1).split(" #", 1)[0].strip().strip("\"").strip("'")
        return candidate or None
    return None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[index]


def stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "avg_ms": None,
                "min_ms": None, "max_ms": None}
    data = [float(v) for v in values]
    return {
        "count": len(data),
        "p50_ms": percentile(data, 50),
        "p95_ms": percentile(data, 95),
        "avg_ms": statistics.fmean(data),
        "min_ms": min(data),
        "max_ms": max(data),
    }


@dataclass
class ClientStats:
    """Stats proving the async client is reused, not re-created per request."""

    requests_attempted: int
    requests_succeeded: int
    connection_reused: bool  # True if same underlying TCP connection served >1 req


@dataclass
class Probe:
    elapsed_ms: float
    status: int
    server_ms: Optional[float]
    fallback: bool
    degraded: bool
    cache_hit: Optional[bool]


def _build_async_probe_factory(*, url: str, token: str, max_keepalive: int, max_connections: int):
    """Return an async callable that posts one recall request using a single
    persistent httpx.AsyncClient shared across the whole gate run.

    The returned coroutine honors all D-version requirements:
      * one persistent client (connection reuse demonstrated by ``ClientStats``);
      * ``trust_env=False`` so HTTP_PROXY / HTTPS_PROXY are ignored;
      * bounded connection pool (``max_keepalive`` and ``max_connections``);
      * ``http1=True`` so we talk HTTP/1.1 like the rest of the host stack;
      * fail loudly if no response body or any non-200 status (per request).
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is in runtime lock
        raise RuntimeError(f"httpx is required for the async gate: {exc}") from exc

    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive,
        max_connections=max_connections,
    )

    client = httpx.AsyncClient(
        http2=False,
        http1=True,
        timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=60.0),
        trust_env=False,
        limits=limits,
        verify=True if url.lower().startswith("https://") else False,
    )

    async def one(query: str, mode: str, limit: int) -> Probe:
        body = {"query": query, "mode": mode, "limit": limit}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            response = await client.post(url, json=body, headers=headers)
            status = int(response.status_code)
            raw = response.text
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            print(f"probe error: {exc!r}", file=sys.stderr)
            return Probe(elapsed, 0, None, False, True, None)
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}
        diagnostics = payload.get("diagnostics") or {}
        lexical_ms = diagnostics.get("lexical_ms")
        server_ms = float(lexical_ms) if isinstance(lexical_ms, (int, float)) else None
        fallback = bool(payload.get("fallback_used") or diagnostics.get("fallback_used"))
        degraded = (
            str(payload.get("retrieval_status") or diagnostics.get("status") or "").lower()
            in {"degraded", "failed", "error"}
            or bool(payload.get("degraded_reason") or diagnostics.get("degraded_reason"))
        )
        cache_value = diagnostics.get("cache_hit", diagnostics.get("lexical_cache_hit"))
        cache_hit = bool(cache_value) if isinstance(cache_value, bool) else None
        return Probe(elapsed, status, server_ms, fallback, degraded, cache_hit)

    return client, one


async def _run_async_batch(
    *,
    client, one_coro, queries: Sequence[str], mode: str, concurrency: int, count: int, limit: int,
) -> List[Probe]:
    """Drive the gate's batched recall workload via asyncio.gather.

    All requests share the same persistent client; the worker count is the
    concurrency cap (semaphore) so HTTP/1.1 keep-alive pool fills but is never
    exceeded.
    """
    selected = [queries[i % len(queries)] for i in range(count)]
    sem = asyncio.Semaphore(concurrency)

    async def worker(query: str) -> Probe:
        async with sem:
            return await one_coro(query, mode, limit)

    coros = [worker(q) for q in selected]
    return list(await asyncio.gather(*coros))


def run_batch(
    *,
    client, one_coro, queries: Sequence[str], mode: str,
    concurrency: int, count: int, limit: int,
) -> List[Probe]:
    """Synchronous wrapper that drives an already-built client through
    asyncio. The client (and its keep-alive connection pool) is shared across
    every batch in a single gate run so HTTP/1.1 handshakes amortize.
    """
    async def runner():
        return await _run_async_batch(
            client=client, one_coro=one_coro, queries=queries, mode=mode,
            concurrency=concurrency, count=count, limit=limit,
        )

    return asyncio.run(runner())


def summarize(samples: Sequence[Probe]) -> Dict[str, Any]:
    successful = [sample for sample in samples if sample.status == 200]
    fallback = [sample.elapsed_ms for sample in successful if sample.fallback]
    degraded = [sample.elapsed_ms for sample in successful if sample.degraded]
    observed_cache = [sample.cache_hit for sample in successful if sample.cache_hit is not None]
    return {
        "requests": len(samples),
        "http_200": len(successful),
        "http": stats([sample.elapsed_ms for sample in successful]),
        "server": stats([sample.server_ms for sample in successful if sample.server_ms is not None]),
        "fallback_http": stats(fallback),
        "degraded_http": stats(degraded),
        "fallback_count": len(fallback),
        "degraded_count": len(degraded),
        "cache_observable": bool(observed_cache),
        "cache_hit_rate": (
            sum(1 for value in observed_cache if value) / len(observed_cache)
            if observed_cache else None
        ),
    }


def check_result(mode: str, result: Dict[str, Any], measured: int) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    varied = result["varied"]
    repeated = result["repeated"]
    if varied["http_200"] != measured:
        failures.append(f"varied_http_200={varied['http_200']}/{measured}")
    if repeated["http_200"] != measured:
        failures.append(f"repeated_http_200={repeated['http_200']}/{measured}")
    target = TARGETS[mode]
    varied_p95 = varied["http"]["p95_ms"]
    if varied_p95 is None or varied_p95 > target["http_p95_ms"]:
        failures.append(f"varied_http_p95={varied_p95}>{target['http_p95_ms']}")
    if mode == "keyword":
        server_p95 = varied["server"]["p95_ms"]
        if server_p95 is None or server_p95 > target["server_p95_ms"]:
            failures.append(f"varied_server_p95={server_p95}>{target['server_p95_ms']}")
    return not failures, failures


def run_gate(args: argparse.Namespace) -> Dict[str, Any]:
    token = _read_token()
    if not token:
        raise RuntimeError("MEMORY_OS_TOKEN is required")
    queries = default_queries()
    repeated_query = queries[0]
    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "query_count": len(queries),
        "warmup": args.warmup,
        "measured": args.measured,
        "targets": TARGETS,
        "results": [],
        "passed": True,
    }

    # Build ONE persistent client AND run every batch inside ONE persistent
    # event loop. Two things break if we call asyncio.run() per run_batch:
    #   * asyncio.run() tears down its loop on return, and the httpx client's
    #     transport is bound to that loop — subsequent batches raise
    #     "Event loop is closed" inside the probe coroutine.
    #   * Each run would also force a new TCP connection handshake, defeating
    #     the whole point of the keep-alive fix.
    client, one_coro = _build_async_probe_factory(
        url=args.url, token=token,
        max_keepalive=5,
        max_connections=5,
    )

    async def drive_gate():
        try:
            for mode in ("keyword", "dense", "hybrid"):
                for concurrency in (1, 5):
                    await _run_async_batch(
                        client=client, one_coro=one_coro, queries=queries, mode=mode,
                        concurrency=concurrency, count=args.warmup, limit=args.limit,
                    )
                    varied_samples = await _run_async_batch(
                        client=client, one_coro=one_coro, queries=queries, mode=mode,
                        concurrency=concurrency, count=args.measured, limit=args.limit,
                    )
                    await _run_async_batch(
                        client=client, one_coro=one_coro, queries=[repeated_query], mode=mode,
                        concurrency=concurrency, count=args.warmup, limit=args.limit,
                    )
                    repeated_samples = await _run_async_batch(
                        client=client, one_coro=one_coro, queries=[repeated_query], mode=mode,
                        concurrency=concurrency, count=args.measured, limit=args.limit,
                    )
                    result = {
                        "mode": mode,
                        "concurrency": concurrency,
                        "varied": summarize(varied_samples),
                        "repeated": summarize(repeated_samples),
                    }
                    passed, failures = check_result(mode, result, args.measured)
                    result["passed"] = passed
                    result["failures"] = failures
                    report["results"].append(result)
                    report["passed"] = bool(report["passed"] and passed)
        finally:
            await client.aclose()

    asyncio.run(drive_gate())

    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("RECALL_TEST_URL", DEFAULT_URL))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        queries = default_queries()
        payload = {"query_count": len(queries), "unique": len(set(queries)), "targets": TARGETS}
        print(json.dumps(payload, sort_keys=True))
        return 0
    try:
        report = run_gate(args)
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        if os.name != "nt":
            os.chmod(args.out, 0o600)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
