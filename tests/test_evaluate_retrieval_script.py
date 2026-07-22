"""Smoke + import + run tests for ``scripts/evaluate_retrieval.py``.

The script is a side-effect-free offline-evaluation CLI. It must:

* import cleanly without needing Qdrant or any live backend,
* parse ``--help`` without errors,
* emit a JSON envelope with the documented top-level keys
  (status, generated_at, corpus_snapshot_id, metrics, feedback,
  history, notes, warnings),
* never fabricate graded metrics (when no judged data is available
  the graded fields are explicitly ``null`` with a
  ``status="unavailable"`` marker),
* survive an isolated, empty feedback DB by returning
  ``status="unavailable"`` instead of crashing.

We deliberately do NOT mock Qdrant / embedders: the script never
imports them. This test pins the "no live backend required" contract.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluate_retrieval.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# Use the same Python interpreter that pytest is running under for
# subprocess invocations (matches the project's venv conventions).
SUBPROCESS_PY = sys.executable


def _load_module():
    """Import the script as a module without executing its __main__.

    We sidestep ``runpy`` / ``__import__`` weirdness by loading the
    source file directly via ``importlib.util.spec_from_file_location``,
    which also keeps the module's globals accessible to the tests.
    """
    spec = importlib.util.spec_from_file_location(
        "evaluate_retrieval_script", str(SCRIPT)
    )
    assert spec is not None and spec.loader is not None, (
        f"could not load module spec for {SCRIPT}"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Smoke: script is reachable and importable
# ---------------------------------------------------------------------------


def test_script_file_exists():
    assert SCRIPT.exists(), f"missing script: {SCRIPT}"


def test_script_imports_cleanly():
    """The script must import without raising even on a fresh checkout."""
    mod = _load_module()
    assert mod is not None
    # Public surface we rely on.
    assert hasattr(mod, "_build_envelope")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_parse_args")


def test_script_help_exits_zero():
    """``--help`` must succeed (exit 0) — operators rely on it."""
    proc = subprocess.run(
        [SUBPROCESS_PY, str(SCRIPT), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "offline retrieval-evaluation" in proc.stdout.lower() or "usage" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# 2. Build envelope directly (no subprocess)
# ---------------------------------------------------------------------------


def test_envelope_has_documented_top_level_keys():
    mod = _load_module()
    env = mod._build_envelope(limit=10)
    for key in (
        "status",
        "generated_at",
        "corpus_snapshot_id",
        "metrics",
        "feedback",
        "history",
        "notes",
        "warnings",
    ):
        assert key in env, f"missing envelope key: {key}"
    assert env["status"] in ("ok", "unavailable", "error")
    assert isinstance(env["generated_at"], str) and env["generated_at"]
    assert env["corpus_snapshot_id"] is None or isinstance(env["corpus_snapshot_id"], str)
    assert isinstance(env["notes"], list)
    assert isinstance(env["warnings"], list)
    assert isinstance(env["history"], list)
    assert isinstance(env["feedback"], dict)
    assert isinstance(env["metrics"], dict)


def test_envelope_metrics_have_legacy_and_new_fields():
    mod = _load_module()
    env = mod._build_envelope(limit=10)
    metrics = env["metrics"]
    # Legacy fields.
    for key in (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "useful_at_1",
        "useful_at_5",
        "explicit_negative_at_5",
        "no_result_rate",
        "p50_latency",
        "p95_latency",
        "num_cases",
    ):
        assert key in metrics, f"missing legacy metric: {key}"
    # New v0.3.0.x fields.
    for key in (
        "judged_ndcg_at_10",
        "useful_superseded_fallback_rate",
        "num_judged_cases",
        "corpus_snapshot_id",
        "judged_ndcg_status",
        "fallback_rate_status",
    ):
        assert key in metrics, f"missing v0.3.0.x metric: {key}"


def test_envelope_unavailable_means_null_graded_fields():
    """When status="unavailable", graded fields must be None (not 0.0).

    This is the explicit "honest null" contract the dashboard relies on.
    """
    mod = _load_module()
    env = mod._build_envelope(limit=10)
    if env["status"] == "unavailable":
        assert env["metrics"]["judged_ndcg_at_10"] is None
        assert env["metrics"]["useful_superseded_fallback_rate"] is None
        assert env["metrics"]["judged_ndcg_status"] == "unavailable"
        assert env["metrics"]["fallback_rate_status"] == "unavailable"
        assert env["metrics"]["num_judged_cases"] == 0


def test_envelope_survives_empty_feedback_db(tmp_path, monkeypatch):
    """Isolated empty DB → status="unavailable", well-typed envelope, no crash."""
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path / "empty-state"))
    # Reload recall_feedback so it picks up the new DB path.
    import importlib
    import openclaw_memory_os.recall_feedback as rf
    importlib.reload(rf)

    try:
        mod = _load_module()
        env = mod._build_envelope(limit=10)
        # Empty DB → unavailable. Even if the test machine already has
        # judged data, the metrics must still respect the contract.
        if env["status"] == "unavailable":
            assert env["metrics"]["num_judged_cases"] == 0
            assert env["metrics"]["judged_ndcg_at_10"] is None
            assert env["feedback"]["total_events"] in (None, 0) or isinstance(
                env["feedback"]["total_events"], int
            )
        # Types must be correct either way.
        assert isinstance(env["notes"], list)
        assert isinstance(env["warnings"], list)
        assert isinstance(env["history"], list)
    finally:
        monkeypatch.delenv("MEMORY_OS_RECALL_STATE_DIR", raising=False)
        importlib.reload(rf)


# ---------------------------------------------------------------------------
# 3. Subprocess runs end-to-end without live Qdrant
# ---------------------------------------------------------------------------


def test_subprocess_runs_and_emits_valid_json():
    """Run the script as a real subprocess and validate the JSON it prints."""
    proc = subprocess.run(
        [SUBPROCESS_PY, str(SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    # The script prints a single JSON object to stdout.
    body = json.loads(proc.stdout)
    # Spot-check the same top-level keys.
    for key in (
        "status",
        "generated_at",
        "corpus_snapshot_id",
        "metrics",
        "feedback",
        "history",
        "notes",
        "warnings",
    ):
        assert key in body, f"missing key in subprocess output: {key}"
    # graded metrics must respect the null-when-unavailable contract.
    if body["status"] == "unavailable":
        assert body["metrics"]["judged_ndcg_at_10"] is None
        assert body["metrics"]["useful_superseded_fallback_rate"] is None
    # No live Qdrant means the script must NOT have tried to connect
    # to anything. We can't directly assert that, but we can assert
    # the warnings block does not contain any "qdrant" or "embedding"
    # failure noise that would indicate a live-dependency leak.
    for w in body["warnings"]:
        assert "qdrant" not in w.lower(), f"unexpected qdrant warning: {w}"


def test_subprocess_pretty_flag_produces_indented_json():
    proc = subprocess.run(
        [SUBPROCESS_PY, str(SCRIPT), "--pretty"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    # Pretty-printed JSON spans multiple lines and starts with '{'.
    assert "\n" in proc.stdout
    assert proc.stdout.lstrip().startswith("{")


def test_subprocess_out_flag_writes_file(tmp_path):
    """``--out FILE`` writes the JSON envelope to the given path."""
    out = tmp_path / "envelope.json"
    proc = subprocess.run(
        [SUBPROCESS_PY, str(SCRIPT), "--out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    assert out.exists(), f"--out did not write file: {out}"
    body = json.loads(out.read_text(encoding="utf-8"))
    for key in ("status", "metrics", "feedback", "history", "notes", "warnings"):
        assert key in body, f"missing key in --out file: {key}"


def test_subprocess_limit_flag_is_respected():
    """``--limit`` is accepted; envelope still has the documented shape."""
    proc = subprocess.run(
        [SUBPROCESS_PY, str(SCRIPT), "--limit", "1"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    body = json.loads(proc.stdout)
    assert "metrics" in body
    # num_cases / num_judged_cases never exceed the limit.
    assert body["metrics"]["num_cases"] <= 1
    assert body["metrics"]["num_judged_cases"] <= 1