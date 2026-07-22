#!/usr/bin/env python3
"""Wave 5 (v0.3.0): 120-query synthetic feedback loop.

Verifies end-to-end that the G6.5 / G6.8 promotion + circuit breaker
machinery actually fires when given a realistic stream of feedback:

  - Cycles 1..4 are kicked off every 30 queries (30/60/90/120).
  - The first cycle is expected to either be ``skipped`` (cold start:
    <30 judged queries) or ``shadow`` (cold start cleared after the
    first 30). Cycles 2..4 should at least return ``ok`` or
    ``shadow``; promotion / rollback / circuit-breaker / cooldown
    behaviour is observed opportunistically (some runs may trigger
    rollback depending on how the candidate pool shakes out).
  - The script prints a single JSON envelope at the end:

        {"cycles": int,
         "promoted": [list of cycle indices that returned promoted],
         "rolled_back": [list of cycle indices that returned rolled_back],
         "breaker_open_count": int,
         "cooldown_hits": int,
         "status_breakdown": {"ok": N, "shadow": N, ...}}

  - All feedback / recall rows land in a tmp SQLite DB (path chosen
    via ``MEMORY_OS_RECALL_STATE_DIR`` + tmp dir) so the live OS
    audit DB is never touched.

Backends
--------

The script prefers the live Qdrant collection when ``QDRANT_URL`` is
reachable. When the live Qdrant is unreachable OR the operator
explicitly forces the sample path (``--force-sample`` or
``MEMORY_OS_SAMPLE_PATH`` set), the loop falls back to the bundled
:class:`SampleBackend` with a 3-doc / 3-query corpus so the test
remains runnable in any environment.

Usage::

    scripts/synth_feedback_loop.py                 # default
    scripts/synth_feedback_loop.py --queries 5 --repeats 24
    scripts/synth_feedback_loop.py --cycle-every 30 --pos-ratio 0.8

Exit codes:
  0  - loop completed (regardless of how many cycles promoted).
  2  - the script could not initialise any backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("synth_feedback_loop")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


#: Default 5 queries. Each is a short keyword phrase matched 1:1 against
#: one memory in tests/data/sample_synth.json. The list is duplicated
#: ``repeats`` times to make 120 events.
DEFAULT_QUERIES = [
    "gorilla deploy latency",
    "lemur consolidation dedup",
    "otter observability prometheus",
    "gorilla load balancer rolling",
    "lemur near-duplicate canonical",
]

#: Tmp SQLite DB root (per-run). The DB is created fresh so runs are
#: fully isolated from the live OS audit log.
DEFAULT_STATE_DIR_NAME = "synth-feedback-loop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_qdrant_reachable(url: str, timeout: float = 0.5) -> bool:
    """Cheap TCP probe for ``QDRANT_URL`` (host:port)."""
    parsed_url = url.replace("http://", "").replace("https://", "")
    if "/" in parsed_url:
        parsed_url = parsed_url.split("/", 1)[0]
    if ":" not in parsed_url:
        return False
    host, port = parsed_url.split(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_backend(
    args: argparse.Namespace,
) -> Tuple[str, Optional[Path]]:
    """Return ``(mode, sample_path)``.

    ``mode`` is one of ``"qdrant"`` or ``"sample"``. ``sample_path`` is
    set when ``mode == "sample"``.
    """
    if args.force_sample or os.environ.get("MEMORY_OS_SAMPLE_PATH"):
        sample_path = Path(
            os.environ.get("MEMORY_OS_SAMPLE_PATH")
            or (PROJECT_DIR / "tests" / "data" / "sample_synth.json")
        )
        return "sample", sample_path
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url and _is_qdrant_reachable(qdrant_url):
        return "qdrant", None
    # Fall back to bundled sample.
    sample_path = PROJECT_DIR / "tests" / "data" / "sample_synth.json"
    if not sample_path.exists():
        sample_path = PROJECT_DIR / "data" / "sample_memories.json"
    return "sample", sample_path


def _setup_state_dir() -> Path:
    """Return a per-run tmp state dir and point env at it.

    Honours ``MEMORY_OS_RECALL_STATE_DIR`` from the caller when set;
    otherwise creates a fresh tmpdir. Returns the resolved path.
    """
    caller_dir = os.environ.get("MEMORY_OS_RECALL_STATE_DIR")
    if caller_dir:
        root = Path(caller_dir) / DEFAULT_STATE_DIR_NAME
    else:
        root = Path(tempfile.mkdtemp(prefix="openclaw-synth-")) / DEFAULT_STATE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORY_OS_RECALL_STATE_DIR"] = str(root)
    # Pin the policy path so the script doesn't accidentally overwrite
    # an active policy elsewhere on disk.
    os.environ["MEMORY_OS_POLICY_PATH"] = str(root / "policy.json")
    return root


# ---------------------------------------------------------------------------
# Feedback / recall row injection
# ---------------------------------------------------------------------------


def _inject_sample_feedback(
    query_id: str,
    query_text: str,
    *,
    useful: bool,
    backend_kind: str,
    sample_path: Optional[Path],
) -> None:
    """Write one recall_runs row + recall_results row + feedback_event.

    The candidate_key matches what the SampleBackend's lexical search
    returns for the given query (substring match on the memory text),
    so the eval pipeline sees consistent relevance judgements and the
    per-candidate rank_fn closure actually picks up a different top-1
    for the perturbed candidate policies.
    """
    from openclaw_memory_os.recall_feedback import (
        record_feedback_v030,
        record_recall_result,
        record_recall_run,
    )

    # Resolve the deterministic top-1 candidate_key for this query.
    candidate_key = _candidate_key_for_query(query_text, backend_kind, sample_path)

    # recall_runs (always needed — eval joins on it).
    record_recall_run(
        query_id,
        query_text,
        retrieval_mode="hybrid",
        policy_version="synth",
        latency_ms=0.0,
        retrieval_status="ok",
        degraded_reason=None,
        fallback_used=False,
    )

    # recall_results — at least one row so the candidate_key exists
    # in the join. The eval pipeline does NOT actually consult
    # recall_results for relevance judgements (those come from
    # feedback_events), but we still seed the table for parity with
    # a real recall trace.
    record_recall_result(
        query_id,
        candidate_key,
        memory_id=candidate_key.split(":", 1)[1] if ":" in candidate_key else candidate_key,
        collection="sample",
        rank=1,
        status="active",
        final_score=0.95,
    )

    # feedback_events — the actual judgement.
    record_feedback_v030(query_id, candidate_key, useful)


def _candidate_key_for_query(
    query_text: str,
    backend_kind: str,
    sample_path: Optional[Path],
) -> str:
    """Return the deterministic top-1 ``candidate_key`` for ``query_text``.

    For the SampleBackend we scan the corpus JSON, do the same
    case-insensitive substring match the backend uses, and emit the
    first hit as ``"sample:<memory_id>"``. For Qdrant we just emit a
    stable ``"qdrant:syn-mem-NNN"`` placeholder because we don't try
    to reach into Qdrant from inside this helper.
    """
    if backend_kind == "qdrant":
        # We don't have a live RetrievalEngine here; the live Qdrant
        # branch is best-effort: use a stable placeholder so the
        # feedback_events schema is satisfied.
        return "qdrant:syn-mem-001"
    assert sample_path is not None
    if not sample_path.exists():
        return "sample:syn-mem-001"
    try:
        raw = json.loads(sample_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover — defensive
        return "sample:syn-mem-001"
    items = raw.get("memories", []) if isinstance(raw, dict) else raw
    q_low = query_text.lower()
    for m in items:
        if q_low in (m.get("text", "") or "").lower():
            return f"sample:{m['id']}"
    # Fallback: first memory.
    if items:
        return f"sample:{items[0]['id']}"
    return "sample:syn-mem-001"


def _inject_negative_feedback(
    query_id: str,
    query_text: str,
    *,
    backend_kind: str,
    sample_path: Optional[Path],
) -> None:
    """Emit one negative-only feedback event for ``query_id``.

    Picks a candidate_key that is *different* from the top-1
    candidate, so the eval pipeline's ``negatives`` set is
    populated. We just use a ``syn-neg-XXX`` suffix on the memory
    id so the candidate_key is clearly marked as a known negative.

    G5.1 (strong validation): we now seed a ``recall_results`` row
    for the negative key before recording the feedback, because
    ``record_feedback_v030`` refuses to record feedback for a
    candidate that was never returned.
    """
    from openclaw_memory_os.recall_feedback import (
        record_feedback_v030,
        record_recall_result,
    )

    if backend_kind == "qdrant":
        negative_key = "qdrant:syn-neg-001"
    else:
        assert sample_path is not None
        try:
            raw = json.loads(sample_path.read_text(encoding="utf-8"))
            items = raw.get("memories", []) if isinstance(raw, dict) else raw
        except Exception:  # pragma: no cover — defensive
            items = []
        if items:
            # Pick a memory that is NOT the top-1 hit.
            top = _candidate_key_for_query(query_text, backend_kind, sample_path)
            top_id = top.split(":", 1)[1] if ":" in top else top
            for m in items:
                if m.get("id") != top_id:
                    negative_key = f"sample:neg-{m['id']}"
                    break
            else:
                negative_key = "sample:neg-placeholder"
        else:
            negative_key = "sample:neg-placeholder"

    # Seed a recall_results row so the strong validation in
    # ``record_feedback_v030`` accepts the negative candidate_key.
    try:
        record_recall_result(
            query_id,
            negative_key,
            memory_id=(
                negative_key.split(":", 1)[1] if ":" in negative_key else negative_key
            ),
            collection="sample" if backend_kind != "qdrant" else "qdrant",
            rank=99,  # low rank; this is a negative, not a real hit
            status="active",
            final_score=0.01,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("recall_results seed for negative_key failed: %s", exc)

    record_feedback_v030(query_id, negative_key, False)


# ---------------------------------------------------------------------------
# Evolution cycle driver
# ---------------------------------------------------------------------------


def _run_evolution_cycle(env_extra: Dict[str, str]) -> Dict[str, Any]:
    """Invoke ``scripts/run_evolution_cycle.py`` and parse its JSON.

    The runner prints a single JSON envelope on stdout (per its
    contract: ``{"status": ..., ...}``). We capture stdout/stderr,
    return code, and elapsed wall time so the caller can record
    each cycle's outcome.
    """
    env = os.environ.copy()
    env.update(env_extra)
    script = PROJECT_DIR / "scripts" / "run_evolution_cycle.py"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "reason": "timeout",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out = proc.stdout.strip() or "{}"
    try:
        payload = json.loads(out.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {
            "status": "error",
            "reason": f"unparseable_stdout: {out[:200]!r}",
            "stderr": proc.stderr[-200:] if proc.stderr else "",
        }
    if isinstance(payload, dict):
        payload.setdefault("elapsed_ms", round(elapsed_ms, 3))
        payload.setdefault("returncode", proc.returncode)
    return payload


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=5,
                        help="Number of distinct queries to cycle (default 5).")
    parser.add_argument("--repeats", type=int, default=24,
                        help="Repeats per query (default 24 → 120 total).")
    parser.add_argument("--cycle-every", type=int, default=30,
                        help="Run an evolution cycle every N queries (default 30).")
    parser.add_argument("--pos-ratio", type=float, default=0.8,
                        help="Probability of positive feedback per event (default 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default 42).")
    parser.add_argument("--force-sample", action="store_true",
                        help="Skip the live Qdrant probe and use the SampleBackend.")
    parser.add_argument("--keep-state-dir", action="store_true",
                        help="Do not delete the tmp state dir on exit.")
    parser.add_argument("--envelope-out", type=str, default=None,
                        help="Optional path to also write the JSON envelope to (in addition to stdout).")
    args = parser.parse_args()

    if args.queries < 1 or args.repeats < 1:
        logger.error("--queries and --repeats must be >= 1")
        return 2
    total = args.queries * args.repeats
    if total < args.cycle_every:
        logger.error("total queries (%d) < cycle-every (%d); bump repeats",
                     total, args.cycle_every)
        return 2

    rng = random.Random(args.seed)
    state_dir = _setup_state_dir()
    backend_kind, sample_path = _resolve_backend(args)
    logger.info(
        "synth loop starting: %d queries x %d repeats = %d events; "
        "cycle every %d; backend=%s; state_dir=%s",
        args.queries, args.repeats, total, args.cycle_every,
        backend_kind, state_dir,
    )

    # Build the query stream.
    queries: List[str] = []
    base_queries = DEFAULT_QUERIES[: args.queries]
    while len(base_queries) < args.queries:
        base_queries.append(DEFAULT_QUERIES[len(base_queries) % len(DEFAULT_QUERIES)])
    for _ in range(args.repeats):
        queries.extend(base_queries)

    cycles: List[Dict[str, Any]] = []
    promoted: List[int] = []
    rolled_back: List[int] = []
    breaker_open_count = 0
    cooldown_hits = 0
    status_breakdown: Dict[str, int] = {}

    last_cycle_at = 0
    for idx, query in enumerate(queries, start=1):
        is_positive = rng.random() < args.pos_ratio
        qid = f"synth-{idx:04d}-{uuid.uuid4().hex[:8]}"
        try:
            # Always emit a positive feedback so the query counts
            # as "judged" (the eval pipeline requires at least one
            # useful=True entry per query_id). On the negative
            # subset we additionally emit a second event with the
            # same query_id but a *different* candidate_key to give
            # the eval pipeline real negative signal without losing
            # the positive count.
            _inject_sample_feedback(
                qid,
                query,
                useful=True,
                backend_kind=backend_kind,
                sample_path=sample_path,
            )
            if not is_positive:
                _inject_negative_feedback(
                    qid,
                    query,
                    backend_kind=backend_kind,
                    sample_path=sample_path,
                )
        except Exception as exc:  # pragma: no cover — DB write failure
            logger.warning("feedback inject failed at idx=%d: %s", idx, exc)

        if idx - last_cycle_at >= args.cycle_every:
            last_cycle_at = idx
            logger.info("[cycle %d] kicking off evolution cycle after %d queries",
                        len(cycles) + 1, idx)
            env_extra = {
                "MEMORY_OS_SAMPLE_PATH": str(sample_path) if backend_kind == "sample" else "",
            }
            # Strip when empty so subprocess sees the caller's default.
            if not env_extra["MEMORY_OS_SAMPLE_PATH"]:
                env_extra.pop("MEMORY_OS_SAMPLE_PATH")
            cycle = _run_evolution_cycle(env_extra)
            cycles.append(cycle)
            status = str(cycle.get("status", "error"))
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
            if status == "promoted":
                promoted.append(len(cycles))
            elif status == "rolled_back":
                rolled_back.append(len(cycles))
                # The cycle reports breaker open / cooldown via its
                # ``reason`` field when those gates fire.
                reason = str(cycle.get("reason", ""))
                if "cooldown" in reason.lower():
                    cooldown_hits += 1
                if "breaker" in reason.lower() or "consecutive_rollbacks" in reason.lower():
                    breaker_open_count += 1

    envelope = {
        "cycles": len(cycles),
        "promoted": promoted,
        "rolled_back": rolled_back,
        "breaker_open_count": breaker_open_count,
        "cooldown_hits": cooldown_hits,
        "status_breakdown": status_breakdown,
        "total_events": idx,
        "backend": backend_kind,
        "state_dir": str(state_dir),
        "cycles_detail": cycles,
    }
    # Final, single-line JSON to stdout so the wrapping test can
    # parse it unambiguously.
    envelope_json = json.dumps(envelope)
    print(envelope_json)
    sys.stdout.flush()
    if args.envelope_out:
        Path(args.envelope_out).write_text(envelope_json, encoding="utf-8")

    if not args.keep_state_dir and os.environ.get("KEEP_SYNTH_STATE_DIR") != "1":
        shutil.rmtree(state_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())