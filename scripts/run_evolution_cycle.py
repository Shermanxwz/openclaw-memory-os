#!/usr/bin/env python3
"""v0.3.0: Run one evolution cycle.

Called by autonomous_governance.sh after the maintenance run and
lexical refresh. Acquires the evolution lock, runs candidate
generation, evaluation, shadow, and (if conditions are met)
promotion.

The exit code is 0 for all normal outcomes (promoted, shadow,
skipped, rolled_back) and non-zero only for unexpected errors
so the parent bash script does not show a false failure on its
governance status card.

Backends / policy
-----------------

The script supports two backend configurations:

* **Live Qdrant** (``QDRANT_URL`` set): instantiate a real
  :class:`QdrantBackend` and a :class:`RetrievalEngine` on top
  of it. The dense + BM25 + RRF + feature-rerank pipeline is
  exercised end-to-end.

* **Sample / offline** (``MEMORY_OS_SAMPLE_PATH`` set, no
  QDRANT_URL): fall back to a :class:`SampleBackend` so the
  script remains runnable on a fresh checkout, in CI, and on
  the operator\'s laptop without a live Qdrant process.

The script intentionally **does NOT** call ``backend.search``
directly (B3-3). It instantiates :class:`RetrievalEngine` once
and routes every candidate-policy evaluation through it; the
``rank_fn_with_policy`` helper then applies each policy\'s
additive-rerank weights on top of the engine\'s hybrid hits so
the candidate pass ranks differently from the active pass
(B3-2 / B3-5).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _build_backend_and_store():
    """Instantiate a ``(backend, store, sample_path)`` triple.

    Returns the live Qdrant backend (or the SampleBackend fallback),
    the process-wide :class:`PolicyStore`, and the resolved sample
    data path (or ``None``).

    G6.10 (runbook): the on-disk active policy is the source of
    truth. The runner **never** calls ``store.set(...)`` to reset
    it to the shipped baseline — doing so would clobber every
    policy that evolution promoted in a previous cycle. The store
    is constructed with the standard env-var resolution
    (``MEMORY_OS_POLICY_PATH`` / ``MEMORY_OS_POLICY_DIR`` / XDG
    default) so whatever is currently on disk is what the runner
    uses. If no policy file exists yet (first-ever run), the
    store falls back to :data:`baseline_policy` automatically
    inside ``PolicyStore.__init__`` — without an explicit
    ``.set()`` call.
    """
    from openclaw_memory_os.backends import QdrantBackend, SampleBackend
    from openclaw_memory_os.policy_store import PolicyStore

    # Honour MEMORY_OS_POLICY_PATH (file path) — same resolution
    # chain as PolicyStore.__init__ itself, exposed here so an
    # operator can force a specific on-disk location without
    # touching the PolicyStore default. We only forward the file
    # path here; the directory-based / XDG-default branches are
    # already handled by PolicyStore when ``path`` is None.
    _policy_path_env = os.environ.get("MEMORY_OS_POLICY_PATH")
    if _policy_path_env:
        store = PolicyStore(path=Path(_policy_path_env))
    else:
        store = PolicyStore()
    qdrant_url = os.environ.get("QDRANT_URL")
    sample_path = os.environ.get("MEMORY_OS_SAMPLE_PATH")

    if qdrant_url:
        collection = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")
        secondary = os.environ.get("QDRANT_SECONDARY_COLLECTIONS", "")
        secondary_list = [c.strip() for c in secondary.split(",") if c.strip()]
        backend = QdrantBackend(
            qdrant_url,
            collection,
            secondary_collections=secondary_list,
        )
        return backend, store, None

    if sample_path:
        sp = Path(sample_path)
        backend = SampleBackend(sp)
        return backend, store, str(sp)

    # No live Qdrant AND no sample path: produce a tiny in-process
    # corpus (empty list) so the script doesn\'t crash with
    # ``ImportError`` for QdrantBackend. The evolution loop will
    # then no-op because no candidates match any positive case.
    empty_sample = PROJECT_DIR / "data" / "_empty_sample.json"
    empty_sample.parent.mkdir(parents=True, exist_ok=True)
    if not empty_sample.exists():
        empty_sample.write_text(json.dumps({"memories": []}), encoding="utf-8")
    backend = SampleBackend(empty_sample)
    return backend, store, str(empty_sample)


# --- Exit codes (runbook G6.9) ------------------------------------------
EXIT_OK = 0            # Planned outcome (any cycle status)
EXIT_UNEXPECTED = 1    # Unexpected runner-side exception
EXIT_CONFIG = 2        # argparse / config error


def main() -> int:
    """Run one evolution cycle and return a process exit code.

    Exit-code contract (runbook G6.9):

    * ``EXIT_OK`` (``0``) — normal completion regardless of the cycle
      verdict: ``promoted``, ``shadow``, ``skipped`` (covers both
      ``lock_held`` and ``cold_start`` reasons), ``rolled_back``,
      ``ok``, ``val_failed``, and the structured ``error`` status
      that ``run_evolution_cycle`` itself emits for internal
      evaluation failures. Bash supervisors must be able to gate on
      "non-zero means something the runner did NOT plan for",
      otherwise the governance status card flips to red on every
      weekly silent skip.
    * ``EXIT_UNEXPECTED`` (``1``) — any unexpected exception raised
      by runner-side code (backend build, policy deserialisation,
      ``RetrievalEngine`` construction, the wrapper-around-
      ``run_evolution_cycle`` plumbing). The exception is logged as a
      JSON line with the traceback and the script exits 1.
    * ``EXIT_CONFIG`` (``2``) — reserved for argparse / configuration
      errors. Currently unreachable from this script because there are
      no CLI flags; it is the documented return value a future
      ``--dry-run`` / ``--policy-path`` validation step will emit.

    The ``run_evolution_cycle`` internal ``{"status": "error", ...}``
    branch is *not* an "unexpected exception" — evolution can finish
    an evaluation pass but decide to emit ``error`` because every
    candidate failed the gate. That path stays at exit 0.
    """
    import traceback

    try:
        from openclaw_memory_os.evolution import (
            rank_fn_with_policy,
            run_evolution_cycle,
        )

        backend, store, _sample = _build_backend_and_store()
        try:
            from openclaw_memory_os.retrieval_engine import RetrievalEngine
        except Exception as exc:
            print(json.dumps({
                "status": "runner_unexpected_error",
                "reason": f"engine_import: {exc}",
            }))
            return EXIT_UNEXPECTED

        engine = RetrievalEngine(backend=backend, policy_store=store)

        # Active ranking uses the same reusable raw CandidatePool as every
        # generated candidate. Candidate policies are created inside the evolution
        # loop; this runner must not invent a fixed importance+0.05 pseudo-candidate.
        active_policy = store.get()
        active_rank_fn = rank_fn_with_policy(engine, active_policy)

        result = run_evolution_cycle(
            store,
            active_rank_fn,
            # Pass engine= so each candidate gets its OWN
            # rank_fn_with_policy(engine, cand) closure (G6.1). The
            # legacy candidate_rank_fn kwarg is omitted on purpose;
            # the new per-candidate closure is the v0.3.0 graduation
            # contract.
            engine=engine,
        )
        print(json.dumps(result))
        if str(result.get("status") or "").lower() == "error":
            return EXIT_UNEXPECTED
        return EXIT_OK
    except Exception as exc:
        # Unexpected runner-side failure (backend build, policy parse,
        # engine wiring, anything NOT inside ``run_evolution_cycle``).
        # Print the traceback as JSON so the governance dashboard can
        # surface the underlying error in a structured way.
        print(
            json.dumps(
                {
                    "status": "runner_unexpected_error",
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
