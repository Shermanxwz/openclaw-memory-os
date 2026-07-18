"""G6.1: per-candidate rank_fn closure in evolution search loop.

These tests pin the contract that every candidate policy is evaluated
with its OWN rank closure (bound to that candidate's
``importance_weight`` / ``recency_weight`` / ``feedback_weight``), not
a single shared closure that's reused across all candidates.

Without this fix, the search loop degenerates to "rank with the same
function for every candidate" — every candidate produces the same
ordering as the active policy, and the candidate-selection step
becomes a no-op (every candidate is "as good as the active").
"""

from __future__ import annotations

import inspect
from typing import Dict, List

import pytest


def _src_contains_per_candidate_loop() -> bool:
    """Read evolution.py source and assert per-candidate closure is wired up.

    The original v0.3.0.3 fix used an explicit ``for cand in candidates:``
    loop with ``rank_fn_with_policy(...)`` inside the loop body. The
    G6.2 funnel refactor moved the per-candidate evaluation into a
    helper (``_score_on``) called from a list comprehension — the
    semantic contract is preserved (each candidate still gets its
    own ``rank_fn_with_policy`` closure bound to its policy) but
    the literal ``for cand in candidates:`` header is no longer in
    the source. We therefore accept either the legacy loop pattern
    OR the modern helper-call pattern, both of which are
    functionally equivalent.
    """
    from pathlib import Path
    src = Path("openclaw_memory_os/evolution.py").read_text(encoding="utf-8")
    # Legacy pattern (v0.3.0.3): explicit for-loop header followed
    # by a window containing ``rank_fn_with_policy(``.
    legacy_idx = src.find("for cand in candidates:")
    if legacy_idx >= 0:
        window = src[legacy_idx: legacy_idx + 4096]
        return "rank_fn_with_policy(" in window
    # Modern pattern (G6.2 funnel): ``_score_on`` helper invoked
    # once per candidate via list comprehension; the helper itself
    # calls ``rank_fn_with_policy(engine, cand)`` so the per-candidate
    # closure contract is preserved.
    if "def _score_on(" in src and "rank_fn_with_policy(engine, cand)" in src:
        return True
    return False


def test_loop_calls_rank_fn_with_policy_inside_body():
    """The fix must construct a fresh rank_fn_with_policy(...) per candidate.

    Accepts either the legacy ``for cand in candidates:`` loop or
    the G6.2 funnel's ``_score_on(cand, ...)`` helper, both of
    which bind ``rank_fn_with_policy(engine, cand)`` once per
    candidate.
    """
    assert _src_contains_per_candidate_loop(), (
        "rank_fn_with_policy(engine, cand) is not invoked per candidate "
        "in evolution.py. G6.1 requires a per-candidate closure; the "
        "G6.2 funnel refactor wraps this in a helper but the contract "
        "is unchanged."
    )


def test_signature_accepts_engine_kwarg():
    """run_evolution_cycle must accept engine= as a kwarg (new G6.1 hook)."""
    from openclaw_memory_os.evolution import run_evolution_cycle
    sig = inspect.signature(run_evolution_cycle)
    assert "engine" in sig.parameters, "run_evolution_cycle missing engine= kwarg"
    # engine must be Optional (None default) so backward compat is preserved.
    eng = sig.parameters["engine"]
    assert eng.default is None, f"engine default must be None, got {eng.default!r}"
    # candidate_rank_fn kwarg must still exist (backward compat).
    assert "candidate_rank_fn" in sig.parameters, (
        "run_evolution_cycle must keep candidate_rank_fn kwarg for backward compat"
    )


def test_each_candidate_gets_distinct_policy_passed_to_rank_fn_with_policy():
    """When engine= is provided, the loop must invoke rank_fn_with_policy
    once per candidate with that specific candidate's Policy.

    Uses monkeypatch on rank_fn_with_policy to record the policies
    passed in. Drives run_evolution_cycle through the candidate
    evaluation region by stubbing _load_cases / split_cases.
    """
    from openclaw_memory_os import evolution as evo
    from openclaw_memory_os.contracts import (
        CandidateStatus,
        CandidateTier,
        ScoredMemoryCandidate,
    )
    from openclaw_memory_os.policy_store import Policy, baseline_policy
    from datetime import datetime, timezone

    # --- stubs ----------------------------------------------------------
    class _FixedEngine:
        """A retrieval engine stub that returns a fixed hit list."""

        def __init__(self) -> None:
            self.calls: List[Policy] = []

        def retrieve(self, query, *, mode="hybrid", limit=10, status_filter=None, policy=None):
            now = datetime.now(timezone.utc)
            from openclaw_memory_os.retrieval_engine import RetrievalResult, RetrievalDiagnostics
            hits = [
                ScoredMemoryCandidate(
                    collection="t",
                    memory_id=f"m{i}",
                    candidate_key=f"t:m{i}",
                    text=f"hit {i}",
                    status=CandidateStatus.ACTIVE,
                    tier=CandidateTier.MEDIUM,
                    importance=0.5 + 0.05 * i,
                    created_at=now,
                    dense_score=1.0 - 0.1 * i,
                )
                for i in range(5)
            ]
            diag = RetrievalDiagnostics(
                status="ok",
                degraded_reason=None,
                dense_available=True,
                lexical_available=True,
                collections_searched=[],
                candidate_count=len(hits),
            )
            return RetrievalResult(
                hits=hits,
                diagnostics=diag,
                active_count=len(hits),
                fallback_used=False,
                fallback_added=0,
            )

    # Stub load_cases / split so the runner doesn't need real DB feedback.
    # 50 cases to clear the cold-start gate (which requires >= 30 judged
    # queries; otherwise the runner short-circuits before reaching the
    # candidate loop).
    fake_cases = [
        evo._EvaluationCase(
            query_id=f"q{i}",
            query_text=f"query {i}",
            positives={"t:m0"},
            negatives=set(),
        )
        for i in range(50)
    ]

    def _fake_load_cases(limit: int = 500):
        return list(fake_cases)

    def _fake_split(cases):
        return list(cases), [], []

    # Recorder wrapped around rank_fn_with_policy.
    engine = _FixedEngine()
    seen: Dict[int, Policy] = {}

    real_rank_fn_with_policy = evo.rank_fn_with_policy

    def _recording_rank_fn_with_policy(retrieval_engine, policy):
        seen[id(policy)] = policy
        return real_rank_fn_with_policy(retrieval_engine, policy)

    # Stub PolicyStore to avoid disk writes; emit 3 candidates with
    # distinct weights.
    class _StubStore:
        def __init__(self):
            self._active = Policy(
                **{**baseline_policy, "version": 1, "status": evo.PolicyStatus.ACTIVE.value}
            )
            self._shadow = None
            self._previous = None

        def get(self):
            return self._active

        def set(self, p):
            self._active = p

        def get_shadow(self):
            return self._shadow

        def set_shadow(self, p):
            self._shadow = p

        def save(self, p):
            return None

        def revert(self):
            return None

        def checksum(self):
            return "stub"

    # --- run -----------------------------------------------------------
    candidates = [
        Policy(**{**baseline_policy, "version": 100, "importance_weight": 0.80, "recency_weight": 0.10, "feedback_weight": 0.10}),
        Policy(**{**baseline_policy, "version": 101, "importance_weight": 0.20, "recency_weight": 0.60, "feedback_weight": 0.20}),
        Policy(**{**baseline_policy, "version": 102, "importance_weight": 0.40, "recency_weight": 0.30, "feedback_weight": 0.30}),
    ]

    # Generate exactly these candidates by monkeypatching generate_candidates.
    def _fake_generate_candidates(_baseline, *, n_candidates=20, max_delta=0.05, seed=42):
        return list(candidates)

    import openclaw_memory_os.evolution as _evo
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(_evo, "_load_cases", _fake_load_cases)
        monkey.setattr(_evo, "split_cases", _fake_split)
        monkey.setattr(_evo, "generate_candidates", _fake_generate_candidates)
        monkey.setattr(_evo, "rank_fn_with_policy", _recording_rank_fn_with_policy)

        store = _StubStore()
        def rank_fn(query, query_id):
            return ["t:m0", "t:m1", "t:m2"]
        # Pre-acquire the evolution lock so the inline fcntl.lockf call
        # succeeds (otherwise a stale lock file from a previous test run
        # would force the runner into lock_held / skipped path).
        import fcntl
        lock_fd = open(_evo._EVOLUTION_LOCK_PATH, "w")
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _evo.run_evolution_cycle(store, rank_fn, engine=engine)
        finally:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    finally:
        monkey.undo()

    # --- assert -------------------------------------------------------
    # The recorder must have seen EACH candidate's policy passed to
    # rank_fn_with_policy at least once (train loop + val re-bind).
    seen_versions = {p.version for p in seen.values()}
    assert {100, 101, 102}.issubset(seen_versions), (
        f"per-candidate closures not all invoked; saw versions {seen_versions}"
    )


def test_existing_candidate_search_test_still_passes():
    """Backward compat: the legacy single-closure path still works."""
    import subprocess
    import sys
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            "--timeout=30",
            "-p", "no:cacheprovider",
            "tests/test_candidate_search.py::test_runner_uses_retrieval_engine_not_backend_search",
        ],
        capture_output=True, text=True, cwd=".", timeout=60,
    )
    assert proc.returncode == 0, (
        f"legacy candidate_search test broke: stderr={proc.stderr[-500:]!r}"
    )


def test_legacy_path_unchanged_when_engine_kwarg_omitted():
    """When engine= is not provided AND candidate_rank_fn is not provided,
    the runner falls back to rank_fn for every candidate (v0.3.0.x
    legacy single-closure behaviour). The fix must NOT change this.
    """
    from openclaw_memory_os import evolution as evo
    from openclaw_memory_os.policy_store import Policy, baseline_policy

    class _StubStore:
        def __init__(self):
            self._active = Policy(**{**baseline_policy, "version": 1})

        def get(self):
            return self._active

        def set(self, p):
            self._active = p

        def get_shadow(self):
            return None

        def checksum(self):
            return "stub"

    # 50 cases so cold-start gate (>= 30) passes; otherwise the runner
    # short-circuits before reaching the candidate loop.
    fake_cases = [
        evo._EvaluationCase(
            query_id=f"q{i}", query_text=f"q{i}",
            positives={"t:m0"}, negatives=set(),
        )
        for i in range(50)
    ]

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
        monkey.setattr(evo, "split_cases", lambda cases: (list(cases), [], []))
        monkey.setattr(evo, "generate_candidates", lambda baseline, *, n_candidates=20, max_delta=0.05, seed=42: [
            Policy(**{**baseline_policy, "version": 200, "importance_weight": 0.9}),
            Policy(**{**baseline_policy, "version": 201, "importance_weight": 0.7}),
        ])

        called_with: List[Policy] = []
        real_with_policy = evo.rank_fn_with_policy
        def _spy(retrieval_engine, policy):
            called_with.append(policy)
            return real_with_policy(retrieval_engine, policy)
        monkey.setattr(evo, "rank_fn_with_policy", _spy)

        store = _StubStore()
        def rank_fn(query, query_id):
            return ["t:m0"]
        # Pre-acquire the evolution lock so the inline fcntl.lockf succeeds.
        import fcntl
        lock_fd = open(evo._EVOLUTION_LOCK_PATH, "w")
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # No engine kwarg, no candidate_rank_fn kwarg → legacy path.
            evo.run_evolution_cycle(store, rank_fn)
        finally:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    finally:
        monkey.undo()

    # rank_fn_with_policy should NOT be called in legacy mode (no engine).
    assert called_with == [], (
        f"legacy mode should not construct per-candidate closures; got {len(called_with)} calls"
    )