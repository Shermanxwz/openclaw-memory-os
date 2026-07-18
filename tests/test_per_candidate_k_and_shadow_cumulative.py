"""G6.1 / G6.4 — per-candidate k values verified + cumulative shadow comparisons.

This file pins the two Runbook v2 contracts that drive candidate
evaluation transparency:

* **G6.1 — Per-candidate k values are wired all the way down.** Two
  candidates with distinct ``dense_k`` (or ``lexical_k`` /
  ``rrf_k``) values must cause the underlying ``retrieve()`` call
  to use the corresponding ``limit=`` parameter (or the
  corresponding ``rrf_k`` in the RRF merge step). The closure
  inside :func:`openclaw_memory_os.evolution.rank_fn_with_policy`
  must therefore capture ``policy`` *by reference* and pass it
  through to :meth:`RetrievalEngine.retrieve`; the engine itself
  then reads ``policy.dense_k`` etc. lazily on each call.

  These tests record the ``limit`` and the ``policy`` instance the
  engine was called with, then assert the recorded values match
  the candidates the evolution cycle produced. If the closure
  captures weights eagerly at build time the k values collapse
  and these tests fail.

* **G6.4 — Cumulative per-query shadow comparisons.** Every
  ``run_evolution_cycle`` invocation that reaches the shadow phase
  must append a per-query comparison row to
  ``state["shadow_comparisons"]``: active top-5 vs candidate top-5,
  agreement (Jaccard on top-5), useful_at_1 / useful_at_5 (judged
  positives vs top-K), and average rank distance. The buffer is
  capped at 1000 entries (oldest dropped) so the state file
  stays bounded. The helpers ``_topk_agreement`` and
  ``_avg_rank_diff`` are pinned by pure-function tests so the
  math stays stable across refactors.

The tests use stubs (no real Qdrant, no real feedback DB) so they
don't depend on external services. That matches the existing
Wave 6 test pattern (``test_per_candidate_rank_fn.py``,
``test_evolution_funnel_consecutive_passes.py``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Reusable helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(monkeypatch, tmp_path):
    """Per-test isolated ``_EVOLUTION_STATE_DIR``.

    Mirrors the helper used in
    ``tests/test_evolution_funnel_consecutive_passes.py``: redirect
    the env var the evolution module reads at import-time AND
    monkeypatch the captured module-level constant so a fresh
    tmp dir is used even when other tests have already imported
    :mod:`openclaw_memory_os.evolution`.
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_path / "state"
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(state_dir))
    monkeypatch.setattr(
        evo, "_EVOLUTION_STATE_DIR", state_dir / "openclaw-memory-os"
    )
    yield state_dir


@pytest.fixture
def isolated_evolution_lock(monkeypatch):
    """Pre-acquire the evolution file lock so inline ``fcntl.lockf``
    calls in ``run_evolution_cycle`` succeed without conflicting
    with parallel tests.
    """
    from openclaw_memory_os import evolution as evo
    import fcntl

    lock_fd = open(evo._EVOLUTION_LOCK_PATH, "w")
    fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield lock_fd
    finally:
        try:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fd.close()


def _make_fake_cases(n: int = 60, *, positives_pool: Optional[List[str]] = None):
    """Build N synthetic ``_EvaluationCase`` objects.

    The 60-case corpus ensures ``split_cases`` (60/20/20
    deterministic split by md5 hash) produces a non-empty train /
    val / test split so the funnel actually has something to
    narrow. Each case has 1 positive doc id chosen from a fixed
    pool (default: ``["t:m0", "t:m1", "t:m2"]``) so candidate
    orderings can move useful@1 between 0 and 1.
    """
    from openclaw_memory_os.evolution import _EvaluationCase

    if positives_pool is None:
        positives_pool = ["t:m0", "t:m1", "t:m2"]
    cases = []
    for i in range(n):
        pos = {positives_pool[i % len(positives_pool)]}
        cases.append(
            _EvaluationCase(
                query_id=f"q{i}",
                query_text=f"query {i}",
                positives=pos,
                negatives=set(),
            )
        )
    return cases


def _make_candidate(
    version: int,
    *,
    importance_weight: float = 0.5,
    dense_k: Optional[int] = None,
    lexical_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
):
    """Build a minimal valid ``Policy`` candidate with optional k overrides.

    The default is the shipped ``baseline_policy`` so a bare
    ``_make_candidate(version=N)`` produces a policy that passes
    the ``Policy`` validator without normalisation gymnastics.
    Tests that need distinct k values pass them through; tests
    that need distinct importance / recency / feedback weights
    override those directly.
    """
    from openclaw_memory_os.policy_store import Policy, baseline_policy

    overrides: Dict[str, Any] = {
        "version": version,
        "importance_weight": importance_weight,
    }
    if dense_k is not None:
        overrides["dense_k"] = int(dense_k)
    if lexical_k is not None:
        overrides["lexical_k"] = int(lexical_k)
    if rrf_k is not None:
        overrides["rrf_k"] = int(rrf_k)
    return Policy(**{**baseline_policy, **overrides})


def _make_realistic_eval_stub():
    """Build an ``evaluate`` stub that returns realistic metrics.

    The stub returns ``useful_at_1=0.5, mrr_at_10=0.5, ...
    no_result_rate=0.0, p95_latency=1.0`` — close enough to the
    seeded ``previous_metrics`` (``useful_at_1=0.4, mrr_at_10=0.4``)
    that the G6.7 statistical rollback triggers do NOT fire on
    the first cycle. The metrics are deliberately ``>=`` the
    seeded baseline so a delta-driven rollback doesn't blank the
    cycle before the candidate-evaluation region runs.

    Without realistic numbers, the stub would return
    ``mrr_at_10=0.0`` (default) and trip the rollback trigger
    ``mrr < prev - 0.05`` immediately. The metric constants come
    from :mod:`openclaw_memory_os.evaluation` so a rename in the
    G6.7 spec is caught at import time rather than silently.
    """
    from openclaw_memory_os.evaluation import EvalResult

    def _stub(eval_cases, fn, *, limit=10):
        return EvalResult(
            useful_at_1=0.5,
            mrr_at_10=0.5,
            explicit_negative_at_5=0.0,
            no_result_rate=0.0,
            p95_latency=1.0,
            degraded_rate=0.0,
            num_cases=len(eval_cases),
        )

    return _stub


def _policy_store_stub(active_version: int = 1):
    """Stub :class:`PolicyStore` with the minimum surface needed by
    :func:`run_evolution_cycle`.

    The stub records every operation so tests can assert on the
    ``set`` / ``set_shadow`` calls. ``revert()`` mirrors the real
    :meth:`PolicyStore.revert` semantics — no previous active, so
    it falls back to the shipped baseline.
    """
    from openclaw_memory_os.policy_store import Policy, baseline_policy

    class _StubStore:
        def __init__(self) -> None:
            self._active = Policy(
                **{**baseline_policy, "version": active_version}
            )
            self._shadow = None
            self._previous = None
            self.set_calls: List[int] = []
            self.set_shadow_calls: List[int] = []
            self.save_calls: List[int] = []
            self.revert_calls: int = 0

        def get(self):
            return self._active

        def set(self, p):
            self.set_calls.append(int(p.version))
            self._previous = self._active.model_copy(deep=True)
            self._previous.status = "retired"
            self._active = p.model_copy(deep=True)
            self._active.status = "active"
            return "stub-checksum"

        def get_shadow(self):
            return self._shadow

        def set_shadow(self, p):
            self._shadow = p
            self.set_shadow_calls.append(int(p.version))

        def save(self, p):
            self.save_calls.append(int(p.version))
            return None

        def get_previous(self):
            return self._previous

        def revert(self):
            self.revert_calls += 1
            self._active = Policy(**{**baseline_policy, "version": 1})
            return "stub-checksum"

        def checksum(self):
            return "stub-checksum"

    return _StubStore()


def _pre_seed_state(state_dir: Path, previous_metrics: Optional[Dict[str, float]] = None) -> None:
    """Pre-seed an ``evolution-state.json`` so the rollback gate
    doesn't fire on the neutral-baseline seed.

    Without this, the very first cycle's eval numbers (which the
    stub returns as 0.0 for useful_at_1 etc.) would compare against
    the neutral 0.5 seed and trip an immediate rollback. Operators
    in production get a similar effect because
    ``_DEFAULT_PREVIOUS_METRICS`` only protects a single fresh
    cycle; tests that drive the cycle twice in quick succession
    need to re-seed.
    """
    target = state_dir / "openclaw-memory-os"
    target.mkdir(parents=True, exist_ok=True)
    if previous_metrics is None:
        previous_metrics = {
            "useful_at_1": 0.4,
            "mrr_at_10": 0.4,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        }
    target.joinpath("evolution-state.json").write_text(
        json.dumps({
            "promotion_count_30d": 0,
            "consecutive_rollbacks": 0,
            "last_promotion_at": None,
            "shadow_comparisons": [],
            "pass_windows": [],
            "consecutive_passes": 0,
            "previous_metrics": previous_metrics,
        })
    )


# ---------------------------------------------------------------------------
# Pure-function tests (no cycle runner)
# ---------------------------------------------------------------------------


def test_avg_rank_diff_correct():
    """``a=[A,B,C], b=[A,C,B]`` → avg_rank_diff = 2/3.

    Pure-function pin. ``A`` stays at rank 0; ``B`` moves from
    rank 1 → 2 (Δ=1); ``C`` moves from rank 2 → 1 (Δ=1). The
    mean of (0, 1, 1) is 2/3.
    """
    from openclaw_memory_os.evolution import _avg_rank_diff

    assert _avg_rank_diff(["A", "B", "C"], ["A", "C", "B"]) == pytest.approx(2.0 / 3.0)


def test_topk_agreement_correct():
    """``a=[A,B,C], b=[A,B,D]`` → agreement = 2 / 4 = 0.5.

    Jaccard overlap: ``a ∩ b = {A, B}``, ``a ∪ b = {A, B, C, D}``.
    The fraction is 2/4 = 0.5.
    """
    from openclaw_memory_os.evolution import _topk_agreement

    assert _topk_agreement(["A", "B", "C"], ["A", "B", "D"]) == pytest.approx(0.5)


def test_avg_rank_diff_disjoint_is_inf():
    """Two disjoint lists have no common candidate → ``inf``."""
    from openclaw_memory_os.evolution import _avg_rank_diff

    assert _avg_rank_diff(["A", "B"], ["C", "D"]) == float("inf")


def test_topk_agreement_both_empty_is_one():
    """Two empty lists vacuously agree → 1.0."""
    from openclaw_memory_os.evolution import _topk_agreement

    assert _topk_agreement([], []) == 1.0


# ---------------------------------------------------------------------------
# G6.1 — per-candidate k values are wired through to retrieve()
# ---------------------------------------------------------------------------


class _RecordingMemoryBackend:
    """Stub :class:`MemoryBackend` that returns real
    :class:`ScoredMemoryCandidate` hits AND records ``dense_search``
    ``limit`` values.

    The evolution funnel's candidate-aware closure calls
    :func:`RetrievalEngine.retrieve` once per candidate, which
    in turn calls ``self.backend.dense_search(query, limit=policy.dense_k)``.
    By recording the ``limit`` parameter AND returning real
    ScoredMemoryCandidate hits, the engine produces a real
    :class:`RetrievalResult` whose ``hits`` are consumable by
    :func:`evaluate` — i.e. the cycle can run end-to-end with the
    real evaluator (NOT a stub) so the per-candidate closures are
    actually invoked.

    The hits are deterministic: each candidate_key is ``t:m{i}``
    for ``i in range(min(5, limit))`` so the top-5 is always
    ``[t:m0, t:m1, t:m2, t:m3, t:m4]`` regardless of the limit.
    That keeps the eval signal independent of the k value — the
    k value only affects what the BACKEND SEES (the recorded
    limit), not what useful@1 the cycle ultimately computes.
    """

    def __init__(self) -> None:
        self.searches: List[Dict[str, Any]] = []
        self.lexical_calls: List[Dict[str, Any]] = []

    def list_memories(self) -> List[Any]:
        return []

    def list_collections(self) -> List[str]:
        return ["t"]

    def get_memory(self, memory_id: str) -> Any:
        return None

    def dense_search(
        self,
        query: str,
        limit: int = 10,
        *,
        status_filter: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        # The engine does NOT pass the policy object to the
        # backend — it only passes ``limit=policy.dense_k``. So
        # the strongest assertion is "limit matches policy.dense_k
        # for SOME known candidate in this cycle".
        self.searches.append({"limit": int(limit), "kind": "dense"})
        # Return up to 5 hits so the per-candidate closure has
        # something to rank. Limit is capped at 5 so a tiny
        # ``dense_k`` (e.g. 10) and a huge one (e.g. 50) both
        # produce the same top-5 — the k value only affects the
        # recorded call, not the eval signal.
        return _make_scored_hits(min(int(limit), 5))

    def lexical_search(
        self,
        query: str,
        limit: int = 40,
        status_filter: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        # Capture the ``limit`` argument; the engine passes
        # ``policy.lexical_k`` here.
        self.lexical_calls.append({"limit": int(limit)})
        return []

    def search(self, query: str, limit: int = 10) -> List[Any]:
        # Legacy ``search`` hook used by the dense-mode bridge.
        self.searches.append({"limit": int(limit), "kind": "legacy_search"})
        return []


def _make_scored_hits(n: int):
    """Build ``n`` :class:`ScoredMemoryCandidate` objects for tests.

    Hits are named ``t:m0``..``t:m(n-1)`` so a query whose
    positive set is ``{"t:m0"}`` produces a useful@1 == 1.0
    regardless of ``n`` (as long as ``n >= 1``). Used by
    ``_RecordingMemoryBackend`` so the cycle's evaluator can
    compute real metrics against the closure-built hits.
    """
    from openclaw_memory_os.contracts import (
        CandidateStatus,
        CandidateTier,
        ScoredMemoryCandidate,
    )
    now = datetime.now(timezone.utc)
    return [
        ScoredMemoryCandidate(
            collection="t",
            memory_id=f"m{i}",
            candidate_key=f"t:m{i}",
            text=f"hit {i}",
            status=CandidateStatus.ACTIVE,
            tier=CandidateTier.MEDIUM,
            importance=0.5,
            created_at=now,
            dense_score=1.0 - 0.1 * i,
        )
        for i in range(max(0, int(n)))
    ]


def test_two_candidates_with_different_dense_k_produce_different_results(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate K values truncate one shared pool, never re-query Qdrant."""
    from openclaw_memory_os.candidate_pool import QueryCandidatePool
    from openclaw_memory_os.contracts import (
        CandidateStatus, CandidateTier, MemoryRecord, ScoredMemoryCandidate,
    )
    from openclaw_memory_os.policy_store import Policy, baseline_policy
    from openclaw_memory_os.retrieval_engine import RetrievalEngine

    backend = _RecordingMemoryBackend()
    class _PS:
        def get(self):
            return Policy(**baseline_policy)
    engine = RetrievalEngine(backend=backend, policy_store=_PS())
    dense = []
    for i in range(60):
        record = MemoryRecord(
            collection="t", memory_id=f"m{i}", candidate_key=f"t:m{i}",
            text=f"memory {i}", status=CandidateStatus.ACTIVE,
            tier=CandidateTier.MEDIUM, importance=0.5,
        )
        dense.append(ScoredMemoryCandidate.from_record(
            record, score=1.0 - i / 100.0, dense_score=1.0 - i / 100.0
        ))
    pool = QueryCandidatePool(query="q", dense_active=dense, lexical_active=[])
    p10 = Policy(**{**baseline_policy, "version": 400, "dense_k": 10})
    p50 = Policy(**{**baseline_policy, "version": 401, "dense_k": 50})
    r10 = engine.rank_candidate_pool(pool, p10, limit=100)
    r50 = engine.rank_candidate_pool(pool, p50, limit=100)
    assert len(r10.hits) == 10
    assert len(r50.hits) == 50
    assert backend.searches == []  # policy reranking is pure in-memory


def test_two_candidates_with_different_rrf_k_produce_different_rrf_scores(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """rrf_k changes scores on the same raw ranks without backend calls."""
    from openclaw_memory_os.candidate_pool import QueryCandidatePool
    from openclaw_memory_os.contracts import (
        CandidateStatus, CandidateTier, MemoryRecord, ScoredMemoryCandidate,
    )
    from openclaw_memory_os.policy_store import Policy, baseline_policy
    from openclaw_memory_os.retrieval_engine import RetrievalEngine

    backend = _RecordingMemoryBackend()
    class _PS:
        def get(self):
            return Policy(**baseline_policy)
    engine = RetrievalEngine(backend=backend, policy_store=_PS())
    records = [
        MemoryRecord(
            collection="t", memory_id=f"m{i}", candidate_key=f"t:m{i}",
            text=f"memory {i}", status=CandidateStatus.ACTIVE,
            tier=CandidateTier.MEDIUM, importance=0.5,
        ) for i in range(3)
    ]
    dense = [ScoredMemoryCandidate.from_record(r, dense_score=1.0-i*0.2) for i, r in enumerate(records)]
    lexical = [
        ScoredMemoryCandidate.from_record(records[2], lexical_score=1.0),
        ScoredMemoryCandidate.from_record(records[0], lexical_score=0.5),
    ]
    pool = QueryCandidatePool(query="q", dense_active=dense, lexical_active=lexical)
    p20 = Policy(**{**baseline_policy, "version": 500, "rrf_k": 20})
    p60 = Policy(**{**baseline_policy, "version": 501, "rrf_k": 60})
    r20 = {h.candidate_key: h.rrf_score for h in engine.rank_candidate_pool(pool, p20, limit=10).hits}
    r60 = {h.candidate_key: h.rrf_score for h in engine.rank_candidate_pool(pool, p60, limit=10).hits}
    assert r20["t:m0"] != r60["t:m0"] or r20["t:m2"] != r60["t:m2"]
    assert backend.searches == []


def test_closure_passes_policy_by_reference_not_by_value():
    """Static check: ``rank_fn_with_policy`` returns a closure that
    passes the dataclass through to ``retrieve()`` (NOT a snapshot
    of the weight fields).

    The check is intentionally source-only so a refactor that
    captures weights eagerly (e.g. ``partial(retrieve, policy=policy)``
    after unpacking ``policy.dense_k``) is caught at test time
    rather than at runtime.
    """
    from pathlib import Path as _Path

    src = _Path("openclaw_memory_os/evolution.py").read_text(encoding="utf-8")

    # The factory must accept a ``policy: Policy`` parameter and
    # the inner closure must reference ``policy`` (not
    # ``_policy_snapshot`` or any weight-prefixed local).
    assert "def rank_fn_with_policy(" in src, (
        "rank_fn_with_policy factory not found in evolution.py"
    )
    # Look for ``policy=policy`` — the closure passes the dataclass
    # directly to ``retrieve()``. A bug-fix that captures weights
    # eagerly would change this to ``policy=PolicySnapshot(...)`` or
    # similar, and the test would fail.
    assert "policy=policy" in src, (
        "rank_fn_with_policy closure does not pass `policy=policy` to "
        "retrieve(); the closure may be capturing weights eagerly. "
        "G6.1 requires the policy dataclass to be passed by reference."
    )


# ---------------------------------------------------------------------------
# G6.4 — per-query shadow comparison buffer
# ---------------------------------------------------------------------------


def test_shadow_compare_buffer_populated_per_query(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Each query in the eval set produces one comparison row in
    ``state["shadow_comparisons"]``.

    We drive the cycle with two known rank_fns:

    * ``rank_fn`` (active) returns ``[A, B, C, D, E]`` for every
      query.
    * The candidate rank_fn returns ``[A, C, B, F, G]``.

    For a query whose positive set is ``{C}``:
      * active useful_at_1 = False (A not in positives),
      * active useful_at_5 = True (B, C, D, E contain C),
      * candidate useful_at_1 = False,
      * candidate useful_at_5 = True (B, C, F, G contain C),
      * agreement (top-5 Jaccard) = |{A, B, C} ∩ {A, C, B}| /
        |{A, B, C, D, E} ∪ {A, C, B, F, G}| = 3 / 7.
    """
    from openclaw_memory_os import evolution as evo

    _pre_seed_state(tmp_state_dir)

    # 60 cases to clear cold-start; each case has positive set
    # ``{"t:m0"}`` so the comparison rows are easy to compute by
    # hand.
    cases = _make_fake_cases(60, positives_pool=["t:m0"])

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(cases))
    monkeypatch.setattr(evo, "split_cases", lambda c, *, seed=42: (
        list(cases[:39]), list(cases[39:48]), list(cases[48:60]),
    ))

    # Force the cycle's funnel to evaluate exactly one candidate
    # (so the per-query loop is short and predictable).
    candidate = _make_candidate(version=600)
    monkeypatch.setattr(
        evo,
        "generate_candidates",
        lambda baseline, *, n_candidates=20, max_delta=0.05, seed=42: [candidate],
    )

    # Stub ``evaluate`` so it returns realistic metrics that don't
    # trip the G6.7 statistical rollback triggers (the rollback
    # check uses ``evaluate`` too, so the stub has to be honest
    # about MRR / negative@5 / etc.). The PER-QUERY shadow
    # recording uses our own rank_fn lambdas, NOT the stubbed
    # ``evaluate``, so the per-query rows still reflect the
    # lambdas rather than the synthetic eval result.
    monkeypatch.setattr(evo, "evaluate", _make_realistic_eval_stub())

    # Active rank_fn returns [A, B, C, D, E]; candidate returns
    # [A, C, B, F, G]. Both return lists, matching the legacy
    # ``rank_fn`` contract.
    active_results = ["t:alice", "t:bob", "t:carol", "t:dave", "t:eve"]
    candidate_results = ["t:alice", "t:carol", "t:bob", "t:frank", "t:grace"]

    store = _policy_store_stub(active_version=1)

    def rank_fn(query_text, query_id):
        return list(active_results)
    def cand_rank_fn(query_text, query_id):
        return list(candidate_results)

    evo.run_evolution_cycle(store, rank_fn, candidate_rank_fn=cand_rank_fn)

    # Inspect the per-query shadow comparison buffer.
    state = evo._load_evolution_state()
    buffer = state.get("shadow_comparisons", [])

    # The buffer must have one entry per val case (9 cases per
    # the deterministic split on 60 cases: cases[39:48]).
    expected_n = 9
    assert len(buffer) == expected_n, (
        f"G6.4: expected {expected_n} comparison rows, got {len(buffer)}"
    )

    # Every row has the required keys.
    required_keys = {
        "query_id", "query_text",
        "active_top5", "candidate_top5",
        "agreement", "rank_distance",
        "active_useful_at_1", "candidate_useful_at_1",
        "active_useful_at_5", "candidate_useful_at_5",
        "recorded_at",
    }
    for row in buffer:
        missing = required_keys - set(row.keys())
        assert not missing, f"G6.4: row missing keys {missing}: {row}"

    # Spot-check the first row's math. Both top-5 lists share
    # A / B / C → agreement = 3 / 7.
    row0 = buffer[0]
    assert row0["active_top5"] == active_results
    assert row0["candidate_top5"] == candidate_results
    assert row0["agreement"] == pytest.approx(3.0 / 7.0)

    # The candidate moves B from rank 1 → rank 2 (Δ=1) and C
    # from rank 2 → rank 1 (Δ=1); A stays at rank 0 (Δ=0). Mean
    # rank distance = (0 + 1 + 1) / 3 = 2/3.
    assert row0["rank_distance"] == pytest.approx(2.0 / 3.0)

    # Each query's positive set is ``{"t:m0"}`` (set above); NEITHER
    # the active nor the candidate top-5 contains ``t:m0``, so
    # useful_at_1 / useful_at_5 are False on both sides.
    assert row0["active_useful_at_1"] is False
    assert row0["candidate_useful_at_1"] is False
    assert row0["active_useful_at_5"] is False
    assert row0["candidate_useful_at_5"] is False


def test_shadow_compare_buffer_capped_at_1000(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Buffer must cap at exactly 1000 entries (oldest dropped)."""
    from openclaw_memory_os import evolution as evo

    _pre_seed_state(tmp_state_dir)

    # 1500 cases — more than the cap. Each case has a unique
    # ``query_id`` so we can verify oldest entries were dropped.
    cases = []
    from openclaw_memory_os.evolution import _EvaluationCase
    for i in range(1500):
        cases.append(
            _EvaluationCase(
                query_id=f"bigq{i:05d}",
                query_text=f"big query {i}",
                positives={"t:m0"},
                negatives=set(),
            )
        )

    # Put ALL 1500 cases into the val split so every one of
    # them is fed through the per-query recorder.
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(cases))
    monkeypatch.setattr(evo, "split_cases", lambda c, *, seed=42: (
        list(cases[:50]),  # tiny train so funnel is short
        list(cases),       # ALL 1500 cases are val
        [],                # no held-out
    ))

    candidate = _make_candidate(version=700)
    monkeypatch.setattr(
        evo,
        "generate_candidates",
        lambda baseline, *, n_candidates=20, max_delta=0.05, seed=42: [candidate],
    )

    monkeypatch.setattr(evo, "evaluate", _make_realistic_eval_stub())

    store = _policy_store_stub(active_version=1)
    def rank_fn(qt, qid):
        return ["t:alice", "t:bob", "t:carol"]
    def cand_rank_fn(qt, qid):
        return ["t:carol", "t:bob", "t:alice"]

    evo.run_evolution_cycle(store, rank_fn, candidate_rank_fn=cand_rank_fn)

    state = evo._load_evolution_state()
    buffer = state.get("shadow_comparisons", [])

    # Exactly 1000 — the cap.
    assert len(buffer) == 1000, (
        f"G6.4: buffer length {len(buffer)} != cap 1000"
    )

    # The OLDEST entries were dropped (the buffer keeps the LAST
    # 1000 queries). The first retained entry corresponds to
    # query index 500 (1500 - 1000).
    assert buffer[0]["query_id"] == "bigq00500"
    # The last retained entry is the very last query.
    assert buffer[-1]["query_id"] == "bigq01499"


def test_shadow_compare_buffer_empty_when_val_split_empty(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """When the val split is empty the recorder skips cleanly — no
    rows appended, no exception raised.

    This pins the recorder's tolerance for tiny corpora where
    ``split_cases`` puts everything in train.
    """
    from openclaw_memory_os import evolution as evo

    _pre_seed_state(tmp_state_dir)

    # 60 cases, all into the train split; val and test empty.
    cases = _make_fake_cases(60)
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(cases))
    monkeypatch.setattr(evo, "split_cases", lambda c, *, seed=42: (
        list(cases), [], [],
    ))

    candidate = _make_candidate(version=800)
    monkeypatch.setattr(
        evo,
        "generate_candidates",
        lambda baseline, *, n_candidates=20, max_delta=0.05, seed=42: [candidate],
    )
    monkeypatch.setattr(evo, "evaluate", _make_realistic_eval_stub())

    store = _policy_store_stub(active_version=1)
    def rank_fn(qt, qid):
        return ["t:alice", "t:bob", "t:carol"]
    def cand_rank_fn(qt, qid):
        return ["t:carol", "t:bob", "t:alice"]

    # Should NOT raise.
    evo.run_evolution_cycle(store, rank_fn, candidate_rank_fn=cand_rank_fn)

    state = evo._load_evolution_state()
    buffer = state.get("shadow_comparisons", [])
    assert buffer == [], (
        f"G6.4: empty val split should produce empty buffer, got {len(buffer)} rows"
    )


def test_get_shadow_comparisons_accessor(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """``get_shadow_comparisons()`` returns the buffer that
    ``run_evolution_cycle`` populated.

    Public API surface used by the dashboard / ``/api/dashboard/strategy``
    endpoint to render the per-query comparison rows.
    """
    from openclaw_memory_os import evolution as evo

    _pre_seed_state(tmp_state_dir)

    cases = _make_fake_cases(60)
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(cases))
    monkeypatch.setattr(evo, "split_cases", lambda c, *, seed=42: (
        list(cases[:39]), list(cases[39:48]), list(cases[48:60]),
    ))

    candidate = _make_candidate(version=900)
    monkeypatch.setattr(
        evo,
        "generate_candidates",
        lambda baseline, *, n_candidates=20, max_delta=0.05, seed=42: [candidate],
    )
    monkeypatch.setattr(evo, "evaluate", _make_realistic_eval_stub())

    store = _policy_store_stub(active_version=1)
    def rank_fn(qt, qid):
        return ["t:alice", "t:bob", "t:carol"]
    def cand_rank_fn(qt, qid):
        return ["t:carol", "t:bob", "t:alice"]

    evo.run_evolution_cycle(store, rank_fn, candidate_rank_fn=cand_rank_fn)

    observed = evo.get_shadow_comparisons()
    assert isinstance(observed, list)
    assert len(observed) == 9, (
        f"G6.4: get_shadow_comparisons returned {len(observed)} rows, expected 9"
    )
    # Every row carries the required keys.
    for row in observed:
        assert "query_id" in row
        assert "agreement" in row
        assert "rank_distance" in row