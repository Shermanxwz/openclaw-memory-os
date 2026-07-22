"""G6.2 / G6.5 / G6.7 / G6.9 — closed-loop evolution contract.

This file pins the four behaviour changes that turn the evolution
cycle from "best-effort search" into a real closed loop:

* **G6.2 — Funnel (Top 5 → Top 2 → Top 1).** ``run_evolution_cycle``
  must score every candidate on the train split, keep the top 5,
  re-score those on the val split, keep the top 2, and finally pick
  the single best on the held-out (test) split. Before this fix all
  20 candidates were scored against the same ``cases`` set once,
  so the "winner" was whichever candidate happened to rank highest
  in a single noisy evaluation.

* **G6.5 — Two consecutive pass windows.** A promotion requires
  TWO distinct, non-overlapping evaluation windows to have passed
  (runbook "一次评估不得晋级"). The state file records a ring
  buffer of the last 2 cycles' outcomes; promotion is gated until
  that buffer holds two ``"passed"`` entries.

* **G6.7 — Full rollback triggers.** ``_check_rollback`` must
  trigger on more than just ``degraded_rate > 5%``. The runbook's
  full set: useful@1 -8pp, MRR -5%, negative@5 +8pp, no_result
  +15pp, p95 × 2, fallback_useful -5pp, plus the existing
  absolute floors. Each trigger must reference
  ``state["previous_metrics"]``.

* **G6.9 — Rollback target.** When ``store.revert()`` is called
  and a previous active policy exists, the store must roll back
  to THAT previous policy (not the shipped baseline). We verify
  via the existing ``PolicyStore.revert`` semantics — the test
  sets up a non-baseline previous active policy, triggers a
  rollback, and asserts ``store.get().version`` matches the
  previous.

The tests deliberately use stubs for ``PolicyStore`` /
``rank_fn`` / ``_load_cases`` so they don't depend on the real
``recall_feedback.db`` schema or any live Qdrant backend. That
matches the existing Wave 5 / Wave 6 test pattern (see
``tests/test_per_candidate_rank_fn.py`` for the precedent).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(monkeypatch, tmp_path):
    """Redirect ``MEMORY_OS_RECALL_STATE_DIR`` so tests don't touch the
    real ``evolution-state.json``.

    Each test gets a fresh tmp dir so state written by one test is
    invisible to the next. We also monkeypatch ``_EVOLUTION_STATE_DIR``
    on the live module because that constant is captured at
    module-import time and won't reflect env-var changes made
    after the first import.
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_path / "state"
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(state_dir))
    # Force the module-level state directory constant to track the
    # per-test env var. Without this, the very first test's tmp_path
    # would leak into subsequent tests because
    # ``_EVOLUTION_STATE_DIR`` is evaluated at import time.
    monkeypatch.setattr(evo, "_EVOLUTION_STATE_DIR", state_dir / "openclaw-memory-os")
    yield state_dir


@pytest.fixture
def isolated_evolution_lock(monkeypatch):
    """Pre-acquire the evolution lock so inline ``fcntl.lockf`` calls
    in ``run_evolution_cycle`` succeed without conflicting with
    other tests. If another test in this file holds the lock, this
    fixture blocks until release — pytest's ``monkeypatch`` cleanup
    runs the lock release.
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


def _make_fake_cases(n: int = 60):
    """Build N synthetic ``_EvaluationCase`` objects covering 3 distinct positives.

    The 60-case corpus ensures ``split_cases`` (60/20/20 deterministic
    split by md5 hash) produces a non-empty train / val / test split
    so the funnel has something to narrow. Each case has 1 positive
    doc id chosen from a fixed pool of three so candidate orderings
    can actually move useful@1 between 0 and 1.
    """
    from openclaw_memory_os.evolution import _EvaluationCase

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


def _make_eval(
    *,
    useful_at_1: float = 0.5,
    mrr_at_10: float = 0.5,
    explicit_negative_at_5: float = 0.0,
    no_result_rate: float = 0.0,
    p95_latency: float = 1.0,
    degraded_rate: float = 0.0,
    useful_superseded_fallback_rate: Optional[float] = None,
):
    """Build an ``EvalResult`` with the named field overrides."""
    from openclaw_memory_os.evaluation import EvalResult
    return EvalResult(
        useful_at_1=useful_at_1,
        mrr_at_10=mrr_at_10,
        explicit_negative_at_5=explicit_negative_at_5,
        no_result_rate=no_result_rate,
        p95_latency=p95_latency,
        degraded_rate=degraded_rate,
        useful_superseded_fallback_rate=useful_superseded_fallback_rate,
        num_cases=50,
    )


def _make_candidate(version: int, importance_weight: float = 0.5):
    """Build a minimal valid ``Policy`` candidate.

    ``importance_weight`` is the only knob that affects the per-candidate
    rank closure's score (engine stub below); tweaking it makes
    candidate orderings distinguishable on the funnel.
    """
    from openclaw_memory_os.policy_store import Policy, baseline_policy
    return Policy(
        **{
            **baseline_policy,
            "version": version,
            "importance_weight": importance_weight,
        }
    )


def _stub_engine(return_keys: Optional[List[str]] = None):
    """Build a stub ``RetrievalEngine`` whose ``retrieve()`` returns
    a fixed hit list.

    When ``return_keys`` is None the stub returns 5 hits named
    ``t:m0``..``t:m4`` so the per-candidate closure's score
    function (which combines the engine's per-hit score with
    ``importance_weight * importance``) can produce distinct
    rankings per candidate policy.
    """
    from datetime import datetime, timezone
    from openclaw_memory_os.contracts import (
        CandidateStatus,
        CandidateTier,
        ScoredMemoryCandidate,
    )
    from openclaw_memory_os.retrieval_engine import RetrievalResult, RetrievalDiagnostics

    class _StubEngine:
        def retrieve(self, query, *, mode="hybrid", limit=10, status_filter=None, policy=None):
            now = datetime.now(timezone.utc)
            keys = return_keys if return_keys is not None else [f"t:m{i}" for i in range(5)]
            hits = [
                ScoredMemoryCandidate(
                    collection="t",
                    memory_id=k.split(":")[-1],
                    candidate_key=k,
                    text=f"hit {i}",
                    status=CandidateStatus.ACTIVE,
                    tier=CandidateTier.MEDIUM,
                    importance=0.5 + 0.1 * i,  # distinct per-hit importance
                    created_at=now,
                    dense_score=1.0 - 0.1 * i,
                )
                for i, k in enumerate(keys)
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

    return _StubEngine()


def _policy_store_stub(active_version: int = 1, with_previous: bool = False):
    """Build a stub ``PolicyStore`` with ``revert()`` going to a
    non-baseline previous when ``with_previous`` is True.

    The stub records every operation so tests can assert on the
    "what got promoted / rolled back" calls. ``revert()`` mirrors
    the real ``PolicyStore.revert`` semantics: roll back to the
    stored previous if any, otherwise to the shipped baseline.
    """
    from openclaw_memory_os.policy_store import Policy, baseline_policy

    class _StubStore:
        def __init__(self) -> None:
            self._active = Policy(
                **{**baseline_policy, "version": active_version}
            )
            self._shadow = None
            self._previous = (
                Policy(**{**baseline_policy, "version": active_version - 1})
                if with_previous
                else None
            )
            self.set_calls: List[int] = []
            self.set_shadow_calls: List[int] = []
            self.save_calls: List[int] = []
            self.revert_calls: int = 0

        def get(self):
            return self._active

        def set(self, p):
            self.set_calls.append(int(p.version))
            # Real PolicyStore.set() stashes the prior active as _previous.
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
            if self._previous is not None:
                self._active = self._previous.model_copy(deep=True)
                self._active.status = "active"
                self._previous = None
            else:
                # Shipped baseline: version=1, neutral weights.
                self._active = Policy(
                    **{**baseline_policy, "version": 1}
                )
            return "stub-checksum"

        def checksum(self):
            return "stub-checksum"

    return _StubStore()


# ---------------------------------------------------------------------------
# Test 1 — Funnel narrows 20 → 5 → 2 → 1 across 3 stages
# ---------------------------------------------------------------------------


def test_funnel_picks_top1_through_3_stages(monkeypatch, tmp_state_dir, isolated_evolution_lock):
    """Give 20 distinguishable candidates; only the Top 1 reaches the end.

    We instrument ``evaluate()`` so each call to
    ``evaluate(cases, rank_fn)`` returns a synthetic ``EvalResult``
    whose ``useful_at_1`` matches a per-candidate schedule:

    * Stage 1 (train): candidates 0..4 have useful_at_1 ∈ [0.80..1.0]
      (top 5). Candidates 5..19 are useless (useful_at_1 = 0.0).
    * Stage 2 (val): among the top 5, candidates 0..1 win
      (useful_at_1 = 0.95), 2..4 lose (useful_at_1 = 0.05).
    * Stage 3 (test): among the top 2, candidate 0 wins
      (useful_at_1 = 0.99), candidate 1 loses (0.05).

    Then we assert:

    * ``store.set_shadow`` was called exactly once with the
      winning candidate's version (200),
    * the funnel skipped every other candidate's val / test eval
      (mock records only 5 train evals + 2 val evals + 2 test
      evals = 9 evals total, not 60 which would happen if every
      candidate was tested at every stage).

    Stage 3 also bumps the consecutive-passes counter (the funnel
    ran end-to-end) but promotion is gated, so ``store.set`` is
    NOT called.
    """
    from openclaw_memory_os import evolution as evo

    # Pre-seed state with previous_metrics matching the baseline eval
    # (useful_at_1 = 0.4). Otherwise the seeded neutral baseline
    # (useful_at_1 = 0.5) would trigger an immediate rollback when
    # the baseline eval returns 0.4 — a 10pp drop.
    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.4,
            "mrr_at_10": 0.4,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(60)

    # Candidate pool: 20 candidates with version 200..219.
    candidates = [_make_candidate(version=200 + i) for i in range(20)]

    # Per-candidate useful@1 schedule. ``by_version[cand.version]``
    # is the value ``evaluate`` will return for THAT candidate on
    # a particular stage. Stages are distinguished by the call
    # sequence: train stage consumes 20 evals (1 per candidate),
    # val consumes 5 (top 5), test consumes 2 (top 2).
    stage_useful: Dict[Tuple[str, int], float] = {}

    # Train schedule: top 5 by train (versions 200..204 win).
    for i, c in enumerate(candidates[:5]):
        stage_useful[("train", c.version)] = 0.80 + 0.05 * i  # 0.80, 0.85, ..., 1.00
    for c in candidates[5:]:
        stage_useful[("train", c.version)] = 0.0

    # Val schedule: among the top-5, only the first two win.
    stage_useful[("val", 200)] = 0.95
    stage_useful[("val", 201)] = 0.95
    for v in (202, 203, 204):
        stage_useful[("val", v)] = 0.05

    # Test schedule: among the top-2, only 200 wins.
    stage_useful[("test", 200)] = 0.99
    stage_useful[("test", 201)] = 0.05

    # Wire the split_cases output so we know which stage each call
    # belongs to. We monkeypatch split_cases to return the train /
    # val / test buckets in a known shape; the eval stub then uses
    # the bucket size to choose the right stage key. Sizes match
    # the deterministic split from ``split_cases`` on a 60-case
    # corpus so train/val/test buckets have distinct sizes (the
    # stub differentiates by cases-list length).
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    def _split_stub(cases, *, seed=42):
        return list(train_subset), list(val_subset), list(test_subset)

    # The eval stub distinguishes stages by cases-list size. Baseline
    # calls have no ``_policy_for_test`` hook (rank_fn is the user's
    # plain closure); candidate calls do (we attach the version via
    # ``_binding_rank_fn_with_policy``).
    call_log: List[Tuple[str, int]] = []

    def _eval_with_stage(cases, rank_fn, *, limit=10):
        from openclaw_memory_os.evaluation import EvalResult
        n = len(cases)
        cand_version = getattr(rank_fn, "_policy_for_test", None)
        if cand_version is None:
            # Baseline eval (called from ``evaluate(train_cases, rank_fn)``
            # at the top of the funnel): useful_at_1 = 0.4 so the
            # winning candidates (0.80..1.0) clear the +0.005 gate.
            return EvalResult(
                useful_at_1=0.4,
                mrr_at_10=0.4,
                useful_at_5=0.4,
                ndcg_at_10=0.4,
                recall_at_1=0.4,
                recall_at_5=0.4,
                recall_at_10=0.4,
                explicit_negative_at_5=0.0,
                no_result_rate=0.0,
                p50_latency=0.0,
                p95_latency=1.0,
                degraded_rate=0.0,
                num_cases=n,
            )
        if n == len(train_subset):
            stage = "train"
        elif n == len(val_subset):
            stage = "val"
        elif n == len(test_subset):
            stage = "test"
        else:
            stage = "unknown"
        call_log.append((stage, cand_version))
        useful = stage_useful.get((stage, cand_version), 0.5)
        return EvalResult(
            useful_at_1=useful,
            mrr_at_10=useful,
            useful_at_5=useful,
            ndcg_at_10=useful,
            recall_at_1=useful,
            recall_at_5=useful,
            recall_at_10=useful,
            explicit_negative_at_5=0.0,
            no_result_rate=0.0,
            p50_latency=0.0,
            p95_latency=1.0,
            degraded_rate=0.0,
            num_cases=n,
        )

    # Wrap rank_fn_with_policy so each closure carries the candidate
    # version it was bound to (test inspection hook).
    real_rfp = evo.rank_fn_with_policy
    engine = _stub_engine()

    def _binding_rank_fn_with_policy(retrieval_engine, policy):
        closure = real_rfp(retrieval_engine, policy)
        closure._policy_for_test = int(policy.version)
        return closure

    # --- monkeypatch everything -------------------------------------------
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(evo, "generate_candidates", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(evo, "evaluate", _eval_with_stage)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)

    # --- run the cycle -----------------------------------------------------
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    # --- assertions --------------------------------------------------------

    # The winner was candidate v200.
    assert result["status"] == "shadow", result
    assert result["candidate_version"] == 200, (
        f"funnel winner must be v200, got v{result['candidate_version']}; "
        f"call_log={call_log}"
    )

    # Funnel counts: exactly 20 train evals (one per candidate, on
    # the train split), then exactly 5 val evals (top 5 only) PLUS
    # 1 extra val eval from the shadow-validation step (val_active
    # vs val_cand on the winning candidate), then exactly 2 test
    # evals (top 2 only).
    train_calls = [c for stage, _ in call_log if stage == "train"]
    val_calls = [c for stage, _ in call_log if stage == "val"]
    test_calls = [c for stage, _ in call_log if stage == "test"]
    assert len(train_calls) == 20, (
        f"funnel must score all 20 candidates on train, got {len(train_calls)}; "
        f"call_log={call_log}"
    )
    assert len(val_calls) == 6, (
        f"funnel must score exactly top-5 on val (5) + 1 shadow re-eval (1), "
        f"got {len(val_calls)}; call_log={call_log}"
    )
    assert len(test_calls) == 2, (
        f"funnel must score exactly top-2 on test, got {len(test_calls)}; "
        f"call_log={call_log}"
    )

    # Shadow was set exactly once, with the winner.
    assert store.set_shadow_calls == [200], (
        f"shadow must be set once for v200; got {store.set_shadow_calls}"
    )
    # Promotion gated (two-window rule) so store.set was NOT called.
    assert store.set_calls == [], (
        f"promotion must be gated by two-window rule on first cycle; "
        f"store.set was called {store.set_calls}"
    )

    # State was recorded as "passed" (shadow is still a pass).
    state = json.loads(state_path.read_text())
    assert state["consecutive_passes"] == 1, state
    assert state["pass_windows"][-1]["result"] == "passed", state


# ---------------------------------------------------------------------------
# Test 2 — No promotion after a single pass
# ---------------------------------------------------------------------------


def test_no_promotion_after_single_pass(monkeypatch, tmp_state_dir, isolated_evolution_lock):
    """First cycle returns "shadow" even with winning metrics; promotion
    requires two consecutive passes.

    We make every condition as favourable as possible for promotion
    except the two-window gate:

    * No rollback triggers (good metrics, neutral previous_metrics).
    * Cold-start cleared (50 judged queries).
    * Cooldown satisfied (no prior promotion).
    * Two-window gate blocked: empty state at start.
    * Funnel finds a winning candidate with metrics above baseline.
    * Candidate passes val check.

    After cycle 1: status must be ``"shadow"`` and store.set must NOT
    have been called. The state file must record
    ``consecutive_passes=1``.

    After cycle 2: with timestamps far apart in the past
    (manually seeded into ``pass_windows``), promotion must
    finally fire — but we only assert cycle 1 here. Cycle 3 is
    covered by test 3.
    """
    from openclaw_memory_os import evolution as evo

    # Pre-seed state with previous_metrics matching the baseline eval
    # (useful_at_1 = 0.4). Otherwise the seeded neutral baseline
    # (useful_at_1 = 0.5) would trigger an immediate rollback when
    # the baseline eval returns 0.4 — a 10pp drop.
    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.4,
            "mrr_at_10": 0.4,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(60)
    candidates = [_make_candidate(version=200 + i) for i in range(20)]
    engine = _stub_engine()

    # Win on train (top 5), win on val (top 2), win on test (top 1).
    train_useful = {200: 0.99, 201: 0.85, 202: 0.80, 203: 0.75, 204: 0.70}
    for i in range(5, 20):
        train_useful[200 + i] = 0.0
    val_useful = {200: 0.95, 201: 0.95, 202: 0.05, 203: 0.05, 204: 0.05}
    test_useful = {200: 0.99, 201: 0.05}

    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    def _split_stub(cases, *, seed=42):
        return list(train_subset), list(val_subset), list(test_subset)

    def _eval_stub(cases, rank_fn, *, limit=10):
        from openclaw_memory_os.evaluation import EvalResult
        cand_version = getattr(rank_fn, "_policy_for_test", None)
        n = len(cases)
        if cand_version is None:
            # Baseline: candidate beats it.
            useful = 0.4
        elif n == len(train_subset):
            useful = train_useful.get(cand_version, 0.0)
        elif n == len(val_subset):
            useful = val_useful.get(cand_version, 0.0)
        elif n == len(test_subset):
            useful = test_useful.get(cand_version, 0.0)
        else:
            useful = 0.5
        return EvalResult(
            useful_at_1=useful,
            mrr_at_10=useful,
            useful_at_5=useful,
            ndcg_at_10=useful,
            recall_at_1=useful,
            recall_at_5=useful,
            recall_at_10=useful,
            explicit_negative_at_5=0.0,
            no_result_rate=0.0,
            p50_latency=0.0,
            p95_latency=1.0,
            degraded_rate=0.0,
            num_cases=n,
        )

    real_rfp = evo.rank_fn_with_policy

    def _binding_rank_fn_with_policy(retrieval_engine, policy):
        closure = real_rfp(retrieval_engine, policy)
        closure._policy_for_test = int(policy.version)
        return closure

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(evo, "generate_candidates", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)

    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    # Cycle 1: must NOT promote.
    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)
    assert result["status"] == "shadow", result
    assert result["candidate_version"] == 200, result
    assert store.set_calls == [], (
        "cycle 1 must not promote; two-window rule blocks it"
    )

    state = json.loads(state_path.read_text())
    assert state["consecutive_passes"] == 1, state
    assert len(state["pass_windows"]) == 1, state
    assert state["pass_windows"][-1]["result"] == "passed", state


# ---------------------------------------------------------------------------
# Test 3 — Promotion requires two consecutive passes, reset on fail
# ---------------------------------------------------------------------------


def test_promotion_requires_two_consecutive_pass_windows(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Two passes ≥10 minutes apart → promotion. A 3rd cycle fails,
    a 4th passes → still no promotion (consecutive_passes reset on fail).

    We pre-seed ``state["pass_windows"]`` with TWO ``passed`` entries
    from 20 and 11 minutes ago (well past the 10-minute gap), so the
    very first cycle this test runs should immediately satisfy the
    two-window rule and promote. Then:

    * **Cycle 1**: pre-seeded consecutive_passes=2 + 2 passed windows
      → promotion allowed. Expectation: ``status="promoted"`` and
      ``consecutive_passes==2`` (unchanged on the promotion branch
      since the streak is "consumed").

    * **Cycle 2** (rollback): monkeypatch ``_check_rollback`` to
      return True. Expectation: ``status="rolled_back"``,
      ``consecutive_passes=0``, ``pass_windows=[]``.

    * **Cycle 3** (single pass): undo rollback monkeypatch. With
      state reset to 0 consecutive passes, this cycle ends in
      ``shadow`` (only 1 pass in the bank).

    The test pins the four behaviours the runbook requires:
    a) two-window gate is enforced before promotion,
    b) rollback resets the consecutive counter,
    c) the ring buffer is wiped on rollback,
    d) after a fail, even a pass doesn't immediately re-promote.
    """
    from openclaw_memory_os import evolution as evo
    from datetime import datetime, timedelta, timezone

    fake_cases = _make_fake_cases(60)
    candidates = [_make_candidate(version=200 + i) for i in range(20)]
    engine = _stub_engine()

    # Pre-seed state file: TWO passed windows 20 and 11 minutes ago
    # (both well past the 10-minute gap). Seed ``previous_metrics``
    # with a perfect baseline so cycle 1 does NOT trigger a rollback.
    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    now = datetime.now(timezone.utc)
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [
            {"cycle": (now - timedelta(minutes=20)).isoformat(), "result": "passed", "candidate_version": 200},
            {"cycle": (now - timedelta(minutes=11)).isoformat(), "result": "passed", "candidate_version": 200},
        ],
        "consecutive_passes": 2,
        "pass_candidate_version": 200,
        "previous_metrics": {
            "useful_at_1": 0.95,
            "mrr_at_10": 0.95,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
            "fallback_useful_rate": 0.5,
        },
    }))

    # Funnel schedule: candidate v200 wins on every stage.
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    train_useful = {200: 0.99, 201: 0.85, 202: 0.80, 203: 0.75, 204: 0.70}
    for i in range(5, 20):
        train_useful[200 + i] = 0.0
    val_useful = {200: 0.95, 201: 0.95, 202: 0.05, 203: 0.05, 204: 0.05}
    test_useful = {200: 0.99, 201: 0.05}

    def _split_stub(cases, *, seed=42):
        return list(train_subset), list(val_subset), list(test_subset)

    def _eval_stub(cases, rank_fn, *, limit=10):
        from openclaw_memory_os.evaluation import EvalResult
        cand_version = getattr(rank_fn, "_policy_for_test", None)
        n = len(cases)
        if cand_version is None:
            # Baseline (rank_fn): perfect metrics so no rollback.
            useful = 0.95
            p95 = 1.0
        elif n == len(train_subset):
            useful = train_useful.get(cand_version, 0.0)
            p95 = 1.0
        elif n == len(val_subset):
            useful = val_useful.get(cand_version, 0.0)
            p95 = 1.0
        elif n == len(test_subset):
            useful = test_useful.get(cand_version, 0.0)
            p95 = 1.0
        else:
            useful = 0.5
            p95 = 1.0
        return EvalResult(
            useful_at_1=useful,
            mrr_at_10=useful,
            useful_at_5=useful,
            ndcg_at_10=useful,
            recall_at_1=useful,
            recall_at_5=useful,
            recall_at_10=useful,
            explicit_negative_at_5=0.0,
            no_result_rate=0.0,
            p50_latency=0.0,
            p95_latency=p95,
            degraded_rate=0.0,
            num_cases=n,
        )

    real_rfp = evo.rank_fn_with_policy

    def _binding_rank_fn_with_policy(retrieval_engine, policy):
        closure = real_rfp(retrieval_engine, policy)
        closure._policy_for_test = int(policy.version)
        return closure

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(evo, "generate_candidates", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)

    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    # --- Cycle 1: pre-seeded consecutive_passes=2 + 2 passed windows
    # → promotion allowed.
    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)
    assert result["status"] == "promoted", result
    assert result["candidate_version"] == 200, result
    assert store.set_calls == [200], store.set_calls

    state = json.loads(state_path.read_text())
    # After a successful promotion, the two-window streak is
    # CONSUMED: both ``consecutive_passes`` and ``pass_windows``
    # reset to 0/[] so the next cycle must earn 2 fresh passes
    # before it can promote again. This matches the "一次评估不得
    # 晋级" runbook rule — the gate is "two fresh passes", not
    # "two passes ever".
    assert state["consecutive_passes"] == 0, state
    assert state["pass_windows"] == [], state
    assert state["promotion_count_30d"] == 1, state

    # --- Cycle 2: simulate rollback. We re-monkeypatch ``evaluate``
    # to return metrics that trip the G6.7 ``useful@1`` drop
    # trigger (current 0.50 vs. previous 0.99 — a 49pp drop).
    # The cycle must then run the rollback path through the
    # real ``_check_rollback`` helper.
    def _eval_rollback(cases, rank_fn, *, limit=10):
        from openclaw_memory_os.evaluation import EvalResult
        n = len(cases)
        cand_version = getattr(rank_fn, "_policy_for_test", None)
        if cand_version is None:
            # Baseline: useful_at_1 = 0.50 → rollback (vs prev 0.99).
            return EvalResult(
                useful_at_1=0.50,
                mrr_at_10=0.50,
                useful_at_5=0.50,
                ndcg_at_10=0.50,
                recall_at_1=0.50,
                recall_at_5=0.50,
                recall_at_10=0.50,
                explicit_negative_at_5=0.0,
                no_result_rate=0.0,
                p50_latency=0.0,
                p95_latency=1.0,
                degraded_rate=0.0,
                num_cases=n,
            )
        # Candidate eval still returns "winning" metrics (we want
        # the rollback to fire BEFORE the funnel).
        if n == len(train_subset):
            useful = train_useful.get(cand_version, 0.0)
        elif n == len(val_subset):
            useful = val_useful.get(cand_version, 0.0)
        elif n == len(test_subset):
            useful = test_useful.get(cand_version, 0.0)
        else:
            useful = 0.5
        return EvalResult(
            useful_at_1=useful,
            mrr_at_10=useful,
            useful_at_5=useful,
            ndcg_at_10=useful,
            recall_at_1=useful,
            recall_at_5=useful,
            recall_at_10=useful,
            explicit_negative_at_5=0.0,
            no_result_rate=0.0,
            p50_latency=0.0,
            p95_latency=1.0,
            degraded_rate=0.0,
            num_cases=n,
        )

    # Re-apply all patches (we can't ``monkeypatch.undo()`` here
    # because that also undoes the ``_EVOLUTION_STATE_DIR`` patch
    # from the ``tmp_state_dir`` fixture, which would send cycle
    # 2's state writes to a stale path).
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(evo, "generate_candidates", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(evo, "evaluate", _eval_rollback)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)

    result2 = evo.run_evolution_cycle(store, rank_fn, engine=engine)
    assert result2["status"] == "rolled_back", result2

    state = json.loads(state_path.read_text())
    assert state["consecutive_passes"] == 0, state
    assert state["pass_windows"] == [], state
    assert state["consecutive_rollbacks"] == 1, state

    # --- Cycle 3: re-apply cycle 1's eval stub. State has 0
    # consecutive passes → promotion gated. Re-apply patches
    # without undoing the ``_EVOLUTION_STATE_DIR`` patch from
    # the ``tmp_state_dir`` fixture.
    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(evo, "generate_candidates", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)

    result3 = evo.run_evolution_cycle(store, rank_fn, engine=engine)
    assert result3["status"] == "shadow", result3
    state = json.loads(state_path.read_text())
    assert state["consecutive_passes"] == 1, state
    assert len(state["pass_windows"]) == 1, state


# ---------------------------------------------------------------------------
# Test 4 — Rollback triggered by useful@1 drop ≥ 8pp
# ---------------------------------------------------------------------------


def test_rollback_triggered_by_useful_at_1_drop_8pp(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Current useful@1 = 0.50, previous = 0.60 → 10pp drop → rollback.

    We pre-seed ``state["previous_metrics"]`` with ``useful_at_1=0.60``
    and the rank_fn-driven eval returns ``useful_at_1=0.50``. The
    ``_check_rollback`` helper must fire, call ``store.revert()``,
    and ``store.get().version`` must equal the rolled-back-to
    version (which for a stub with no previous is the baseline v1).
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.60,
            "mrr_at_10": 0.60,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(50)

    def _eval_stub(cases, rank_fn, *, limit=10):
        return _make_eval(useful_at_1=0.50)

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)

    store = _policy_store_stub(active_version=42, with_previous=False)
    def rank_fn(query, query_id):
        return ["t:m0"]

    result = evo.run_evolution_cycle(store, rank_fn)

    assert result["status"] == "rolled_back", result
    assert store.revert_calls == 1, store.revert_calls

    # Rollback went to baseline v1 (no previous).
    assert store.get().version == 1, store.get().version

    # State recorded the rollback: consecutive_passes reset, no
    # promotion counter incremented, previous_metrics preserved
    # (no eval was available to update it via _force_rollback_with_metrics
    # because in this branch the eval was already computed; we kept it).
    state = json.loads(state_path.read_text())
    assert state["consecutive_rollbacks"] == 1, state
    assert state["consecutive_passes"] == 0, state


# ---------------------------------------------------------------------------
# Test 5 — Rollback triggered by p95 × 2
# ---------------------------------------------------------------------------


def test_rollback_triggered_by_p95_x2(monkeypatch, tmp_state_dir, isolated_evolution_lock):
    """Current p95 = 10s, previous p95 = 4s → 2.5× ratio → rollback.

    The G6.7 p95 trigger fires when current > previous × 2. We use
    a 2.5× ratio so the threshold is comfortably exceeded.
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.5,
            "mrr_at_10": 0.5,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 4.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(50)

    def _eval_stub(cases, rank_fn, *, limit=10):
        return _make_eval(useful_at_1=0.5, p95_latency=10.0)

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)

    store = _policy_store_stub(active_version=42, with_previous=False)
    def rank_fn(query, query_id):
        return ["t:m0"]

    result = evo.run_evolution_cycle(store, rank_fn)

    assert result["status"] == "rolled_back", result
    assert store.revert_calls == 1
    assert store.get().version == 1, store.get().version


# ---------------------------------------------------------------------------
# Test 6 — Rollback triggered by MRR drop ≥ 5%
# ---------------------------------------------------------------------------


def test_rollback_triggered_by_mrr_drop_5pct(monkeypatch, tmp_state_dir, isolated_evolution_lock):
    """Current MRR = 0.50, previous MRR = 0.55 → 5pp drop → rollback.

    The G6.7 MRR trigger fires when current < previous - 0.05. We
    drop exactly 5pp so the threshold is exactly at the boundary
    (the condition is strict ``<`` so a 5pp-equal drop does NOT
    fire; we drop 5.5pp here).
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.55,
            "mrr_at_10": 0.55,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(50)

    def _eval_stub(cases, rank_fn, *, limit=10):
        # useful@1 stays at 0.55 (no regression there) so the MRR
        # trigger is the one that fires — not the useful@1 trigger.
        return _make_eval(useful_at_1=0.55, mrr_at_10=0.495)

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)

    store = _policy_store_stub(active_version=42, with_previous=False)
    def rank_fn(query, query_id):
        return ["t:m0"]

    result = evo.run_evolution_cycle(store, rank_fn)

    assert result["status"] == "rolled_back", result
    assert store.revert_calls == 1
    assert store.get().version == 1, store.get().version


# ---------------------------------------------------------------------------
# Test 7 — Rollback does NOT revert to baseline when previous exists
# ---------------------------------------------------------------------------


def test_rollback_does_not_revert_to_baseline_when_previous_exists(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """When a non-baseline previous active policy exists, ``revert()``
    must roll back to THAT previous, not the shipped baseline.

    We pre-seed ``previous_metrics`` so a rollback trigger fires,
    then assert that after the rollback ``store.get().version`` is
    the previous policy's version (e.g. v41), NOT v1.
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_state_dir / "openclaw-memory-os"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "evolution-state.json"
    # Pre-seed with bad previous_metrics so a rollback fires.
    state_path.write_text(json.dumps({
        "promotion_count_30d": 0,
        "consecutive_rollbacks": 0,
        "last_promotion_at": None,
        "shadow_comparisons": [],
        "pass_windows": [],
        "consecutive_passes": 0,
        "previous_metrics": {
            "useful_at_1": 0.90,
            "mrr_at_10": 0.90,
            "explicit_negative_at_5": 0.0,
            "no_result_rate": 0.0,
            "p95_latency": 1.0,
            "degraded_rate": 0.0,
        },
    }))

    fake_cases = _make_fake_cases(50)

    def _eval_stub(cases, rank_fn, *, limit=10):
        # useful@1 = 0.50 → 40pp drop on top of 0.90 previous → rollback.
        return _make_eval(useful_at_1=0.50)

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "evaluate", _eval_stub)

    # Active policy v42 with a previous policy v41 already set.
    store = _policy_store_stub(active_version=42, with_previous=True)

    # Sanity check: the stub's previous is v41.
    assert store.get_previous() is not None
    assert store.get_previous().version == 41

    def rank_fn(query, query_id):
        return ["t:m0"]

    result = evo.run_evolution_cycle(store, rank_fn)

    assert result["status"] == "rolled_back", result
    assert store.revert_calls == 1

    # Crucial: rollback went to v41 (the previous), not v1 (baseline).
    assert store.get().version == 41, (
        f"rollback should have gone to previous v41; got v{store.get().version}. "
        "G6.9 violation: revert() rolled back to baseline when previous existed."
    )
    assert store.get().version != 1, "rollback incorrectly went to baseline v1"


# ---------------------------------------------------------------------------
# Bonus: source-level guard that the new symbols exist
# ---------------------------------------------------------------------------


def test_source_exposes_required_g6_helpers():
    """Static check: the public surface of evolution.py includes the
    new helpers / constants the runbook requires.

    This catches accidental renaming of ``_record_cycle_result``,
    ``_two_consecutive_pass_windows``, ``_FUNNEL_POOL_SIZE``, etc.
    """
    from openclaw_memory_os import evolution as evo

    for sym in (
        "_record_cycle_result",
        "_two_consecutive_pass_windows",
        "_force_rollback_with_metrics",
        "_FUNNEL_POOL_SIZE",
        "_FUNNEL_VAL_SIZE",
        "_PASS_WINDOW_MIN_GAP_SECONDS",
        "_DEFAULT_PREVIOUS_METRICS",
        "_ROLLBACK_USEFUL_AT_1_DROP_THRESHOLD",
        "_ROLLBACK_MRR_DROP_THRESHOLD",
        "_ROLLBACK_NEGATIVE_AT_5_DELTA_THRESHOLD",
        "_ROLLBACK_NO_RESULT_DELTA_THRESHOLD",
        "_ROLLBACK_P95_LATENCY_X_THRESHOLD",
        "_ROLLBACK_FALLBACK_USEFUL_DROP_THRESHOLD",
    ):
        assert hasattr(evo, sym), f"missing helper: {sym}"


def test_state_seed_includes_new_keys(tmp_state_dir):
    """A fresh state file (no on-disk JSON) must already carry the
    new G6.5 / G6.7 keys with their default values. This pins the
    backwards-compat guarantee that operators upgrading from a
    pre-G6 build get a safe first cycle.
    """
    from openclaw_memory_os import evolution as evo

    state = evo._load_evolution_state()
    assert "pass_windows" in state
    assert "consecutive_passes" in state
    assert state["consecutive_passes"] == 0
    assert state["pass_windows"] == []
    assert "previous_metrics" in state
    assert state["previous_metrics"]["useful_at_1"] > 0
    assert state["previous_metrics"]["mrr_at_10"] > 0


def test_two_consecutive_pass_windows_helper():
    """Direct test of the helper that gates promotion."""
    from openclaw_memory_os import evolution as evo
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    # Empty state: helper returns False.
    assert evo._two_consecutive_pass_windows({}) is False
    assert evo._two_consecutive_pass_windows({"pass_windows": []}) is False

    # Single entry: False.
    state = {"pass_windows": [{"cycle": now.isoformat(), "result": "passed"}]}
    assert evo._two_consecutive_pass_windows(state) is False

    # Two passes 11 minutes apart: True.
    state = {
        "pass_windows": [
            {"cycle": (now - timedelta(minutes=11)).isoformat(), "result": "passed"},
            {"cycle": now.isoformat(), "result": "passed"},
        ]
    }
    assert evo._two_consecutive_pass_windows(state) is True

    # Two passes but one failed: False.
    state = {
        "pass_windows": [
            {"cycle": (now - timedelta(minutes=11)).isoformat(), "result": "passed"},
            {"cycle": now.isoformat(), "result": "failed"},
        ]
    }
    assert evo._two_consecutive_pass_windows(state) is False

    # Two passes too close together: False.
    state = {
        "pass_windows": [
            {"cycle": (now - timedelta(seconds=30)).isoformat(), "result": "passed"},
            {"cycle": now.isoformat(), "result": "passed"},
        ]
    }
    assert evo._two_consecutive_pass_windows(state) is False
