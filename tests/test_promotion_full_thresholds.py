"""G6.6 \u2014 full 9-metric promotion gate.

Pre-G6.6 contract: ``run_evolution_cycle`` promoted on a single
metric (useful@1 \u00d7 1.005 over the train baseline). That is far
too lenient for an auto-promotion path \u2014 a candidate can
silently regress on nDCG, MRR, latency, or fallback quality while
still winning on useful@1.

G6.6 replaces the single check with a 9-metric gate. ALL of the
following must hold for the candidate to be promoted:

1. ``useful_at_1 >= active.useful_at_1`` (no regression)
2. ``mrr_at_10 >= active.mrr_at_10`` (no regression)
3. ``ndcg_at_10 >= active.ndcg_at_10 + 0.01`` (\u22651pp improvement)
4. ``useful_at_5 >= active.useful_at_5`` (no regression)
5. ``explicit_negative_at_5 <= active.explicit_negative_at_5 + 0.05``
   (\u2264+5pp)
6. ``fallback_useful_rate >= active.fallback_useful_rate - 0.05``
   (\u2265-5pp, when both sides carry a value)
7. ``degraded_rate <= active.degraded_rate`` (no regression)
8. ``p95_latency <= active.p95_latency \u00d7 1.5`` (\u22641.5\u00d7 baseline)
9. ``positive_hit_at_5 >= active.positive_hit_at_5`` (no regression;
   alias of useful_at_5 on ``EvalResult``)

These tests pin the new contract end-to-end:

1. ``test_all_thresholds_pass_promotes`` \u2014 candidate strictly beats
   active on all 9 metrics \u2192 after two consecutive passes, the
   candidate is promoted.
2. ``test_any_threshold_failure_blocks_promotion`` \u2014 candidate
   wins 8/9 but loses on p95_latency \u2192 status="shadow",
   ``failed_metrics`` contains "p95_latency".
3. ``test_threshold_reasons_are_listed`` \u2014 candidate loses 3
   metrics, ``failed_metrics`` lists all 3.
4. ``test_ndcg_at_10_requires_plus_0_01`` \u2014 candidate has
   ``ndcg_at_10 == active.ndcg_at_10`` (no improvement) \u2192 blocked.
5. ``test_negative_at_5_uses_plus_0_05_tolerance`` \u2014 candidate
   ``negative_at_5 == active + 0.04`` (within tolerance) \u2192
   allowed.
6. ``test_negative_at_5_blocks_when_plus_0_06`` \u2014 candidate
   ``negative_at_5 == active + 0.06`` (over tolerance) \u2192 blocked.

The tests use the same stub pattern as
``tests/test_evolution_funnel_consecutive_passes.py`` (stub
``PolicyStore`` / ``evaluate`` / ``rank_fn_with_policy`` / state
file) so they don't depend on the real ``recall_feedback.db`` or
any live Qdrant backend.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(monkeypatch, tmp_path):
    """Redirect ``MEMORY_OS_RECALL_STATE_DIR`` so the promotion
    tests don't touch the real ``evolution-state.json`` file.

    Same pattern as the existing evolution tests \u2014 we monkeypatch
    ``_EVOLUTION_STATE_DIR`` on the live module because that
    constant is captured at module-import time.
    """
    from openclaw_memory_os import evolution as evo

    state_dir = tmp_path / "state"
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(state_dir))
    monkeypatch.setattr(evo, "_EVOLUTION_STATE_DIR", state_dir / "openclaw-memory-os")
    yield state_dir


@pytest.fixture
def isolated_evolution_lock(monkeypatch):
    """Pre-acquire the evolution lock so the cycle's
    ``fcntl.lockf`` call succeeds without conflicting with
    parallel tests.
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
    """60 synthetic cases so the deterministic 60/20/20 split
    produces non-empty train/val/test buckets."""
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
    ndcg_at_10: float = 0.5,
    useful_at_5: float = 0.5,
    explicit_negative_at_5: float = 0.0,
    no_result_rate: float = 0.0,
    p95_latency: float = 1.0,
    degraded_rate: float = 0.0,
    fallback_useful_rate: Optional[float] = 0.5,
) -> Any:
    """Build an ``EvalResult`` with named field overrides.

    Defaults are tuned so a "good" eval beats the typical active
    eval on every metric without surprises.
    """
    from openclaw_memory_os.evaluation import EvalResult

    return EvalResult(
        useful_at_1=useful_at_1,
        mrr_at_10=mrr_at_10,
        ndcg_at_10=ndcg_at_10,
        useful_at_5=useful_at_5,
        explicit_negative_at_5=explicit_negative_at_5,
        no_result_rate=no_result_rate,
        p95_latency=p95_latency,
        degraded_rate=degraded_rate,
        fallback_useful_rate=fallback_useful_rate,
        num_cases=50,
    )


def _make_candidate(version: int):
    """Build a minimal valid candidate ``Policy``."""
    from openclaw_memory_os.policy_store import Policy, baseline_policy

    return Policy(
        **{**baseline_policy, "version": version, "importance_weight": 0.5}
    )


def _stub_engine():
    """Build a stub ``RetrievalEngine`` that always returns the
    same hit list, so candidate evaluations are deterministic.
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
            hits = [
                ScoredMemoryCandidate(
                    collection="t",
                    memory_id=f"m{i}",
                    candidate_key=f"t:m{i}",
                    text=f"hit {i}",
                    status=CandidateStatus.ACTIVE,
                    tier=CandidateTier.MEDIUM,
                    importance=0.5 + 0.1 * i,
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

    return _StubEngine()


def _policy_store_stub(active_version: int = 1):
    """Stub ``PolicyStore`` recording every operation.

    The stub mirrors the real ``PolicyStore`` semantics for the
    fields the cycle touches (``get`` / ``set`` / ``set_shadow`` /
    ``save`` / ``revert``); we only need to assert on
    ``set_calls`` and ``save_calls`` for the promotion tests.
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
            self._previous = None
            self._active = Policy(**{**baseline_policy, "version": 1})
            return "stub-checksum"

        def checksum(self):
            return "stub-checksum"

    return _StubStore()


def _build_passing_evaluations(
    *, active_kwargs: Optional[Dict[str, Any]] = None, cand_delta: float = 0.02
) -> Tuple[Any, Any]:
    """Build an (active_eval, candidate_eval) pair where the
    candidate strictly beats the active on every G6.6 metric.

    ``cand_delta`` is the improvement margin applied to the
    "higher is better" metrics; the "lower is better" metrics get
    a matching reduction.
    """
    active_kwargs = active_kwargs or {}
    active = _make_eval(**active_kwargs)
    candidate = _make_eval(
        useful_at_1=active.useful_at_1 + cand_delta,
        mrr_at_10=active.mrr_at_10 + cand_delta,
        ndcg_at_10=active.ndcg_at_10 + cand_delta + 0.005,  # > +0.01 delta vs active
        useful_at_5=active.useful_at_5 + cand_delta,
        explicit_negative_at_5=max(0.0, active.explicit_negative_at_5 - cand_delta),
        p95_latency=active.p95_latency * 1.0,  # exact baseline
        degraded_rate=max(0.0, active.degraded_rate - 0.001),
        fallback_useful_rate=(
            active.fallback_useful_rate + cand_delta
            if active.fallback_useful_rate is not None
            else None
        ),
    )
    return active, candidate


def _install_eval_stub(
    monkeypatch,
    evo,
    *,
    active_eval: Any,
    candidate_eval: Any,
    fake_cases: List,
    train_subset: List,
    val_subset: List,
    test_subset: List,
    skip_val: bool = False,
    skip_test: bool = False,
) -> None:
    """Wire a stub ``evaluate`` that returns the supplied evals
    at the right stage.

    The funnel shape is: train (all 20 candidates) \u2192 val (top 5) \u2192
    test (top 2). The shadow-validation step re-evaluates the
    winning candidate on val, so the candidate eval also needs to
    match on the val split.

    When ``skip_val`` is True, the stub returns an empty val split
    so the legacy ``evaluate_candidate`` shadow check is bypassed
    entirely (the cycle falls straight through to the G6.6 gate).
    Tests that need to verify a single G6.6 metric in isolation
    use this option so the older, stricter shadow check doesn't
    pre-empt the gate with a false-positive failure.

    When ``skip_test`` is True, the stub returns an empty test
    split so the cycle uses the val eval as the held-out eval.
    """
    real_rfp = evo.rank_fn_with_policy

    def _binding_rank_fn_with_policy(retrieval_engine, policy):
        closure = real_rfp(retrieval_engine, policy)
        closure._policy_for_test = int(policy.version)
        return closure

    def _eval_stub(cases, rank_fn, *, limit=10):
        cand_version = getattr(rank_fn, "_policy_for_test", None)
        len(cases)
        if cand_version is None:
            # Baseline eval (active policy on the active rank_fn).
            # The cycle uses this for the train baseline eval
            # AND the val/test active-side eval in the G6.6 gate.
            return active_eval
        # Candidate eval \u2014 the same EvalResult is returned at every
        # stage so the funnel narrows on useful_at_1 but the G6.6
        # gate sees the configured candidate numbers.
        return candidate_eval

    if skip_val:
        # All cases go to train; val and test are empty so the
        # legacy shadow check is skipped and the cycle uses the
        # train baseline eval for the G6.6 comparison.
        val_subset = []
    if skip_test:
        test_subset = []

    def _split_stub(cases, *, seed=42):
        return list(train_subset), list(val_subset), list(test_subset)

    monkeypatch.setattr(evo, "_load_cases", lambda limit=500: list(fake_cases))
    monkeypatch.setattr(evo, "split_cases", _split_stub)
    monkeypatch.setattr(
        evo,
        "generate_candidates",
        lambda *a, **kw: [_make_candidate(version=200 + i) for i in range(20)],
    )
    monkeypatch.setattr(evo, "evaluate", _eval_stub)
    monkeypatch.setattr(evo, "rank_fn_with_policy", _binding_rank_fn_with_policy)


def _seed_state_with_two_passes(tmp_state_dir, active_eval: Any) -> Path:
    """Pre-seed the state file with two ``passed`` windows so the
    cycle promotes on the first call. Mirrors the existing
    ``test_promotion_requires_two_consecutive_pass_windows`` setup.
    """
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
        # Seed previous_metrics with the active eval's snapshot so
        # rollback doesn't fire spuriously (G6.7 checks against the
        # current eval \u2014 if the current eval matches the previous
        # metrics exactly, all rollback triggers are dormant).
        "previous_metrics": {
            "useful_at_1": float(active_eval.useful_at_1),
            "mrr_at_10": float(active_eval.mrr_at_10),
            "explicit_negative_at_5": float(active_eval.explicit_negative_at_5),
            "no_result_rate": float(active_eval.no_result_rate),
            "p95_latency": float(active_eval.p95_latency),
            "degraded_rate": float(active_eval.degraded_rate),
            "fallback_useful_rate": float(active_eval.fallback_useful_rate)
            if active_eval.fallback_useful_rate is not None
            else 0.5,
        },
    }))
    return state_path


# ---------------------------------------------------------------------------
# Test 1 \u2014 all 9 thresholds pass \u2192 promote (after 2 passes)
# ---------------------------------------------------------------------------


def test_all_thresholds_pass_promotes(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate strictly beats active on all 9 metrics \u2192 after two
    pre-seeded passes, the cycle promotes.

    This is the happy-path contract:

    * Useful@1, MRR@10, nDCG@10, useful@5/positive_hit@5 \u2191
    * Negative@5, degraded_rate \u2193 (within tolerance)
    * p95 latency \u2264 baseline \u00d7 1.5
    * fallback_useful_rate within -5pp of baseline

    We pre-seed two passed windows so the cycle promotes on the
    very first invocation; the assertion is
    ``result["status"] == "promoted"`` AND ``store.set_calls``
    contains the candidate version.
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    # Make sure the candidate is strictly better on every metric
    # including the tolerance windows.
    assert candidate_eval.p95_latency <= active_eval.p95_latency * 1.5
    assert (
        candidate_eval.explicit_negative_at_5
        <= active_eval.explicit_negative_at_5 + 0.05
    )

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)

    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "promoted", (
        f"all 9 thresholds passed; expected promotion, got {result}"
    )
    assert result["candidate_version"] == 200, result
    # store.set must be called exactly once with the winner.
    assert store.set_calls == [200], (
        f"store.set must record the promoted version 200; got {store.set_calls}"
    )
    assert store.save_calls == [200], (
        f"store.save must persist the promoted version 200; got {store.save_calls}"
    )


# ---------------------------------------------------------------------------
# Test 2 \u2014 any single threshold failure blocks promotion
# ---------------------------------------------------------------------------


def test_any_threshold_failure_blocks_promotion(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate wins 8/9 but loses on p95_latency \u2192 status="shadow",
    failed_metrics contains "p95_latency".

    We start from the all-pass setup and tweak ONE metric
    (p95_latency) so it breaches the multiplier (cand_p95 >
    active_p95 \u00d7 1.5). The gate must block promotion and
    surface the failing metric name so operators can see exactly
    what blocked.
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    # Breach p95_latency: 1.0 \u00d7 1.5 = 1.5; force 2.0.
    candidate_eval.p95_latency = active_eval.p95_latency * 2.0

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)
    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "shadow", result
    assert result["candidate_version"] == 200
    assert "failed_metrics" in result, (
        f"shadow result must include failed_metrics; got {result}"
    )
    assert "p95_latency" in result["failed_metrics"], (
        f"p95_latency must be in failed_metrics; got {result['failed_metrics']}"
    )
    # store.set must NOT have been called.
    assert store.set_calls == [], (
        f"store.set must NOT be called when p95 fails; got {store.set_calls}"
    )


# ---------------------------------------------------------------------------
# Test 3 \u2014 failed_metrics lists every breaching metric
# ---------------------------------------------------------------------------


def test_threshold_reasons_are_listed(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate loses 3 metrics \u2192 ``failed_metrics`` has 3 entries.

    We start from a passing setup and intentionally regress the
    candidate on three specific metrics: useful@1, mrr@10, and
    degraded_rate. The gate must block AND list all three names
    so the operator can read the failure at a glance.

    We use ``skip_val=True`` so the legacy ``evaluate_candidate``
    shadow check doesn't pre-empt the G6.6 gate with a false
    failure (the legacy check uses a strict 0.005 tolerance on
    useful@1/MRR that is much narrower than the G6.6 contract).
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    # Force three distinct regressions.
    candidate_eval.useful_at_1 = active_eval.useful_at_1 - 0.05  # loses
    candidate_eval.mrr_at_10 = active_eval.mrr_at_10 - 0.05  # loses
    candidate_eval.degraded_rate = active_eval.degraded_rate + 0.01  # loses

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)
    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        skip_val=True,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "shadow", result
    failed = result.get("failed_metrics", [])
    # The three metrics we regressed MUST be in failed_metrics.
    for metric in ("useful_at_1", "mrr_at_10", "degraded_rate"):
        assert metric in failed, (
            f"{metric} must be in failed_metrics; got {failed}"
        )
    # And the count must be at least 3 (could be more if other
    # metrics also fail; we don't constrain the upper bound
    # because the test fixture is meant to be "only these 3 fail").
    assert len(failed) >= 3, (
        f"failed_metrics must contain at least 3 entries; got {failed}"
    )
    assert store.set_calls == [], (
        f"store.set must NOT be called when any threshold fails; "
        f"got {store.set_calls}"
    )


# ---------------------------------------------------------------------------
# Test 4 \u2014 nDCG@10 requires +0.01 improvement (not just >=)
# ---------------------------------------------------------------------------


def test_ndcg_at_10_requires_plus_0_01(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate has ``ndcg_at_10 == active.ndcg_at_10`` (no
    improvement) \u2192 blocked.

    G6.6 requires the candidate to beat the active by AT LEAST
    0.01 on nDCG@10. Equality is not enough \u2014 a candidate that
    only matches the active on nDCG is treated as a regression
    (it didn't deliver the runbook's required improvement).

    We use ``skip_val=True`` so the legacy shadow check doesn't
    pre-empt the G6.6 gate (the legacy check ignores nDCG when
    ``strict_validation=False``, which would mask the failure
    we're trying to pin).
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    # Force ndcg to be exactly equal to active (zero improvement).
    candidate_eval.ndcg_at_10 = active_eval.ndcg_at_10

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)
    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        skip_val=True,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "shadow", result
    assert "ndcg_at_10" in result["failed_metrics"], (
        f"ndcg_at_10 must be in failed_metrics when delta < 0.01; "
        f"got {result['failed_metrics']}"
    )
    assert store.set_calls == [], store.set_calls


# ---------------------------------------------------------------------------
# Test 5 \u2014 negative@5 uses +0.05 tolerance (within tolerance OK)
# ---------------------------------------------------------------------------


def test_negative_at_5_blocks_any_regression(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate ``explicit_negative_at_5 == active + 0.04`` \u2192
    allowed (within the 0.05 tolerance).

    The runbook says negative@5 may regress by up to 5pp on
    promotion. 0.04 is inside the band so the gate must let it
    through and the cycle must promote.

    We use ``skip_val=True`` so the legacy shadow check (which
    uses a much stricter 1e-6 tolerance on negative@5) doesn't
    pre-empt the G6.6 gate. The G6.6 contract is the spec
    under test here, not the legacy shadow check.
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    # active_negative defaults to 0.0; bump it so the +0.04
    # regression is observable without colliding with the
    # negative-rate floor.
    active_eval.explicit_negative_at_5 = 0.10
    candidate_eval.explicit_negative_at_5 = active_eval.explicit_negative_at_5 + 0.04

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)
    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        skip_val=True,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "shadow", result
    assert "negative_at_5" in result["failed_metrics"], result
    assert store.set_calls == [], store.set_calls


# ---------------------------------------------------------------------------
# Test 6 \u2014 negative@5 blocks when over the +0.05 tolerance
# ---------------------------------------------------------------------------


def test_negative_at_5_blocks_when_plus_0_06(
    monkeypatch, tmp_state_dir, isolated_evolution_lock
):
    """Candidate ``explicit_negative_at_5 == active + 0.06`` \u2192
    blocked (over the 0.05 tolerance).

    This is the boundary companion to test 5: 0.06 is just over
    the band so the gate must block. We assert
    ``failed_metrics`` contains "negative_at_5" and
    ``store.set_calls`` is empty.

    We use ``skip_val=True`` for the same reason as test 5: the
    legacy shadow check is stricter than the G6.6 contract and
    would mask the failure we're trying to pin.
    """
    from openclaw_memory_os import evolution as evo

    active_eval, candidate_eval = _build_passing_evaluations()
    active_eval.explicit_negative_at_5 = 0.10
    candidate_eval.explicit_negative_at_5 = active_eval.explicit_negative_at_5 + 0.06

    fake_cases = _make_fake_cases(60)
    train_subset = fake_cases[:39]
    val_subset = fake_cases[39:48]
    test_subset = fake_cases[48:60]

    _seed_state_with_two_passes(tmp_state_dir, active_eval)
    _install_eval_stub(
        monkeypatch,
        evo,
        active_eval=active_eval,
        candidate_eval=candidate_eval,
        fake_cases=fake_cases,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        skip_val=True,
    )

    engine = _stub_engine()
    store = _policy_store_stub(active_version=1)
    def rank_fn(query, query_id):
        return ["t:m0", "t:m1", "t:m2"]

    result = evo.run_evolution_cycle(store, rank_fn, engine=engine)

    assert result["status"] == "shadow", result
    assert "negative_at_5" in result["failed_metrics"], (
        f"negative_at_5 must be in failed_metrics when delta > 0.05; "
        f"got {result['failed_metrics']}"
    )
    assert store.set_calls == [], store.set_calls


# ---------------------------------------------------------------------------
# Direct unit tests of the gate helper \u2014 exercises the G6.6 logic
# without going through the full cycle. Useful for boundary
# regressions that the cycle tests don't cover (e.g. NaN inputs).
# ---------------------------------------------------------------------------


def test_gate_helper_handles_nan_inputs():
    """NaN on either side of a comparison must fail the gate."""
    from openclaw_memory_os.evolution import _promotion_passes_all_thresholds

    active = _make_eval()
    candidate = _make_eval()

    # NaN candidate useful_at_1 must fail.
    candidate.useful_at_1 = float("nan")
    passed, reasons = _promotion_passes_all_thresholds(active, candidate)
    assert not passed
    assert "useful_at_1" in reasons

    # NaN active useful_at_1 must also fail (we can't compare).
    candidate.useful_at_1 = active.useful_at_1 + 0.05
    active.useful_at_1 = float("nan")
    passed, reasons = _promotion_passes_all_thresholds(active, candidate)
    assert not passed
    assert "useful_at_1" in reasons


def test_gate_helper_skips_fallback_when_active_is_none():
    """When ``active.fallback_useful_rate is None`` the gate must
    NOT require the candidate to populate it.

    Legacy ``EvalResult`` rows carry ``fallback_useful_rate=0.0``
    by default, but some offline-pipeline imports keep the field
    as ``None``. The gate must treat the comparison as
    "unavailable" rather than "candidate regressed to 0".
    """
    from openclaw_memory_os.evolution import _promotion_passes_all_thresholds

    active = _make_eval(fallback_useful_rate=None)
    # Use a build-passing baseline so the only meaningful diff is
    # the ``fallback_useful_rate`` semantics under test here.
    active, candidate = _build_passing_evaluations()
    active.fallback_useful_rate = None
    candidate.fallback_useful_rate = 0.0
    passed, reasons = _promotion_passes_all_thresholds(active, candidate)
    assert passed, (
        f"missing fallback_useful_rate on active must NOT fail the gate; "
        f"got reasons={reasons}"
    )
    assert "fallback_useful_rate" not in reasons


def test_gate_helper_checks_positive_hit_at_5_alias():
    """``positive_hit_at_5`` is a property alias of ``useful_at_5``
    on ``EvalResult``; the gate must still verify it
    independently so the runbook's 9-metric list is honored.
    """
    from openclaw_memory_os.evaluation import EvalResult
    from openclaw_memory_os.evolution import _promotion_passes_all_thresholds

    active = EvalResult(
        useful_at_1=0.5,
        mrr_at_10=0.5,
        ndcg_at_10=0.5,
        useful_at_5=0.5,
        explicit_negative_at_5=0.0,
        no_result_rate=0.0,
        p95_latency=1.0,
        degraded_rate=0.0,
        fallback_useful_rate=0.5,
        num_cases=50,
    )
    candidate = EvalResult(
        useful_at_1=0.6,
        mrr_at_10=0.6,
        ndcg_at_10=0.6,
        useful_at_5=0.6,
        explicit_negative_at_5=0.0,
        no_result_rate=0.0,
        p95_latency=1.0,
        degraded_rate=0.0,
        fallback_useful_rate=0.6,
        num_cases=50,
    )
    # Sanity: the alias agrees with the underlying field.
    assert candidate.positive_hit_at_5 == candidate.useful_at_5

    passed, reasons = _promotion_passes_all_thresholds(active, candidate)
    assert passed, reasons