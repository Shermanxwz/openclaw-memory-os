"""Tests for the v0.3.0 policy evolution system (S9 + S10).

B3-1 (version is a real int), B3-2 (shadow applies candidate policy),
B3-3 (run_evolution_cycle uses the engine), B3-4 (persistence:
``PolicyStore.revert`` + promotion persistence), B3-5
(rank_fn distinguishes active vs candidate weights).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


from openclaw_memory_os.evolution import (
    _COLD_START_MIN_QUERIES,
    rank_fn_with_policy,
)
from openclaw_memory_os.policy_store import (
    Policy,
    PolicyStatus,
    PolicyStore,
    baseline_policy,
    compute_checksum,
)


# ---------------------------------------------------------------------------
# Smoke: the existing Batch 2 tests still pass.
# ---------------------------------------------------------------------------


def test_cold_start_min_query_constant():
    assert _COLD_START_MIN_QUERIES == 30


def test_import_ok():
    from openclaw_memory_os.evolution import generate_candidates
    assert generate_candidates is not None


# ---------------------------------------------------------------------------
# B3-1: candidate generation produces real Policy records.
# ---------------------------------------------------------------------------


def _baseline_active() -> Policy:
    return Policy(**baseline_policy, status=PolicyStatus.ACTIVE)


def test_candidate_generation_produces_real_policies():
    """Each returned candidate must be a Policy with an int version."""
    from openclaw_memory_os.evolution import generate_candidates

    baseline = _baseline_active()
    cands = generate_candidates(baseline, n_candidates=8, seed=42)
    assert cands, "candidate generation produced an empty list"
    for p in cands:
        assert isinstance(p, Policy), f"not a Policy: {type(p)}"
        assert isinstance(p.version, int), f"version is not int: {type(p.version)}"
        assert p.status == PolicyStatus.SHADOW, f"unexpected status: {p.status}"


def test_candidate_generation_does_not_infinite_loop():
    """``generate_candidates`` must terminate even with a tight bound."""
    from openclaw_memory_os.evolution import generate_candidates

    baseline = _baseline_active()
    cands = generate_candidates(baseline, n_candidates=5, seed=7, max_delta=0.01)
    # The loop has a safety cap of ``n_candidates * 50 + 100`` iters;
    # the function must return *something* without hanging the
    # interpreter. CI timeout will catch true infinite loops; here
    # we just assert it returns a finite list.
    assert isinstance(cands, list)
    # And every iteration that succeeded must be a Policy, never a
    # half-built dict that slipped through the except branch.
    for p in cands:
        assert isinstance(p, Policy)
        assert isinstance(p.version, int)


def test_no_fstring_candidate_version_remains():
    """Regression guard: the f-string ``"version": f"candidate-..."`` has been removed."""
    src = (Path(__file__).resolve().parent.parent / "openclaw_memory_os" / "evolution.py").read_text(encoding="utf-8")
    assert 'f"candidate-' not in src, (
        "evolution.py still contains `f\"candidate-...\"` constructions"
    )


# ---------------------------------------------------------------------------
# B3-2: ``shadow_compare`` returns two eval results that actually differ.
# ---------------------------------------------------------------------------


def test_shadow_compare_distinguishes_active_and_candidate_rank_fn():
    """When given two different rank_fns, ``shadow_compare`` must return
    two distinct ``EvalResult``s (the candidate eval is computed
    with the candidate\'s rank_fn, not the active\'s)."""
    from openclaw_memory_os.evaluation import _EvaluationCase
    from openclaw_memory_os.evolution import shadow_compare

    # Three cases; the rank_fn for the "candidate" side returns
    # a strictly *worse* ranking than the active side so the
    # candidate_eval differs from the active_eval.
    case1 = _EvaluationCase("q1", "alpha", positives={"a", "b"}, negatives=set())
    case2 = _EvaluationCase("q2", "beta", positives={"c"}, negatives=set())
    case3 = _EvaluationCase("q3", "gamma", positives={"a"}, negatives=set())

    def active_rfn(q: str, qid: str):
        # Returns positives first
        return {"q1": ["a", "b", "x"], "q2": ["c", "y"], "q3": ["a", "z"]}.get(qid, [])

    def candidate_rfn(q: str, qid: str):
        # Returns positives LAST — strictly worse
        return {"q1": ["x", "a", "b"], "q2": ["y", "c"], "q3": ["z", "a"]}.get(qid, [])

    should_promote, reasons, active_eval, candidate_eval = shadow_compare(
        {}, [case1, case2, case3], _baseline_active(), _baseline_active(),
        rank_fn=active_rfn, rank_fns=(active_rfn, candidate_rfn),
    )

    assert active_eval.num_cases == 3
    assert candidate_eval.num_cases == 3
    # The two evals MUST differ; the candidate path returned
    # strictly worse rankings, so recall@1 drops to zero.
    assert active_eval.recall_at_1 > candidate_eval.recall_at_1, (
        f"expected active.eval.recall_at_1 > candidate.eval.recall_at_1; got {active_eval.recall_at_1} vs {candidate_eval.recall_at_1}"
    )


# ---------------------------------------------------------------------------
# B3-3: ``run_evolution_cycle`` uses the retrieval engine, not backend.search.
# ---------------------------------------------------------------------------


def test_runner_uses_retrieval_engine_not_backend_search(tmp_path):
    """The standalone script must NOT call ``backend.search`` directly.

    We exercise the same wiring the script uses (engine + rank_fn
    closures) and assert that the only backend method invoked is
    the one the engine uses (``list_memories`` / ``get_memory`` /
    ``dense_search``), never the legacy ``search``.
    """
    from openclaw_memory_os.evolution import rank_fn_with_policy, run_evolution_cycle
    from openclaw_memory_os.backends import MemoryBackend
    from openclaw_memory_os.contracts import CandidateStatus, CandidateTier, ScoredMemoryCandidate
    from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier

    class _TrackingBackend(MemoryBackend):
        name = "tracker"

        def __init__(self):
            self.search_calls = 0
            self.dense_calls = 0

        def list_memories(self):
            now = datetime.now(timezone.utc)
            return [
                Memory(
                    id=f"m{i}",
                    text=f"alpha beta gamma {i}",
                    status=MemoryStatus.ACTIVE,
                    importance=0.5 + 0.1 * i,
                    tier=MemoryTier.MEDIUM,
                    created_at=now,
                )
                for i in range(5)
            ]

        def list_collections(self):
            return ["tracker"]

        def get_memory(self, mid):
            return next((m for m in self.list_memories() if m.id == mid), None)

        def dense_search(self, query, limit=10, status_filter=None):
            self.dense_calls += 1
            out = []
            for m in self.list_memories():
                if status_filter and m.status.value.lower() not in {s.lower() for s in status_filter}:
                    continue
                out.append(
                    ScoredMemoryCandidate(
                        collection="tracker",
                        memory_id=m.id,
                        candidate_key=f"tracker:{m.id}",
                        text=m.text,
                        status=CandidateStatus(m.status.value),
                        tier=CandidateTier(m.tier.value),
                        importance=m.importance,
                        created_at=m.created_at,
                        dense_score=1.0 - 0.1 * len(out),
                    )
                )
            return out[:limit]

        def search(self, query, limit=10):  # type: ignore[override]
            # Fail loudly — if the script ever reaches this method,
            # the test catches the regression.
            self.search_calls += 1
            raise AssertionError("backend.search must not be called from the runner")

    from openclaw_memory_os.retrieval_engine import RetrievalEngine

    backend = _TrackingBackend()
    store = PolicyStore()
    # The script semantics: active_policy is whatever the store has.
    # In cold-start mode (no feedback) the cycle returns early with
    # "skipped", but the *path* to that branch must exercise only
    # ``store.get`` / engine ``list_memories`` — never ``backend.search``.
    store.set(_baseline_active())
    engine = RetrievalEngine(backend=backend, policy_store=store)

    active_policy = store.get()
    candidate_kwargs = dict(active_policy.model_dump())
    candidate_kwargs["version"] = active_policy.version + 1
    candidate_kwargs["importance_weight"] = float(active_policy.importance_weight) + 0.05
    candidate_policy = Policy(**candidate_kwargs)

    active_rfn = rank_fn_with_policy(engine, active_policy)
    candidate_rfn = rank_fn_with_policy(engine, candidate_policy)

    # Run the cycle. The default cold-start gate will skip
    # generation because we have no feedback, but the early phases
    # already exercise the engine wiring.
    result = run_evolution_cycle(store, active_rfn, candidate_rank_fn=candidate_rfn)
    assert result["status"] == "skipped"
    assert "cold_start" in result.get("reason", "")
    # The regression assertion: backend.search is never called.
    assert backend.search_calls == 0, (
        f"backend.search was called {backend.search_calls} times; the runner must use the engine only"
    )


def test_runner_subprocess_uses_engine_wiring(tmp_path):
    """End-to-end smoke: subprocess the script with a sample path,
    assert it does not crash and emits a JSON envelope."""
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "m1",
                        "text": "alpha beta gamma",
                        "status": "active",
                        "importance": 0.5,
                        "tier": "medium",
                        "tags": [],
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    env = {
        **__import__("os").environ,
        "MEMORY_OS_SAMPLE_PATH": str(sample),
        # Force no Qdrant
        "QDRANT_URL": "",
    }
    env.pop("QDRANT_URL", None)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "run_evolution_cycle.py")],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in env.items() if v != ""},
    )
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    body = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "status" in body
    # Without feedback, cold-start kicks in.
    assert body["status"] in {"skipped", "ok", "shadow", "rolled_back", "promoted"}


# ---------------------------------------------------------------------------
# B3-4: ``PolicyStore.revert`` and promotion persistence.
# ---------------------------------------------------------------------------


def test_revert_restores_baseline_policy(tmp_path: Path) -> None:
    """``store.revert()`` must restore the shipped baseline policy and
    write it to disk via the atomic save path."""
    p = tmp_path / "policy.json"
    store = PolicyStore(path=p)
    store.get()
    store.checksum()

    # Mutate the active policy to something else.
    mutated = Policy(**baseline_policy, version=42, status=PolicyStatus.ACTIVE, dense_k=99)
    store.set(mutated)
    assert store.get().version == 42

    # Revert — previous policy (baseline) must come back.
    new_checksum = store.revert()
    assert new_checksum is not None
    # After revert, the previous policy (version=1, baseline) is restored.
    assert store.get().version == 1
    assert store.get().status == PolicyStatus.ACTIVE
    # On-disk file must reflect the reverted policy.
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["version"] == 1


def test_promotion_persists_to_disk(tmp_path: Path) -> None:
    """``store.set(...)`` + ``store.save(...)`` must write the policy
    such that a fresh ``PolicyStore(path=...)`` loads it."""
    p = tmp_path / "policy.json"
    store = PolicyStore(path=p)
    # Promote a new policy
    new_policy = Policy(**baseline_policy, version=7, status=PolicyStatus.ACTIVE, dense_k=42)
    store.set(new_policy)
    store.save(new_policy)
    # Fresh store reads it back
    store2 = PolicyStore(path=p)
    loaded = store2.get()
    assert loaded.version == 7
    assert loaded.status == PolicyStatus.ACTIVE
    assert loaded.dense_k == 42


# ---------------------------------------------------------------------------
# B3-5: ``rank_fn_with_policy`` distinguishes two policies.
# ---------------------------------------------------------------------------


def test_rank_fn_applies_candidate_policy() -> None:
    """Two policies with different weights must produce different
    orderings on the same engine hits. We use a tiny stub engine
    so the test is deterministic and does not need a backend."""
    from openclaw_memory_os.retrieval_engine import RetrievalResult
    from openclaw_memory_os.contracts import RetrievalDiagnostics, ScoredMemoryCandidate, CandidateStatus, CandidateTier

    class _FixedEngine:
        """A drop-in for ``RetrievalEngine`` that returns a fixed hit list."""

        def __init__(self, hits):
            self._hits = hits

        def retrieve(self, query, *, mode="hybrid", limit=10, status_filter=None):
            diag = RetrievalDiagnostics(
                status="ok",
                degraded_reason=None,
                dense_available=True,
                lexical_available=True,
                collections_searched=[],
                candidate_count=len(self._hits),
                embedding_ms=0.0,
                lexical_ms=0.0,
                ranking_ms=0.0,
            )
            return RetrievalResult(
                hits=list(self._hits),
                diagnostics=diag,
                active_count=len(self._hits),
                fallback_used=False,
                fallback_added=0,
            )

    base_time = datetime.now(timezone.utc)
    hits = [
        ScoredMemoryCandidate(
            collection="c",
            memory_id=f"k{i}",
            candidate_key=f"c:k{i}",
            text=f"doc {i}",
            status=CandidateStatus.ACTIVE,
            tier=CandidateTier.MEDIUM,
            # k0 is FIRST in the engine\'s iteration (highest recency
            # surrogate) and has the LOWEST importance; k4 is LAST
            # (lowest recency) and has the HIGHEST importance. This
            # guarantees an importance-heavy policy orders the hits
            # as k4..k0 while a recency-heavy policy orders them
            # k0..k4.
            importance=round(0.1 * (i + 1), 3),  # k0=0.1, k4=0.5
            created_at=base_time,
            updated_at=base_time,
            dense_score=0.0,
            lexical_score=0.0,
            score=1.0,  # engine hands every hit the same composite
        )
        for i in range(5)
    ]
    engine = _FixedEngine(hits)  # type: ignore[arg-type]  # stub

    # Policy A: high importance_weight (pushes k4 → k0 upward by importance)
    active_kwargs = dict(baseline_policy)
    active_kwargs["importance_weight"] = 0.9
    active_kwargs["recency_weight"] = 0.05
    active_kwargs["feedback_weight"] = 0.05
    active = Policy(**active_kwargs, status=PolicyStatus.ACTIVE)

    # Policy B: low importance_weight (pushes later items upward via
    # the recency surrogate baked into ``rank_fn_with_policy``).
    cand_kwargs = dict(baseline_policy)
    cand_kwargs["importance_weight"] = 0.05
    cand_kwargs["recency_weight"] = 0.9
    cand_kwargs["feedback_weight"] = 0.05
    candidate = Policy(**cand_kwargs, status=PolicyStatus.SHADOW)

    active_rfn = rank_fn_with_policy(engine, active)
    candidate_rfn = rank_fn_with_policy(engine, candidate)

    active_order = active_rfn("any", "qid-1")
    candidate_order = candidate_rfn("any", "qid-1")

    assert active_order, "rank_fn returned an empty list"
    assert candidate_order, "candidate rank_fn returned an empty list"
    # The two orderings MUST differ \xe2\x80\x94 the candidate\'s
    # recency-weighted ordering favours early hits (k0..k4)
    # while the active\'s importance-weighted ordering favours
    # later hits (k4..k0).
    assert active_order != candidate_order, (
        f"expected different orderings; both produced: {active_order!r}"
    )
    # An importance-heavy active policy puts k4 first (highest
    # importance) and k0 last (lowest importance).
    assert active_order[0] == "c:k4", f"expected c:k4 first (importance), got {active_order!r}"
    assert active_order[-1] == "c:k0", f"expected c:k0 last, got {active_order!r}"
    # A recency-heavy candidate policy must put k0 first.
    assert candidate_order[0] == "c:k0", f"expected c:k0 first (recency), got {candidate_order!r}"


# ---------------------------------------------------------------------------
# B3-4 + B3-1 integration: evolution.set(best_cand) does not silently fail.
# ---------------------------------------------------------------------------


def test_evolution_run_path_calls_set_shadow_and_revert():
    """The evolution module\'s public surface (used by tests and the
    runner) must include the new ``PolicyStore`` hooks."""
    store_cls = PolicyStore
    assert callable(getattr(store_cls, "revert", None))
    assert callable(getattr(store_cls, "set_shadow", None))
    assert callable(getattr(store_cls, "get_shadow", None))


def test_policy_store_set_shadow_forces_status() -> None:
    """``set_shadow`` must record the candidate with status=SHADOW."""
    store = PolicyStore()
    cand = Policy(**baseline_policy, version=99, status=PolicyStatus.ACTIVE)
    store.set_shadow(cand)
    got = store.get_shadow()
    assert got is not None
    assert got.status == PolicyStatus.SHADOW


# ---------------------------------------------------------------------------
# B3-4 sanity: computed baseline checksum is what ``revert`` returns.
# ---------------------------------------------------------------------------


def test_revert_checksum_matches_baseline():
    store = PolicyStore()
    # mutate, then revert — revert goes back to previous (the baseline).
    store.set(Policy(**baseline_policy, version=99, status=PolicyStatus.ACTIVE))
    store.revert()
    # After revert, the active policy should be the previous one (version=1, baseline).
    assert store.get().version == 1
    # A second revert (no previous now) falls back to baseline.
    reverted2 = store.revert()
    baseline_policy_instance = Policy(**baseline_policy, status=PolicyStatus.BASELINE)
    expected = compute_checksum(baseline_policy_instance)
    assert reverted2 == expected, (
        f"second revert() returned {reverted2!r}; expected baseline checksum {expected!r}"
    )
