"""Tests for v0.3.0.x additions to the offline evaluation layer.

Focus areas
-----------

* ``CandidatePool`` construction, normalisation, accessors, and
  deduplication behaviour.
* ``_judged_ndcg_at_10`` / equivalent graded relevance metric:
  per-case computation, the explicit ``None`` contract, and the
  averaging behaviour applied by ``evaluate``.
* ``_useful_superseded_fallback_rate``: the "never fell back"
  vs "fell back and was useless" distinction.
* Backward compatibility of the augmented ``evaluate`` API and the
  new ``EvalResult`` fields (incl. ``corpus_snapshot_id`` being
  ``None`` when unavailable).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from openclaw_memory_os.evaluation import (
    CandidatePool,
    _EvaluationCase,
    _judged_ndcg_at_10,
    _ndcg_at_k,
    _useful_superseded_fallback_rate,
    evaluate,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _case(positives, negatives=None, query_text="q", query_id="qid"):
    return _EvaluationCase(query_id, query_text, set(positives), set(negatives or []))


def _record(
    key: str,
    *,
    status: str = "active",
    channel: str = "dense",
    score: float = 1.0,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a candidate dict shaped like a recall result row."""
    out: Dict[str, Any] = {
        "candidate_key": key,
        "status": status,
        "channel": channel,
        "score": score,
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# CandidatePool: construction + normalisation
# ---------------------------------------------------------------------------


class TestCandidatePoolBasics:
    def test_empty_factory_has_deterministic_lists(self) -> None:
        pool = CandidatePool.empty(query_id="q1", query_text="hello")
        assert pool.query_id == "q1"
        assert pool.query_text == "hello"
        assert pool.dense_active == []
        assert pool.lexical_active == []
        assert pool.superseded == []
        assert pool.corpus_snapshot_id is None
        assert pool.extra == {}
        # summary should not crash
        s = pool.summary()
        assert s["query_id"] == "q1"
        assert s["dense_active_count"] == 0
        assert s["lexical_active_count"] == 0
        assert s["superseded_count"] == 0
        assert s["corpus_snapshot_id"] is None

    def test_from_ranked_routes_by_channel_and_status(self) -> None:
        ranked = [
            _record("a", channel="dense", score=0.9),
            _record("b", channel="lexical", score=0.8),
            _record("c", status="superseded", channel="dense"),
            _record("d", status="superseded", channel="lexical"),
        ]
        pool = CandidatePool.from_ranked("qid", "text", ranked)
        assert pool.dense_active_keys == ["a"]
        assert pool.lexical_active_keys == ["b"]
        assert pool.superseded_keys == ["c", "d"]

    def test_from_ranked_status_expired_is_fallback_material(self) -> None:
        # Unknown / expired statuses must NOT be routed to *_active;
        # they should appear under superseded so they never silently
        # count as "active candidates".
        ranked = [
            _record("a", channel="dense"),
            _record("b", status="expired", channel="dense"),
            _record("c", status="weird", channel="lexical"),
        ]
        pool = CandidatePool.from_ranked("qid", "text", ranked)
        assert pool.dense_active_keys == ["a"]
        assert set(pool.superseded_keys) == {"b", "c"}
        assert pool.lexical_active_keys == []

    def test_from_ranked_default_channel_is_dense(self) -> None:
        # For backward compatibility with existing recall responses
        # that don't set ``channel`` at all.
        ranked = [
            _record("a"),
            {"candidate_key": "b", "score": 0.5},
        ]
        pool = CandidatePool.from_ranked("qid", "text", ranked)
        assert pool.dense_active_keys == ["a", "b"]

    def test_from_ranked_accepts_dataclass_records(self) -> None:
        # Records that are not dicts (e.g. dataclass-like) must still
        # normalise into a usable dict.
        class Hit:
            candidate_key = "x"
            score = 0.42
            channel = "lexical"

        pool = CandidatePool.from_ranked("qid", "text", [Hit()])
        assert pool.lexical_active_keys == ["x"]
        assert pool.dense_active_keys == []

    def test_from_ranked_normalises_missing_score(self) -> None:
        ranked = [{"candidate_key": "a"}]  # no score
        pool = CandidatePool.from_ranked("qid", "text", ranked)
        assert pool.dense_active[0]["score"] == 0.0

    def test_from_ranked_preserves_corpus_snapshot_id(self) -> None:
        ranked = [_record("a")]
        pool = CandidatePool.from_ranked(
            "qid", "text", ranked, corpus_snapshot_id="snap-2026-07"
        )
        assert pool.corpus_snapshot_id == "snap-2026-07"
        assert pool.summary()["corpus_snapshot_id"] == "snap-2026-07"

    def test_from_ranked_extra_is_copied(self) -> None:
        ranked = [_record("a")]
        extra = {"trace_id": "abc", "engine": "v3"}
        pool = CandidatePool.from_ranked("qid", "text", ranked, extra=extra)
        assert pool.extra == extra
        # And the dict identity must not leak:
        extra["trace_id"] = "mutated"
        assert pool.extra["trace_id"] == "abc"


# ---------------------------------------------------------------------------
# CandidatePool: ordering + deduplication
# ---------------------------------------------------------------------------


class TestCandidatePoolAccessors:
    def _two_channel_pool(self) -> CandidatePool:
        ranked = [
            _record("a", channel="dense"),
            _record("b", channel="dense"),
            _record("a", channel="dense"),  # duplicate within dense
            _record("c", channel="lexical"),
            _record("b", channel="lexical"),  # cross-channel dup
            _record("s1", status="superseded"),
        ]
        return CandidatePool.from_ranked("qid", "text", ranked)

    def test_active_keys_is_deduped_dense_first(self) -> None:
        pool = self._two_channel_pool()
        # Order: dense-first, lexical-fills-after (no re-duplication).
        assert pool.active_keys == ["a", "b", "c"]

    def test_dense_and_lexical_keys_are_deduped_each(self) -> None:
        pool = self._two_channel_pool()
        assert pool.dense_active_keys == ["a", "b"]
        assert pool.lexical_active_keys == ["c", "b"]
        assert pool.superseded_keys == ["s1"]

    def test_summary_counts_unique_active(self) -> None:
        pool = self._two_channel_pool()
        s = pool.summary()
        assert s["active_unique"] == 3  # a, b, c
        assert s["superseded_unique"] == 1
        assert s["dense_active_count"] == 3  # raw count, not deduped
        assert s["lexical_active_count"] == 2


# ---------------------------------------------------------------------------
# _judged_ndcg_at_10 graded relevance helper
# ---------------------------------------------------------------------------


class TestJudgedNdcgAt10:
    def test_returns_none_when_no_judged(self) -> None:
        # Critical contract: no data == None, not 0.0.
        assert _judged_ndcg_at_10([]) is None

    def test_perfect_ranking_scores_one(self) -> None:
        # Top-2 graded: relevance (3,2) at ranks (1,2). Ideal == actual.
        judged = [("a", 3), ("b", 2), ("c", 0)]
        val = _judged_ndcg_at_10(judged, k=10)
        assert val == pytest.approx(1.0)

    def test_perfect_ranking_against_ideal(self) -> None:
        # Sub-optimal ordering.
        judged = [("a", 1), ("b", 3), ("c", 2)]
        # Compare against a manual computation.
        def dcg(items, k):
            s = 0.0
            for i, (_k, g) in enumerate(items[:k]):
                g = max(0, int(g))
                if g > 0:
                    s += (2 ** g - 1) / math.log2(i + 2)
            return s

        d = dcg(judged, 10)
        ideal_grades = sorted((g for (_k, g) in judged if g > 0), reverse=True)
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
        expected = d / idcg
        assert _judged_ndcg_at_10(judged, k=10) == pytest.approx(expected)

    def test_negative_grades_clamped_to_zero(self) -> None:
        # Negative grades must not penalise; they're treated as 0-gain
        # in the actual ranking AND excluded from the ideal ordering.
        judged = [("a", -5), ("b", 2)]
        val = _judged_ndcg_at_10(judged, k=10)
        # Manual expected: actual DCG = 3/log2(3); ideal DCG = 3/1 = 3
        # → val ~= 3/log2(3) / 3 ~= 0.631
        expected = (3 / math.log2(3)) / 3.0
        assert val == pytest.approx(expected)
        # And the value is strictly between 0 and 1 (clamping didn't
        # collapse the metric to either extreme).
        assert val is not None
        assert 0.0 < val < 1.0

    def test_all_zero_graded_returns_zero(self) -> None:
        # No positive grades: not None, just 0.0 (matches dashboard
        # contract: distinguishable from "no judgement").
        judged = [("a", 0), ("b", 0)]
        assert _judged_ndcg_at_10(judged, k=10) == 0.0

    def test_truncates_at_k(self) -> None:
        # Beyond k we ignore items.
        judged = [("a", 3)] + [("x", 0) for _ in range(20)]
        val = _judged_ndcg_at_10(judged, k=1)
        assert val == pytest.approx(1.0)

    def test_in_range_zero_to_one(self) -> None:
        judged = [("a", 3), ("b", 2), ("c", 1), ("d", 0)]
        val = _judged_ndcg_at_10(judged, k=10)
        assert val is not None
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# _useful_superseded_fallback_rate
# ---------------------------------------------------------------------------


class TestUsefulSupersededFallbackRate:
    def test_none_when_no_expansion(self) -> None:
        # No fallback band → None (we must not fake a 0).
        v = _useful_superseded_fallback_rate(
            active_hits=["a", "b"],
            fallback_hits=[],
            judged_positives={"a", "b"},
        )
        assert v is None

    def test_none_when_no_judged_positives(self) -> None:
        # No positives means we can't score usefulness.
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["x", "y"],
            judged_positives=set(),
        )
        assert v is None

    def test_one_when_every_fallback_is_useful(self) -> None:
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["p", "q"],
            judged_positives={"p", "q"},
        )
        assert v == pytest.approx(1.0)

    def test_zero_when_none_useful(self) -> None:
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["x", "y"],
            judged_positives={"a"},  # x, y not useful
        )
        assert v == pytest.approx(0.0)

    def test_partial_fraction(self) -> None:
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["p", "x", "q", "y"],
            judged_positives={"p", "q"},
        )
        # 2 useful / 4 expansion = 0.5
        assert v == pytest.approx(0.5)

    def test_overlap_with_active_is_excluded(self) -> None:
        # Items already in active_hits must NOT count as "new"
        # expansions (the engine didn't actually surface them via
        # the fallback).
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["a", "p"],
            judged_positives={"a", "p"},
        )
        # 1 expansion ("p") and it is useful → 1.0
        assert v == pytest.approx(1.0)

    def test_empty_keys_are_filtered(self) -> None:
        v = _useful_superseded_fallback_rate(
            active_hits=["a"],
            fallback_hits=["", None],  # type: ignore[list-item]
            judged_positives={"a"},
        )
        # No real expansion keys → None (matches "never fell back")
        assert v is None


# ---------------------------------------------------------------------------
# evaluate() backward compatibility with new metrics
# ---------------------------------------------------------------------------


class TestEvaluateBackwardCompatibility:
    def test_evaluate_returns_legacy_and_new_fields(self) -> None:
        def rank_fn(query, qid):
            return ["a", "b", "c"]

        case = _case(query_id="qid", positives=["a", "b"])
        result = evaluate([case], rank_fn)
        # Legacy fields continue to populate.
        assert result.num_cases == 1
        assert result.recall_at_5 == pytest.approx(1.0)
        assert result.ndcg_at_10 == pytest.approx(1.0)
        # New fields default to None / unavailable.
        assert result.judged_ndcg_at_10 is None
        assert result.useful_superseded_fallback_rate is None
        assert result.num_judged_cases == 0
        assert result.corpus_snapshot_id is None
        assert result.judged_ndcg_status == "unavailable"
        assert result.fallback_rate_status == "unavailable"

    def test_evaluate_to_dict_includes_null_new_fields(self) -> None:
        def rank_fn(query, qid):
            return ["a"]

        case = _case(query_id="q1", positives=["a"])
        result = evaluate([case], rank_fn)
        d = result.to_dict()
        # All legacy fields present.
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
            assert key in d
        # New fields present with None where unavailable.
        assert d["judged_ndcg_at_10"] is None
        assert d["useful_superseded_fallback_rate"] is None
        assert d["num_judged_cases"] == 0
        assert d["corpus_snapshot_id"] is None
        assert d["judged_ndcg_status"] == "unavailable"
        assert d["fallback_rate_status"] == "unavailable"

    def test_evaluate_handles_empty_rank(self) -> None:
        def rank_fn(query, qid):
            return []

        case = _case(query_id="qid", positives=["a"])
        result = evaluate([case], rank_fn)
        # No crash, no_result_rate captures it.
        assert result.no_result_rate == pytest.approx(1.0)
        assert result.judged_ndcg_at_10 is None  # still no judgement

    def test_evaluate_zero_cases_returns_default(self) -> None:
        result = evaluate([], lambda q, qid: [])
        assert result.num_cases == 0
        assert result.judged_ndcg_at_10 is None
        assert result.useful_superseded_fallback_rate is None
        assert result.corpus_snapshot_id is None


# ---------------------------------------------------------------------------
# cross-metric consistency smoke check
# ---------------------------------------------------------------------------


def test_binary_relevance_ndcg_at_k_matches_judged_ndcg_at_10_when_grades_are_binary():
    """Sanity: when graded judgements are pure 0/1, the binary nDCG@k
    helper and the graded helper should agree (graded == binary)."""
    ranked = ["a", "b", "c"]
    relevant = {"a"}
    binary = _ndcg_at_k(ranked, relevant, 10, set())
    graded = _judged_ndcg_at_10([(k, 1 if k in relevant else 0) for k in ranked], k=10)
    assert graded is not None
    assert graded == pytest.approx(binary)
