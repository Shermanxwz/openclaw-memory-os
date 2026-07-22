"""Tests for the G5.4 evaluation completeness additions.

Covers:

* ``EvalResult`` carries the v0.3.0.x metrics fields (case-baseline
  dataclass check — backwards-compatible defaults).
* ``useful_at_5`` is the fraction of *cases* where at least one
  top-5 hit is judged useful (not the legacy per-item ratio).
* ``explicit_negative_at_5`` excludes unjudged cases from both the
  numerator and the denominator — unjudged is NOT negative.
* ``no_result_rate`` increments when the ranker returns nothing.
* ``fallback_useful_rate`` increments when the structured ranker
  reports ``fallback_used=True`` AND a fallback key matches a
  judged positive.
* ``degraded_rate`` increments when the structured ranker reports
  ``degraded=True``.
* ``p50_latency`` / ``p95_latency`` are populated from the wall
  clock observed around each ``rank_fn`` call.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from openclaw_memory_os.evaluation import (
    EvalResult,
    _EvaluationCase,
    _RankFnOutcome,
    evaluate,
)


def _case(query_id, positives=None, negatives=None, query_text=""):
    return _EvaluationCase(
        query_id=query_id,
        query_text=query_text or query_id,
        positives=set(positives or []),
        negatives=set(negatives or []),
    )


# ---------------------------------------------------------------------------
# 1. EvalResult has the required fields.
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = (
    "useful_at_5",
    "explicit_negative_at_5",
    "no_result_rate",
    "fallback_useful_rate",
    "p50_latency",
    "p95_latency",
    "degraded_rate",
)


def test_eval_result_has_required_fields():
    field_names = {f.name for f in dataclasses.fields(EvalResult)}
    for field_name in REQUIRED_FIELDS:
        assert field_name in field_names, f"EvalResult missing field: {field_name}"
    # Defaults preserve backward compatibility (no value should crash
    # when the dataclass is instantiated with zero arguments).
    result = EvalResult()
    assert result.useful_at_5 == 0.0
    assert result.explicit_negative_at_5 == 0.0
    assert result.no_result_rate == 0.0
    assert result.fallback_useful_rate == 0.0
    assert result.p50_latency == 0.0
    assert result.p95_latency == 0.0
    assert result.degraded_rate == 0.0
    # And the field round-trips through to_dict().
    d = result.to_dict()
    for key in REQUIRED_FIELDS:
        assert key in d, f"EvalResult.to_dict() missing key: {key}"


# ---------------------------------------------------------------------------
# 2. useful_at_5 = fraction of cases where ≥1 top-5 hit is useful.
# ---------------------------------------------------------------------------


def test_useful_at_5_computes_correctly():
    """Two useful cases, two useless cases → useful_at_5 == 0.5."""
    cases = [
        _case("q1", positives=["a", "b"]),
        _case("q2", positives=["c"]),
        _case("q3", positives=["d"]),
        _case("q4", positives=["never_seen"]),
    ]

    def rank_fn(query_text, query_id):
        if query_id == "q1":
            return ["a", "x", "y", "z", "w"]
        if query_id == "q2":
            return ["c", "x", "y", "z", "w"]
        if query_id == "q3":
            return ["x", "y", "z", "w", "v"]  # no positive in top-5
        if query_id == "q4":
            return ["x", "y", "z", "w", "v"]  # no positive in top-5
        return []

    result = evaluate(cases, rank_fn)
    # 2 of 4 cases have a useful hit in top-5 → 0.5.
    assert result.useful_at_5 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. explicit_negative_at_5 excludes unjudged from both n and d.
# ---------------------------------------------------------------------------


def test_explicit_negative_only_excludes_unjudged():
    """3 cases (positive / negative / unjudged) → 0.5, NOT 0.667."""
    cases = [
        # case 1: top-5 contains a positive (useful=true) only.
        _case("q1", positives=["a"], negatives=[]),
        # case 2: top-5 contains an explicit negative.
        _case("q2", positives=[], negatives=["z"]),
        # case 3: top-5 contains only unjudged candidates (no
        # positive, no negative). Must be excluded from both
        # numerator AND denominator.
        _case("q3", positives=[], negatives=[]),
    ]

    def rank_fn(query_text, query_id):
        if query_id == "q1":
            return ["a", "b", "c", "d", "e"]
        if query_id == "q2":
            return ["z", "b", "c", "d", "e"]
        if query_id == "q3":
            return ["u", "v", "w", "x", "y"]
        return []

    result = evaluate(cases, rank_fn)
    # Denominator = 2 (judged cases only — q3 is unjudged).
    # Numerator   = 1 (only q2 has a top-5 negative).
    assert result.explicit_negative_at_5 == pytest.approx(0.5)
    # Sanity: an older "treat unjudged as negative" implementation
    # would have produced 2/3 ≈ 0.667, which we explicitly forbid.
    assert result.explicit_negative_at_5 != pytest.approx(2 / 3, abs=1e-3)


def test_explicit_negative_at_5_zero_when_no_judged_cases():
    """When no case has any judgement, the metric is 0.0 (not NaN)."""
    cases = [_case("q1"), _case("q2")]

    def rank_fn(query_text, query_id):
        return ["x", "y", "z"]

    result = evaluate(cases, rank_fn)
    assert result.explicit_negative_at_5 == 0.0


# ---------------------------------------------------------------------------
# 4. no_result_rate = fraction of cases with no hits.
# ---------------------------------------------------------------------------


def test_no_result_rate_counts_empty_ranks():
    """Two of four cases return empty → no_result_rate == 0.5."""
    cases = [_case(f"q{i}", positives=[f"p{i}"]) for i in range(4)]

    def rank_fn(query_text, query_id):
        if query_id == "q0":
            return ["p0", "x"]
        if query_id == "q1":
            return ["p1"]
        return []  # q2, q3 → empty

    result = evaluate(cases, rank_fn)
    assert result.no_result_rate == pytest.approx(0.5)


def test_no_result_rate_zero_when_all_return_results():
    cases = [_case("q1", positives=["a"])]

    def rank_fn(query_text, query_id):
        return ["a"]

    result = evaluate(cases, rank_fn)
    assert result.no_result_rate == 0.0


# ---------------------------------------------------------------------------
# 5. fallback_useful_rate increments on a useful fallback hit.
# ---------------------------------------------------------------------------


def test_fallback_useful_rate_only_when_fallback_key_matches_positive():
    """Three cases:
       * fallback_used=True AND fallback_key in positives → useful.
       * fallback_used=True AND fallback_key NOT in positives → not useful.
       * fallback_used=False → not counted as fallback.
    Expected: 1/3 ≈ 0.333.
    """
    cases = [
        # case 1: fallback used, and fallback key matches a positive.
        _case("q1", positives=["fb"]),
        # case 2: fallback used, but fallback key does NOT match a positive.
        _case("q2", positives=["something_else"]),
        # case 3: fallback not used (no fallback metadata).
        _case("q3", positives=["active_hit"]),
    ]

    def rank_fn(query_text, query_id):
        if query_id == "q1":
            return _RankFnOutcome(
                ranked=["a", "b", "c"],
                degraded=False,
                fallback_used=True,
                fallback_keys=["fb"],
            )
        if query_id == "q2":
            return _RankFnOutcome(
                ranked=["a", "b", "c"],
                degraded=False,
                fallback_used=True,
                fallback_keys=["x"],  # not in positives
            )
        if query_id == "q3":
            return ["active_hit", "a", "b"]
        return []

    result = evaluate(cases, rank_fn)
    assert result.fallback_useful_rate == pytest.approx(1 / 3)


def test_fallback_useful_rate_zero_when_legacy_list_returned():
    """Legacy ``rank_fn`` returning ``List[str]`` keeps
    ``fallback_useful_rate`` at 0.0 because no fallback metadata
    is available."""
    cases = [_case("q1", positives=["a"])]

    def rank_fn(query_text, query_id):
        return ["a", "b", "c"]

    result = evaluate(cases, rank_fn)
    assert result.fallback_useful_rate == 0.0


# ---------------------------------------------------------------------------
# 6. degraded_rate increments when the ranker reports degraded=True.
# ---------------------------------------------------------------------------


def test_degraded_rate_counts_degraded_cases():
    cases = [
        _case("q1", positives=["a"]),
        _case("q2", positives=["b"]),
        _case("q3", positives=["c"]),
        _case("q4", positives=["d"]),
    ]

    def rank_fn(query_text, query_id):
        if query_id in {"q1", "q2"}:
            return _RankFnOutcome(
                ranked=["a" if query_id == "q1" else "b"],
                degraded=True,
                fallback_used=False,
            )
        return ["c" if query_id == "q3" else "d"]

    result = evaluate(cases, rank_fn)
    assert result.degraded_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. Latency percentiles populated from wall-clock samples.
# ---------------------------------------------------------------------------


def test_latency_percentiles_reflect_per_case_samples():
    """Force the rank_fn to sleep so latency samples are non-trivial,
    then assert p50 / p95 are positive and sorted sensibly."""

    def rank_fn(query_text, query_id):
        # Each call takes ~1ms — well above noise floor on any host.
        time.sleep(0.001)
        return ["x"]

    cases = [_case(f"q{i}") for i in range(10)]
    result = evaluate(cases, rank_fn)
    assert result.p50_latency > 0.0
    assert result.p95_latency >= result.p50_latency
    # num_cases preserved so dashboards can render "no data" only
    # when the corpus itself is empty.
    assert result.num_cases == 10


def test_latency_defaults_when_no_cases():
    result = evaluate([], lambda q, qid: [])
    assert result.p50_latency == 0.0
    assert result.p95_latency == 0.0
    assert result.num_cases == 0