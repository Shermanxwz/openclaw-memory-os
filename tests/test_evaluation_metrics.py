"""Tests for the v0.3.0 offline evaluation layer (S8)."""

from __future__ import annotations


import pytest

from openclaw_memory_os.evaluation import (
    EvalResult,
    _EvaluationCase,
    _ndcg_at_k,
    _mrr,
    _recall_at_k,
    _useful_at_k,
    _explicit_negative_at_k,
    evaluate,
    evaluate_candidate,
    time_split,
)


def _case(positives, negatives=None, query_text="test"):
    return _EvaluationCase("q1", query_text, set(positives), set(negatives or []))


# ---------------------------------------------------------------------------
# Metric correctness (simple cases)
# ---------------------------------------------------------------------------


def test_recall_at_k_all_relevant():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "b", "c"}
    assert _recall_at_k(ranked, relevant, 5) == pytest.approx(1.0)


def test_recall_at_k_partial():
    ranked = ["a", "b", "c"]
    relevant = {"a", "x"}
    assert _recall_at_k(ranked, relevant, 3) == pytest.approx(1 / 2)


def test_recall_at_k_no_relevant():
    ranked = ["a", "b"]
    assert _recall_at_k(ranked, set(), 3) == 0.0


def test_recall_at_k_divides_by_total_relevant():
    ranked = ["a", "b"] * 10
    relevant = {"a", "x", "y"}
    assert _recall_at_k(ranked, relevant, 1) == pytest.approx(1 / 3)


def test_mrr_first_is_relevant():
    assert _mrr(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_is_relevant():
    assert _mrr(["a", "b"], {"b"}) == pytest.approx(1 / 2)


def test_mrr_no_relevant():
    assert _mrr(["a", "b"], {"c"}) == 0.0


def test_mrr_no_results():
    assert _mrr([], {"a"}) == 0.0


def test_ndcg_no_relevant():
    assert _ndcg_at_k(["a"], set(), 10, set()) == 0.0


def test_ndcg_single_perfect():
    ranked = ["a", "b"]
    relevant = {"a"}
    val = _ndcg_at_k(ranked, relevant, 10, set())
    assert val == pytest.approx(1.0)


def test_ndcg_negative_not_harmful():
    ranked = ["a", "b", "c"]
    relevant = {"a"}
    neg = {"c"}
    val = _ndcg_at_k(ranked, relevant, 3, neg)
    assert 0 < val <= 1.0


def test_useful_at_k_correct_fraction():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert _useful_at_k(ranked, relevant, 2) == pytest.approx(0.5)


def test_useful_at_k_handles_shorter_list():
    assert _useful_at_k(["a"], {"a"}, 5) == pytest.approx(1.0)


def test_explicit_negative_at_k():
    ranked = ["a", "b", "c"]
    neg = {"b"}
    assert _explicit_negative_at_k(ranked, neg, 3) == pytest.approx(1 / 3)


def test_explicit_negative_at_k_excludes_unjudged():
    ranked = ["a", "b", "c"]
    assert _explicit_negative_at_k(ranked, set(), 3) == 0.0


# ---------------------------------------------------------------------------
# Time split
# ---------------------------------------------------------------------------


def test_time_split_honours_ratios():
    cases = [_case([str(i)]) for i in range(100)]
    split = time_split(cases, train_pct=0.6, validation_pct=0.2)
    assert len(split.train) == 60
    assert len(split.validation) == 20
    assert len(split.test) == 20


def test_time_split_empty():
    split = time_split([], train_pct=0.6, validation_pct=0.2)
    assert len(split.train) == 0
    assert len(split.validation) == 0
    assert len(split.test) == 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_uses_rank_fn():
    def rank_fn(query, qid):
        return ["a", "b", "c", "d"]

    case = _case(positives=["a", "b", "c"], negatives=["d"])
    result = evaluate([case], rank_fn)
    assert result.recall_at_1 == pytest.approx(1 / 3)  # a is one of 3 relevant
    assert result.recall_at_5 == pytest.approx(1.0)  # all relevant in top 5
    assert result.num_cases == 1


def test_evaluate_empty_cases():
    def rank_fn(query, qid):
        return []

    result = evaluate([], rank_fn)
    assert result.num_cases == 0


# ---------------------------------------------------------------------------
# evaluate_candidate
# ---------------------------------------------------------------------------


def test_evaluate_passing_candidate():
    baseline = EvalResult(
        recall_at_5=0.5,
        mrr_at_10=0.6,
        ndcg_at_10=0.5,
        useful_at_1=0.4,
        explicit_negative_at_5=0.1,
        no_result_rate=0.0,
        num_cases=10,
    )
    candidate = EvalResult(
        recall_at_5=0.55,
        mrr_at_10=0.61,
        ndcg_at_10=0.504,  # below 1% improvement (0.505) → fail
        useful_at_1=0.42,
        explicit_negative_at_5=0.09,
        no_result_rate=0.0,
        num_cases=10,
    )
    passed, reasons = evaluate_candidate(baseline, candidate, strict_validation=True)
    # nDCG improvement < 1% → fail
    assert passed is False
    assert any("nDCG" in r for r in reasons)


def test_evaluate_passing_candidate_non_strict():
    baseline = EvalResult(
        recall_at_5=0.5,
        mrr_at_10=0.6,
        ndcg_at_10=0.5,
        useful_at_1=0.4,
        explicit_negative_at_5=0.1,
        no_result_rate=0.0,
        num_cases=10,
    )
    candidate = EvalResult(
        recall_at_5=0.55,
        mrr_at_10=0.61,
        ndcg_at_10=0.51,
        useful_at_1=0.42,
        explicit_negative_at_5=0.09,
        no_result_rate=0.0,
        num_cases=10,
    )
    passed, reasons = evaluate_candidate(baseline, candidate, strict_validation=False)
    assert passed is True
