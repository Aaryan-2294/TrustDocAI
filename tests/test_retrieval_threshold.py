import pytest

from trustdoc.retrieval import apply_relevance_threshold

score = lambda t: t[1]  # noqa: E731


def test_above_threshold_keeps_relevant_items_in_order():
    items = [("a", 0.9), ("b", 0.6), ("c", 0.2)]
    d = apply_relevance_threshold(items, 0.5, score)
    assert [i[0] for i in d.kept] == ["a", "b"] and not d.abstain and d.best_score == 0.9


def test_all_below_threshold_abstains():
    d = apply_relevance_threshold([("a", 0.3), ("b", 0.1)], 0.5, score)
    assert d.abstain and d.kept == [] and d.best_score == 0.3 and d.threshold == 0.5


def test_boundary_is_inclusive():
    d = apply_relevance_threshold([("a", 0.5), ("b", 0.4999)], 0.5, score)
    assert [i[0] for i in d.kept] == ["a"] and not d.abstain
    assert apply_relevance_threshold([("a", 0.4999)], 0.5, score).abstain


def test_empty_candidates_abstain():
    d = apply_relevance_threshold([], 0.1, score)
    assert d.abstain and d.best_score is None and d.kept == []


def test_negative_thresholds_work_for_logit_scores():
    d = apply_relevance_threshold([("a", -1.0), ("b", -6.0)], -2.0, score)
    assert [i[0] for i in d.kept] == ["a"]


def test_missing_scores_are_rejected():
    with pytest.raises(ValueError):
        apply_relevance_threshold([("a", None)], 0.0, score)
