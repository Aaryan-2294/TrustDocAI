import pytest

from trustdoc.config import VerificationConfig
from trustdoc.verification.scoring import ScoreInputs, compute_trust_score

CFG = VerificationConfig()                         # weights 0.5/0.25/0.25, penalties 10/5, cap 40


def score(**kw):
    return compute_trust_score(ScoreInputs(**kw), CFG)


def comp(report, name):
    return next(c for c in report.components if c.name == name)


def test_known_fixture_reproduces_the_documented_formula():
    r = score(supported=8, contradicted=1, insufficient=1, grounded=9, public_entities=10, documented_entities=7, high_contradictions=1)
    assert comp(r, "accuracy").raw_score == 0.8 and comp(r, "coverage").raw_score == 0.7 and comp(r, "grounding").raw_score == 0.9
    assert (comp(r, "accuracy").contribution, comp(r, "coverage").contribution, comp(r, "grounding").contribution) == (40.0, 17.5, 22.5)
    assert r.base == 80.0 and comp(r, "severity_penalty").contribution == -5.0 and r.final == 75.0
    assert r.inputs["checked"] == 10 and r.inputs["insufficient_evidence"] == 1


def test_perfect_documentation_scores_100():
    assert score(supported=5, grounded=5, public_entities=4, documented_entities=4).final == 100.0


def test_insufficient_evidence_is_never_counted_as_correct():
    all_supported = score(supported=4, grounded=4, public_entities=1, documented_entities=1)
    some_unverified = score(supported=2, insufficient=2, grounded=2, public_entities=1, documented_entities=1)
    assert comp(some_unverified, "accuracy").raw_score == 0.5 and some_unverified.final < all_supported.final
    # unverified claims also have no evidence, which lowers grounding too
    assert comp(some_unverified, "grounding").raw_score == 0.5


def test_contradictions_lower_accuracy_and_severity_penalties_apply_only_to_contradiction_findings():
    clean = score(supported=4, grounded=4, public_entities=2, documented_entities=2)
    bad = score(supported=3, contradicted=1, grounded=4, public_entities=2, documented_entities=2, critical_contradictions=1)
    assert bad.final < clean.final
    assert comp(bad, "severity_penalty").contribution == -10.0 and bad.penalties == [{"severity": "critical", "count": 1, "points_each": 10.0}]


def test_penalties_are_capped():
    r = score(supported=10, grounded=10, public_entities=1, documented_entities=1, critical_contradictions=9, high_contradictions=9)
    assert comp(r, "severity_penalty").contribution == -40.0 and "capped at 40" in comp(r, "severity_penalty").detail


def test_score_is_bounded_between_0_and_100():
    low = score(contradicted=3, insufficient=2, grounded=0, public_entities=3, documented_entities=0, critical_contradictions=20)
    assert low.final == 0.0
    for kw in ({}, {"supported": 1}, {"public_entities": 3, "documented_entities": 3}):
        r = score(**kw)
        assert r.final is None or 0.0 <= r.final <= 100.0


def test_no_claims_and_no_entities_means_not_measurable_not_a_made_up_number():
    r = score()
    assert r.final is None and r.base is None and "no trust score is given" in " ".join(r.notes).lower()
    assert all(c.weight == 0 and "unavailable" in c.detail for c in r.components if c.name != "severity_penalty")


def test_coverage_alone_never_produces_a_trust_score():
    """Regression: a documented-but-unchecked project used to score 100/100."""
    r = score(public_entities=18, documented_entities=18, not_checkable=5)
    assert r.final is None and r.base is None
    cov = comp(r, "coverage")
    assert cov.raw_score == 1.0 and cov.weight == 0.0 and cov.contribution == 0.0 and "informational, not scored" in cov.detail
    assert any("No claim was checked" in n for n in r.notes) and any("5 claim(s) were not checkable" in n for n in r.notes)


def test_unavailable_components_are_dropped_and_weights_renormalised():
    only_claims = score(supported=1, contradicted=1, grounded=2)
    assert comp(only_claims, "accuracy").weight == pytest.approx(0.6667, abs=1e-3) and comp(only_claims, "coverage").weight == 0
    assert only_claims.final == pytest.approx(100 * (0.5 * 0.5 + 0.25 * 1.0) / 0.75, abs=0.01)
    assert any("coverage is unavailable" in n for n in only_claims.notes)


def test_missing_coverage_is_never_fabricated():
    r = score(supported=2, grounded=2)                     # no public entities -> coverage unmeasurable
    assert comp(r, "coverage").raw_score == 0.0 and comp(r, "coverage").weight == 0.0 and r.inputs["public_entities"] == 0


def test_not_checkable_claims_are_excluded_but_reported():
    r = score(supported=2, grounded=2, not_checkable=5, public_entities=1, documented_entities=1)
    assert r.inputs["checked"] == 2 and comp(r, "accuracy").raw_score == 1.0
    assert any("5 claim(s) were not checkable" in n for n in r.notes)


def test_custom_weights_and_zero_weight_components():
    cfg = VerificationConfig(weight_accuracy=1.0, weight_coverage=0.0, weight_grounding=0.0)
    r = compute_trust_score(ScoreInputs(supported=1, contradicted=1, grounded=2, public_entities=1, documented_entities=1), cfg)
    assert r.final == 50.0 and comp(r, "coverage").weight == 0
    cfg2 = VerificationConfig(weight_accuracy=1.0, weight_coverage=1.0, weight_grounding=2.0, penalty_high=1.0)
    r2 = compute_trust_score(ScoreInputs(supported=1, grounded=0, public_entities=2, documented_entities=1, high_contradictions=3), cfg2)
    assert r2.final == pytest.approx(100 * (0.25 * 1 + 0.25 * 0.5 + 0.5 * 0) - 3.0, abs=0.01)


def test_scoring_is_deterministic_and_components_expose_denominators():
    kw = dict(supported=3, contradicted=1, insufficient=2, grounded=3, public_entities=6, documented_entities=5)
    a, b = score(**kw), score(**kw)
    assert a.final == b.final and [c.__dict__ for c in a.components] == [c.__dict__ for c in b.components]
    assert "3 supported / 6 checked" in comp(a, "accuracy").detail and "5 documented / 6 public" in comp(a, "coverage").detail
