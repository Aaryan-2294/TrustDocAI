"""Transparent, deterministic Trust Score (0-100).

Components (all derived from counts that are stored with the run):

* accuracy  = SUPPORTED / (SUPPORTED + CONTRADICTED + INSUFFICIENT_EVIDENCE)
              over the claims the engine actually checked. INSUFFICIENT_EVIDENCE stays in the
              denominator, so an unverified claim is never counted as correct. NOT_CHECKABLE claims
              (skipped, external, example code, semantic checks disabled) are excluded and reported.
* coverage  = public entities that have a docstring or are mentioned in the verified documents,
              divided by all public entities.
* grounding = checked claims with a decisive verdict (SUPPORTED or CONTRADICTED) AND at least one
              evidence reference, divided by checked claims.

    base    = 100 x (w_a*accuracy + w_c*coverage + w_g*grounding) / (sum of the weights of the available components)
    penalty = min(cap, critical_findings * P_critical + high_findings * P_high)       (contradictions only)
    score   = clamp(base - penalty, 0, 100)

Defaults: weights 0.50 / 0.25 / 0.25, P_critical = 10, P_high = 5, cap = 40.

Measurability rules (no invented values):
* A score is given ONLY if at least one claim was checked. Coverage alone says nothing about whether the
  documentation is correct, so with zero checked claims the score is None ("not measurable") and coverage
  is reported as an informational, unscored figure.
* Coverage with no public entities is UNAVAILABLE and dropped; the remaining weights are renormalised.
"""

from __future__ import annotations

from dataclasses import dataclass

from trustdoc.config import VerificationConfig
from trustdoc.models import TrustScoreComponent
from trustdoc.verification.types import ScoreReport


@dataclass
class ScoreInputs:
    supported: int = 0
    contradicted: int = 0
    insufficient: int = 0
    not_checkable: int = 0
    grounded: int = 0                 # decisive claims with >= 1 evidence reference
    public_entities: int = 0
    documented_entities: int = 0
    critical_contradictions: int = 0  # CRITICAL findings that are contradictions
    high_contradictions: int = 0      # HIGH findings that are contradictions

    @property
    def checked(self) -> int:
        return self.supported + self.contradicted + self.insufficient


def compute_trust_score(inputs: ScoreInputs, cfg: VerificationConfig) -> ScoreReport:
    weights = {"accuracy": cfg.weight_accuracy, "coverage": cfg.weight_coverage, "grounding": cfg.weight_grounding}
    raw: dict[str, float | None] = {
        "accuracy": inputs.supported / inputs.checked if inputs.checked else None,
        "coverage": inputs.documented_entities / inputs.public_entities if inputs.public_entities else None,
        "grounding": inputs.grounded / inputs.checked if inputs.checked else None,
    }
    why_unavailable = {
        "accuracy": "no claim was checked, so accuracy is not measurable",
        "coverage": "the project has no public classes/functions/methods, or coverage was not measured for this run",
        "grounding": "no claim was checked, so grounding is not measurable",
    }
    detail = {
        "accuracy": f"{inputs.supported} supported / {inputs.checked} checked ({inputs.contradicted} contradicted, {inputs.insufficient} insufficient evidence)",
        "coverage": f"{inputs.documented_entities} documented / {inputs.public_entities} public entities",
        "grounding": f"{inputs.grounded} decisive claims with evidence / {inputs.checked} checked",
    }
    available = {k: v for k, v in raw.items() if v is not None and weights[k] > 0}
    total_weight = sum(weights[k] for k in available)
    notes: list[str] = []
    components: list[TrustScoreComponent] = []
    base: float | None = None
    if raw["accuracy"] is None:                      # nothing was checked: do not let coverage masquerade as trust
        components.append(TrustScoreComponent("accuracy", 0.0, 0.0, 0.0, f"unavailable: {why_unavailable['accuracy']}"))
        if raw["coverage"] is not None:
            components.append(TrustScoreComponent("coverage", round(raw["coverage"], 4), 0.0, 0.0, f"informational, not scored: {detail['coverage']}"))
        else:
            components.append(TrustScoreComponent("coverage", 0.0, 0.0, 0.0, f"unavailable: {why_unavailable['coverage']}"))
        components.append(TrustScoreComponent("grounding", 0.0, 0.0, 0.0, f"unavailable: {why_unavailable['grounding']}"))
        notes.append("No claim was checked, so no Trust Score is given: coverage alone says nothing about whether the documentation is correct.")
    elif available and total_weight > 0:
        base = 0.0
        for name in ("accuracy", "coverage", "grounding"):
            if name in available:
                eff = weights[name] / total_weight
                contribution = 100.0 * eff * raw[name]  # type: ignore[operator]
                base += contribution
                components.append(TrustScoreComponent(name, round(raw[name], 4), round(eff, 4), round(contribution, 4),  # type: ignore[arg-type]
                                                      f"{detail[name]} (configured weight {weights[name]:g})"))
            else:
                reason = why_unavailable[name] if raw[name] is None else "its configured weight is 0"
                components.append(TrustScoreComponent(name, 0.0, 0.0, 0.0, f"unavailable: {reason}"))
                notes.append(f"{name} is unavailable ({reason}); the remaining weights were renormalised.")
    else:                                            # every configured weight is zero
        for name in ("accuracy", "coverage", "grounding"):
            components.append(TrustScoreComponent(name, 0.0, 0.0, 0.0, f"unavailable: its configured weight is 0"))
        notes.append("All component weights are 0, so no score is given.")

    penalties = []
    penalty = min(cfg.penalty_cap, inputs.critical_contradictions * cfg.penalty_critical + inputs.high_contradictions * cfg.penalty_high)
    if inputs.critical_contradictions:
        penalties.append({"severity": "critical", "count": inputs.critical_contradictions, "points_each": cfg.penalty_critical})
    if inputs.high_contradictions:
        penalties.append({"severity": "high", "count": inputs.high_contradictions, "points_each": cfg.penalty_high})
    uncapped = inputs.critical_contradictions * cfg.penalty_critical + inputs.high_contradictions * cfg.penalty_high
    components.append(TrustScoreComponent(
        "severity_penalty", round(penalty / 100.0, 4), 1.0, round(0.0 - penalty, 4),
        (f"{inputs.critical_contradictions} critical x {cfg.penalty_critical:g} + {inputs.high_contradictions} high x {cfg.penalty_high:g}"
         + (f" = {uncapped:g}, capped at {cfg.penalty_cap:g}" if uncapped > cfg.penalty_cap else ""))))

    final = None if base is None else round(max(0.0, min(100.0, base - penalty)), 2)
    if inputs.not_checkable:
        notes.append(f"{inputs.not_checkable} claim(s) were not checkable in this run and are excluded from the score.")
    return ScoreReport(final=final, base=None if base is None else round(base, 4), components=components, penalties=penalties,
                       inputs={"supported": inputs.supported, "contradicted": inputs.contradicted, "insufficient_evidence": inputs.insufficient,
                               "not_checkable": inputs.not_checkable, "checked": inputs.checked, "grounded": inputs.grounded,
                               "public_entities": inputs.public_entities, "documented_entities": inputs.documented_entities},
                       notes=notes)
