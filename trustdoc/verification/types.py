"""Shared structures for documentation verification (Part 5).

Existing records are reused as they are: ``Claim`` for extracted claims, ``Finding`` for findings,
``VerificationRun`` / ``TrustScoreBreakdown`` for runs and scores. Extraction and verification are
pure (no database), so they produce *drafts* carrying placeholder ids (0) that the engine stamps when
it persists, the same pattern Part 2 uses for chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from trustdoc.models import (
    Claim,
    ClaimType,
    ClaimVerdict,
    CodeEntity,
    EvidenceStance,
    Finding,
    Severity,
    TrustScoreComponent,
    VerificationRun,
    VerificationTier,
)

UNASSIGNED = 0
STATIC_CATEGORIES = frozenset({ClaimType.ENTITY_REFERENCE, ClaimType.SIGNATURE, ClaimType.PARAMETER, ClaimType.RETURN})
SEMANTIC_CATEGORIES = frozenset({ClaimType.BEHAVIORAL, ClaimType.CONFIG_DEPENDENCY, ClaimType.OTHER})

# claim.extracted keys written by the extractors / engine
UID, METHOD, CONFIDENCE = "claim_uid", "method", "confidence"


def normalize_claim_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def make_claim_uid(source_name: str, category: ClaimType, text: str, occurrence: int = 0) -> str:
    """Content-derived, reproducible claim identity (independent of database ids and line numbers).

    Re-extracting the same document yields the same ids; an identical sentence appearing twice gets
    distinct ids through ``occurrence``.
    """
    raw = f"{source_name}\x1f{category.value}\x1f{normalize_claim_text(text)}\x1f{occurrence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def claim_uid(claim: Claim) -> str | None:
    return claim.extracted.get(UID)


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    mentions: set[str] = field(default_factory=set)      # identifier-like code spans (for doc coverage)
    warnings: list[str] = field(default_factory=list)

    def extend(self, other: "ExtractionResult") -> None:
        self.claims += other.claims
        self.mentions |= other.mentions
        self.warnings += other.warnings


@dataclass(frozen=True)
class EvidenceDraft:
    chunk_id: int
    stance: EvidenceStance
    relevance_score: float | None = None
    rank: int | None = None


@dataclass
class ClaimOutcome:
    """The result of checking one claim, before persistence."""

    claim: Claim
    tier: VerificationTier
    verdict: ClaimVerdict
    rationale: str
    confidence: float | None = None          # deterministic checks: 1.0; LLM checks: None (never trusted)
    findings: list[Finding] = field(default_factory=list)
    evidence: list[EvidenceDraft] = field(default_factory=list)
    entity: CodeEntity | None = None
    model_assisted: bool = False
    skip_reason: str | None = None           # why a NOT_CHECKABLE claim was not checked


# --------------------------------------------------------------------------- report views
@dataclass
class EvidenceRef:
    chunk_id: int
    label: str                       # logical citation label (never a filesystem path)
    stance: str
    relevance_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ClaimReport:
    claim_id: int
    claim_uid: str | None
    text: str
    category: ClaimType
    status: ClaimVerdict
    tier: VerificationTier
    explanation: str
    origin: str
    doc_label: str                   # e.g. "README.md:21" or "pkg/a.py:10 (docstring)"
    subject: str | None = None
    confidence: float | None = None
    model_assisted: bool = False
    evidence: list[EvidenceRef] = field(default_factory=list)
    finding_uids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.update(category=self.category.value, status=self.status.value, tier=self.tier.value,
                 evidence=[e.to_dict() for e in self.evidence])
        return d


@dataclass
class FindingReport:
    uid: str
    claim_uid: str | None
    category: str                    # claim category, or "coverage"
    status: str                      # claim status, or "missing_documentation"
    severity: Severity
    issue_type: str
    message: str
    suggestion: str
    tier: VerificationTier
    doc_location: dict[str, Any] | None = None      # where the documentation says it
    code_location: dict[str, Any] | None = None     # where the source actually is
    evidence: list[dict[str, Any]] = field(default_factory=list)
    model_assisted: bool = False
    claim_text: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.update(severity=self.severity.value, tier=self.tier.value)
        return d


@dataclass
class ScoreReport:
    final: float | None                              # None when nothing measurable exists
    base: float | None
    components: list[TrustScoreComponent]
    penalties: list[dict[str, Any]]
    inputs: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final": self.final, "base": self.base, "penalties": self.penalties, "inputs": self.inputs,
            "notes": self.notes, "components": [dict(c.__dict__) for c in self.components],
        }


@dataclass
class VerificationReport:
    run: VerificationRun
    project_id: int
    documents: list[str]
    counts: dict[str, int]
    claims: list[ClaimReport]
    findings: list[FindingReport]
    score: ScoreReport | None
    skipped: list[str] = field(default_factory=list)       # checks that did not run, and why
    warnings: list[str] = field(default_factory=list)
    semantic: dict[str, Any] = field(default_factory=dict)

    def findings_by_severity(self) -> dict[Severity, list[FindingReport]]:
        out: dict[Severity, list[FindingReport]] = {s: [] for s in sorted(Severity, key=lambda s: -s.rank)}
        for f in self.findings:
            out[f.severity].append(f)
        return out

    def max_severity(self, *, defects_only: bool = True) -> Severity | None:
        """Highest severity among findings. INFO findings (unverified) are never defects."""
        sev = [f.severity for f in self.findings if not (defects_only and f.severity == Severity.INFO)]
        return max(sev, key=lambda s: s.rank) if sev else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id, "project_id": self.project_id, "status": self.run.status.value,
            "documents": self.documents, "counts": self.counts, "trust_score": self.score.to_dict() if self.score else None,
            "claims": [c.to_dict() for c in self.claims], "findings": [f.to_dict() for f in self.findings],
            "skipped": self.skipped, "warnings": self.warnings, "semantic": self.semantic,
        }
