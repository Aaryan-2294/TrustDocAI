"""Shared domain records (plain dataclasses, no database logic).

``id`` is ``None`` until a repository persists the record. Locations are kept
as structured fields (lines / page / heading) so later parts can build
citations and traceability links without parsing strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trustdoc.models.enums import (
    ChunkType,
    ClaimOrigin,
    ClaimType,
    ClaimVerdict,
    EntityType,
    EvidenceStance,
    IssueType,
    ParameterKind,
    RunStatus,
    Severity,
    SourceType,
    VerificationTier,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------- project
@dataclass
class Project:
    name: str
    description: str = ""
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class Source:
    """One uploaded file belonging to a project."""

    project_id: int
    source_type: SourceType
    name: str                       # project-relative path / display name
    content_hash: str               # sha256 of the raw bytes
    size_bytes: int = 0
    stored_path: str | None = None  # where the raw upload is kept on disk
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. page_count, generated
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)


# ---------------------------------------------------------------- code entities
@dataclass(frozen=True)
class Parameter:
    name: str
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD
    annotation: str | None = None
    # Source text of the default expression. ``None`` = no default;
    # the string "None" = a default of ``None``.
    default: str | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "annotation": self.annotation,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Parameter":
        return cls(
            name=data["name"],
            kind=ParameterKind(data.get("kind", ParameterKind.POSITIONAL_OR_KEYWORD.value)),
            annotation=data.get("annotation"),
            default=data.get("default"),
        )


@dataclass
class CodeEntity:
    """A module, class, function or method extracted from Python source."""

    project_id: int
    source_id: int
    entity_type: EntityType
    name: str
    qualified_name: str             # e.g. "pkg.module.Class.method"
    parent_id: int | None = None    # enclosing class/module entity
    parameters: list[Parameter] = field(default_factory=list)
    return_annotation: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    is_public: bool = True
    start_line: int | None = None
    end_line: int | None = None
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)

    @property
    def has_docstring(self) -> bool:
        return bool(self.docstring and self.docstring.strip())


# ----------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    """A retrievable unit of text with full provenance."""

    project_id: int
    source_id: int
    chunk_index: int                # order within the source
    chunk_type: ChunkType
    text: str
    start_line: int | None = None
    end_line: int | None = None
    page_number: int | None = None  # PDFs (1-based)
    heading: str | None = None      # Markdown section path
    entity_id: int | None = None    # code chunk -> CodeEntity
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def location_label(self, source_name: str) -> str:
        """Human-readable citation, e.g. ``pkg/a.py:10-24`` or ``ref.pdf, p.3``."""
        label = source_name
        if self.page_number is not None:
            label += f", p.{self.page_number}"
        if self.start_line is not None:
            if self.end_line is not None and self.end_line != self.start_line:
                label += f":{self.start_line}-{self.end_line}"
            else:
                label += f":{self.start_line}"
        if self.heading:
            label += f" \u00a7 {self.heading}"
        return label


# ----------------------------------------------------------------------- claims
@dataclass
class Claim:
    """A statement made by documentation that can be checked against evidence."""

    project_id: int
    source_id: int
    claim_type: ClaimType
    text: str
    origin: ClaimOrigin = ClaimOrigin.EXISTING_DOC
    subject_name: str | None = None   # entity the claim talks about, if any
    chunk_id: int | None = None
    start_line: int | None = None
    page_number: int | None = None
    extracted: dict[str, Any] = field(default_factory=dict)  # e.g. params listed in the doc
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)


# ------------------------------------------------------------ verification runs
@dataclass
class VerificationRun:
    project_id: int
    status: RunStatus = RunStatus.PENDING
    trust_score: float | None = None            # 0-100, set when completed
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class TrustScoreComponent:
    """One line of the score breakdown shown in the final report."""

    name: str
    raw_score: float        # 0-1 measured value of this factor
    weight: float
    contribution: float     # signed points added to the final score
    detail: str = ""        # human-readable explanation


@dataclass
class TrustScoreBreakdown:
    components: list[TrustScoreComponent]
    final_score: float      # 0-100


@dataclass
class EvidenceLink:
    chunk_id: int
    stance: EvidenceStance = EvidenceStance.RELATED
    relevance_score: float | None = None
    rank: int | None = None
    claim_result_id: int | None = None
    id: int | None = None


@dataclass
class ClaimResult:
    """Outcome of verifying one claim in one run, plus its evidence."""

    run_id: int
    claim_id: int
    tier: VerificationTier
    verdict: ClaimVerdict
    confidence: float | None = None
    rationale: str = ""
    evidence: list[EvidenceLink] = field(default_factory=list)
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class Finding:
    """A detected documentation issue (hallucination, mismatch, gap, ...)."""

    run_id: int
    project_id: int
    tier: VerificationTier
    issue_type: IssueType
    severity: Severity
    message: str
    suggestion: str = ""
    source_id: int | None = None
    entity_id: int | None = None
    claim_id: int | None = None
    start_line: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)
