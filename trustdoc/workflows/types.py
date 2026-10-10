"""Result types for the Ask workflow (pure data; no heavy imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trustdoc.retrieval.types import Evidence


class AskStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"


class GroundingLevel(str, Enum):
    """How the *answer text* relates to the supplied evidence IDs.

    This is a purely deterministic check of citation markers. It does NOT mean
    the claims were semantically verified; that is Part 5's job.
    """

    CITED = "cited"                                          # >=1 valid citation, none invalid
    CITED_WITH_INVALID_REFERENCES = "cited_with_invalid_references"
    UNCITED = "uncited"                                      # no valid citation at all


@dataclass(frozen=True)
class AskCitation:
    """A valid citation: an evidence ID the answer used, mapped to its real source."""

    id: str                      # "E1"
    label: str                   # exactly as retrieval produced it
    source_name: str
    source_type: str
    source_id: int
    chunk_id: int
    mentions: int = 1            # how many times the answer cited this ID
    start_line: int | None = None
    end_line: int | None = None
    page_number: int | None = None
    heading: str | None = None
    qualified_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AskEvidence:
    """One retrieved evidence item exactly as it was shown to the model."""

    id: str
    evidence: Evidence
    truncated: bool = False      # text was cut to fit the character budget
    sanitized: bool = False      # delimiter/ID look-alikes in the text were neutralised

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        e = self.evidence
        data: dict[str, Any] = {
            "id": self.id, "label": e.citation.label, "source_name": e.source_name,
            "chunk_id": e.chunk_id, "rerank_score": e.rerank_score,
            "truncated": self.truncated, "sanitized": self.sanitized,
        }
        if include_text:
            data["text"] = e.text
        return data


@dataclass
class RetrievalSummary:
    mode: str
    abstained: bool
    n_ranked: int = 0
    n_evidence: int = 0
    best_score: float | None = None
    threshold: float | None = None
    threshold_applied: bool = False
    top_candidates: list[dict[str, Any]] = field(default_factory=list)  # labels + scores, no text


@dataclass
class AskResult:
    project_id: int
    question: str
    status: AskStatus
    answer: str                                   # citation-sanitised text, or the abstention message
    abstained: bool = False
    abstention_reason: str | None = None          # no_evidence_retrieved | below_relevance_threshold | model_declared_insufficient
    raw_answer: str | None = None                 # exactly what the model produced
    grounding: GroundingLevel | None = None
    citations: list[AskCitation] = field(default_factory=list)       # valid IDs the answer cited
    invalid_citation_ids: list[str] = field(default_factory=list)
    evidence: list[AskEvidence] = field(default_factory=list)        # everything shown to the model
    dropped_evidence: list[str] = field(default_factory=list)        # labels cut by the prompt budget
    retrieval: RetrievalSummary | None = None
    warnings: list[str] = field(default_factory=list)
    model_name: str | None = None
    backend: str | None = None
    is_stub_model: bool = False
    finish_reason: str | None = None
    error: str | None = None
    error_kind: str | None = None
    prompt_stats: dict[str, Any] = field(default_factory=dict)       # sizes only, never prompt text
    timings_ms: dict[str, float] = field(default_factory=dict)
    claims_verified: bool = False                 # always False in Part 4: no claim-level verification yet

    def to_dict(self, include_evidence_text: bool = False) -> dict[str, Any]:
        return {
            "project_id": self.project_id, "question": self.question, "status": self.status.value,
            "answer": self.answer, "abstained": self.abstained,
            "abstention_reason": self.abstention_reason, "raw_answer": self.raw_answer,
            "grounding": self.grounding.value if self.grounding else None,
            "citations": [c.to_dict() for c in self.citations],
            "invalid_citation_ids": self.invalid_citation_ids,
            "evidence": [e.to_dict(include_evidence_text) for e in self.evidence],
            "dropped_evidence": self.dropped_evidence,
            "retrieval": None if self.retrieval is None else dict(self.retrieval.__dict__),
            "warnings": self.warnings, "model_name": self.model_name, "backend": self.backend,
            "is_stub_model": self.is_stub_model, "finish_reason": self.finish_reason,
            "error": self.error, "error_kind": self.error_kind, "prompt_stats": self.prompt_stats,
            "timings_ms": self.timings_ms, "claims_verified": self.claims_verified,
        }
