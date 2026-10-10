"""Domain-level retrieval structures (pure data; no numpy/faiss imports).

Everything a later module needs from retrieval is in ``Evidence`` and
``SearchResult``. Citations are built only from the logical, project-relative
source name and chunk metadata; the physical ``stored_path`` is never carried.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RetrievalMode(str, Enum):
    BM25 = "bm25"                      # keyword only
    FAISS = "faiss"                    # semantic only
    HYBRID = "hybrid"                  # FAISS + BM25 fused with RRF
    HYBRID_RERANK = "hybrid_rerank"    # hybrid candidates re-scored by a Cross-Encoder


@dataclass(frozen=True)
class RankedHit:
    """One entry of a single retriever's ranked list (rank is 1-based)."""

    chunk_id: int
    rank: int
    score: float          # native score of that retriever (BM25 score / cosine similarity)


@dataclass
class RankedList:
    name: str             # e.g. "bm25", "faiss" - unique within one fusion call
    hits: list[RankedHit] = field(default_factory=list)


@dataclass(frozen=True)
class Contribution:
    """How one ranked list contributed to a fused chunk."""

    list_name: str
    rank: int
    score: float          # the retriever's native score (never mixed across lists)
    rrf_contribution: float


@dataclass(frozen=True)
class FusedHit:
    chunk_id: int
    rrf_score: float
    rank: int             # 1-based position in the fused list
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class Citation:
    """Where a piece of evidence comes from. Safe to show to users."""

    source_id: int
    source_name: str                 # logical, project-relative (e.g. "pkg/a.py")
    source_type: str
    label: str                       # e.g. "pkg/a.py:12-30", "README.md:4-9 § A > B", "ref.pdf, p.2"
    chunk_id: int
    start_line: int | None = None
    end_line: int | None = None
    page_number: int | None = None
    heading: str | None = None
    qualified_name: str | None = None


@dataclass
class Evidence:
    """A retrieved chunk with its provenance and every score that produced its rank."""

    chunk_id: int
    source_id: int
    project_id: int
    text: str
    chunk_type: str
    source_name: str
    source_type: str
    citation: Citation
    method: RetrievalMode
    entity_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # per-retriever evidence (None when that retriever did not return the chunk)
    bm25_rank: int | None = None
    bm25_score: float | None = None
    faiss_rank: int | None = None
    faiss_score: float | None = None        # cosine similarity in [-1, 1]
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    final_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["method"] = self.method.value
        return data


@dataclass
class SearchResult:
    query: str
    project_id: int
    mode: RetrievalMode
    ranked: list[Evidence]            # top_k after ranking, BEFORE the relevance threshold
    evidence: list[Evidence]          # what survives the threshold (== ranked when not applied)
    abstained: bool                   # True when there is no (sufficiently relevant) evidence
    threshold: float | None = None
    threshold_applied: bool = False
    best_score: float | None = None   # best reranker score when a reranker ran
    candidate_count: int = 0          # chunks that entered the final ranking stage
    timings_ms: dict[str, float] = field(default_factory=dict)
    index_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query, "project_id": self.project_id, "mode": self.mode.value,
            "abstained": self.abstained, "threshold": self.threshold,
            "threshold_applied": self.threshold_applied, "best_score": self.best_score,
            "candidate_count": self.candidate_count, "timings_ms": self.timings_ms,
            "ranked": [e.to_dict() for e in self.ranked],
            "evidence": [e.to_dict() for e in self.evidence],
        }
