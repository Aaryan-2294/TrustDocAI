"""Relevance threshold / abstention decision (after reranking).

The decision is made on reranker scores only; BM25 scores and cosine
similarities are not comparable with them. The threshold is configuration
(``TRUSTDOC_RELEVANCE_THRESHOLD``). The shipped default of 0.0 is an
uncalibrated placeholder: the right value depends on the reranker model and the
sentence-transformers version (some versions return sigmoid probabilities in
[0, 1], others raw logits) and must be calibrated experimentally.

Boundary rule: a score exactly equal to the threshold is relevant (``>=``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ThresholdDecision(Generic[T]):
    threshold: float
    best_score: float | None
    abstain: bool                 # True <=> insufficient evidence
    kept: list[T]                 # items with score >= threshold, original order


def apply_relevance_threshold(
    items: Sequence[T], threshold: float, score_of: Callable[[T], float | None]
) -> ThresholdDecision[T]:
    scores = [score_of(i) for i in items]
    if any(s is None for s in scores):
        raise ValueError("every item needs a reranker score before the threshold can be applied")
    kept = [item for item, s in zip(items, scores) if s >= threshold]
    best = max(scores) if scores else None
    return ThresholdDecision(threshold=threshold, best_score=best, abstain=not kept, kept=kept)
