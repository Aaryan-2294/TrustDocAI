"""Retrieval metrics. Relevance is expressed as *groups* of chunk ids: each group
is one relevant item (e.g. a function), and any of its chunks counts as finding it.

* Recall@K  = (relevant groups with >=1 chunk in the top K) / (relevant groups)
* MRR       = mean over queries of 1 / rank of the first relevant chunk
              (0 when none is retrieved)
"""

from __future__ import annotations

from typing import Sequence


def recall_at_k(ranked_ids: Sequence[int], groups: Sequence[set[int]], k: int) -> float:
    if not groups:
        raise ValueError("a query needs at least one relevant group")
    top = set(ranked_ids[:k])
    return sum(1 for g in groups if g & top) / len(groups)


def reciprocal_rank(ranked_ids: Sequence[int], groups: Sequence[set[int]]) -> float:
    relevant = set().union(*groups) if groups else set()
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / position
    return 0.0


def first_relevant_rank(ranked_ids: Sequence[int], groups: Sequence[set[int]]) -> int | None:
    relevant = set().union(*groups) if groups else set()
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return position
    return None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
