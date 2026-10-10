"""Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum over ranked lists L containing d of  1 / (k + rank_L(d))

Only *ranks* are used, so scores from different retrievers (BM25 scores,
cosine similarities) are never normalised or mixed. A chunk found by several
lists accumulates a contribution from each.
"""

from __future__ import annotations

import math
from typing import Sequence

from trustdoc.retrieval.types import Contribution, FusedHit, RankedList


def reciprocal_rank_fusion(ranked_lists: Sequence[RankedList], k: int = 60) -> list[FusedHit]:
    """Fuse ranked lists into one ranking.

    * Ranks are the 1-based ``RankedHit.rank`` values supplied by each list.
    * If a chunk repeats within one list, only its best rank counts.
    * Ordering is deterministic: higher RRF score first; ties go to the chunk
      with the better best-rank in any list, then to the lower chunk id.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    names = [rl.name for rl in ranked_lists]
    if len(set(names)) != len(names):
        raise ValueError(f"ranked list names must be unique, got {names}")

    contributions: dict[int, list[Contribution]] = {}
    for rl in ranked_lists:
        seen: set[int] = set()
        for hit in sorted(rl.hits, key=lambda h: h.rank):
            if hit.rank < 1:
                raise ValueError(f"ranks are 1-based, got {hit.rank} in list {rl.name!r}")
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            contributions.setdefault(hit.chunk_id, []).append(
                Contribution(rl.name, hit.rank, hit.score, 1.0 / (k + hit.rank))
            )

    scored = [
        (math.fsum(c.rrf_contribution for c in contribs), min(c.rank for c in contribs), cid, contribs)
        for cid, contribs in contributions.items()
    ]
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [
        FusedHit(chunk_id=cid, rrf_score=score, rank=position, contributions=tuple(contribs))
        for position, (score, _best, cid, contribs) in enumerate(scored, start=1)
    ]
