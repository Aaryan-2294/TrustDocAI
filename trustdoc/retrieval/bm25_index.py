"""Keyword index: Okapi BM25 (``rank_bm25.BM25Okapi``, k1=1.5, b=0.75).

Exact-term evidence for identifiers (function/class/parameter names, config
keys) that embeddings can blur. Documents and queries share ``tokenize``.
Chunks with a score of 0 (no query term present) are never returned, so a query
with no keyword overlap yields no BM25 hits.

Persistence stores the tokenised corpus as JSON (no pickle) and rebuilds the
in-memory BM25 object on load, which is fast at this scale.

Note: BM25's IDF is ~0 for a term present in half the corpus, so very tiny
corpora (a handful of chunks) can return few hits; real projects are fine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from trustdoc.exceptions import IndexCorruptError, MissingDependencyError
from trustdoc.retrieval.text import tokenize
from trustdoc.retrieval.types import RankedHit

BM25_FILE = "bm25.json"
K1, B = 1.5, 0.75


def _bm25_class():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError as exc:
        raise MissingDependencyError(
            'rank_bm25 is required for keyword retrieval. Install it with: pip install -e ".[retrieval]"'
        ) from exc


class Bm25Index:
    def __init__(self, chunk_ids: Sequence[int], tokens: list[list[str]]) -> None:
        if len(chunk_ids) != len(tokens):
            raise ValueError("chunk_ids and tokens must have the same length")
        self.chunk_ids = [int(c) for c in chunk_ids]
        self._tokens = tokens
        self._bm25 = None
        if any(tokens):
            self._bm25 = _bm25_class()(tokens, k1=K1, b=B)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(cls, chunk_ids: Sequence[int], texts: Sequence[str]) -> "Bm25Index":
        return cls(chunk_ids, [tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[RankedHit]:
        """Top-k chunks by BM25 score, best first, 1-based ranks."""
        if self._bm25 is None or k <= 0:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        scores = np.asarray(self._bm25.get_scores(terms), dtype=np.float64)
        order = [
            (float(scores[i]), self.chunk_ids[i]) for i in np.flatnonzero(scores > 0)
        ]
        order.sort(key=lambda t: (-t[0], t[1]))
        return [
            RankedHit(chunk_id=cid, rank=r, score=s)
            for r, (s, cid) in enumerate(order[:k], start=1)
        ]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"k1": K1, "b": B, "chunk_ids": self.chunk_ids, "tokens": self._tokens}
        (directory / BM25_FILE).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_count: int | None = None) -> "Bm25Index":
        try:
            payload = json.loads((directory / BM25_FILE).read_text(encoding="utf-8"))
            ids, tokens = payload["chunk_ids"], payload["tokens"]
        except (OSError, ValueError, KeyError) as exc:
            raise IndexCorruptError(f"cannot read BM25 index in {directory.name}: {exc}") from exc
        if len(ids) != len(tokens) or (expected_count is not None and len(ids) != expected_count):
            raise IndexCorruptError("BM25 index size does not match the manifest")
        return cls(ids, tokens)
