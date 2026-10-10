"""Semantic index: FAISS ``IndexFlatIP`` over L2-normalised embeddings.

Metric: **inner product on unit vectors == cosine similarity**. Scores are
similarities in [-1, 1] where HIGHER IS BETTER (these are not distances).
``IndexFlatIP`` is exact (brute force), needs no training, and is plenty fast
for the thousands of chunks this project handles on a CPU.

FAISS only knows vector positions ``0..n-1``; the position -> chunk id mapping
is kept alongside it (``chunk_ids``) and persisted with the index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from trustdoc.exceptions import IndexCorruptError, MissingDependencyError, RetrievalError
from trustdoc.retrieval.embedding import l2_normalize
from trustdoc.retrieval.types import RankedHit

METRIC = "inner_product_on_l2_normalised_vectors (cosine similarity; higher is better)"
INDEX_FILE = "faiss.index"
IDS_FILE = "faiss_ids.npy"


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise MissingDependencyError(
            'faiss is required for semantic retrieval. Install it with: pip install -e ".[retrieval]"'
        ) from exc


class FaissIndex:
    def __init__(self, chunk_ids: np.ndarray, index, dimension: int) -> None:
        self.chunk_ids = np.asarray(chunk_ids, dtype=np.int64)
        self._index = index          # None when the index is empty
        self.dimension = dimension

    def __len__(self) -> int:
        return int(len(self.chunk_ids))

    @classmethod
    def build(cls, chunk_ids: Sequence[int], vectors: np.ndarray) -> "FaissIndex":
        ids = np.asarray(list(chunk_ids), dtype=np.int64)
        if len(ids) == 0:
            return cls(ids, None, 0)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise RetrievalError(f"{len(ids)} chunk ids but vectors of shape {vectors.shape}")
        if len(set(ids.tolist())) != len(ids):
            raise RetrievalError("chunk ids must be unique within an index")
        faiss = _faiss()
        data = l2_normalize(vectors)
        index = faiss.IndexFlatIP(data.shape[1])
        index.add(data)
        return cls(ids, index, int(data.shape[1]))

    def search(self, query_vector: np.ndarray, k: int) -> list[RankedHit]:
        """Top-k chunks by cosine similarity, best first, 1-based ranks."""
        if self._index is None or k <= 0:
            return []
        q = l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.dimension:
            raise RetrievalError(
                f"query embedding has dimension {q.shape[1]} but the index has {self.dimension}; "
                "the embedding model changed - rebuild the index"
            )
        scores, positions = self._index.search(q, min(k, len(self)))
        pairs = [
            (float(s), int(self.chunk_ids[p]))
            for s, p in zip(scores[0], positions[0]) if p >= 0
        ]
        pairs.sort(key=lambda t: (-t[0], t[1]))  # deterministic order on exact ties
        return [RankedHit(chunk_id=cid, rank=r, score=s) for r, (s, cid) in enumerate(pairs, start=1)]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / IDS_FILE, self.chunk_ids)
        if self._index is not None:
            _faiss().write_index(self._index, str(directory / INDEX_FILE))

    @classmethod
    def load(cls, directory: Path, expected_count: int | None = None) -> "FaissIndex":
        try:
            ids = np.load(directory / IDS_FILE, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexCorruptError(f"cannot read FAISS id mapping in {directory.name}: {exc}") from exc
        if expected_count is not None and len(ids) != expected_count:
            raise IndexCorruptError("FAISS id mapping size does not match the manifest")
        if len(ids) == 0:
            return cls(ids, None, 0)
        try:
            index = _faiss().read_index(str(directory / INDEX_FILE))
        except Exception as exc:  # faiss raises RuntimeError for unreadable files
            if isinstance(exc, MissingDependencyError):
                raise
            raise IndexCorruptError(f"cannot read FAISS index in {directory.name}: {exc}") from exc
        if index.ntotal != len(ids):
            raise IndexCorruptError("FAISS vector count does not match its id mapping")
        return cls(ids, index, int(index.d))
