"""Embedding models.

``Embedder`` is the interface the index manager and retriever depend on.
Implementations return L2-normalised float32 vectors, so FAISS inner product
equals cosine similarity.

* ``SentenceTransformerEmbedder`` - the real model (configured name, CPU by
  default). The model is loaded lazily on first use and cached per process, so
  it is never reloaded per query.
* ``HashingEmbedder`` - deterministic *lexical* stand-in (hashed bag of
  tokens). It is NOT semantic. It exists for fast offline tests and smoke runs.
"""

from __future__ import annotations

import hashlib
import math
import threading
from typing import Protocol, Sequence

import numpy as np

from trustdoc.exceptions import MissingDependencyError, RetrievalError
from trustdoc.logging_setup import get_logger
from trustdoc.retrieval.text import tokenize

log = get_logger("retrieval.embedding")

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LOCK = threading.Lock()


class Embedder(Protocol):
    model_name: str
    is_stub: bool

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dimension) float32, rows L2-normalised."""

    def embed_query(self, query: str) -> np.ndarray:
        """(dimension,) float32, L2-normalised."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _load_cached(kind: str, name: str, device: str, loader):
    key = (f"{kind}:{name}", device)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            log.info("Loading %s model %r on %s (first use)", kind, name, device)
            model = loader()
            _MODEL_CACHE[key] = model
        return model


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def _import_sentence_transformers():
    try:
        import sentence_transformers
        return sentence_transformers
    except ImportError as exc:
        raise MissingDependencyError(
            "sentence-transformers is required for semantic retrieval and reranking. "
            'Install it with: pip install -e ".[retrieval]"'
        ) from exc


class SentenceTransformerEmbedder:
    is_stub = False

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size

    @property
    def model(self):
        def load():
            st = _import_sentence_transformers()
            return st.SentenceTransformer(self.model_name, device=self._device)

        return _load_cached("embedder", self.model_name, self._device, load)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            list(texts), batch_size=self._batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]


class HashingEmbedder:
    """Deterministic hashed bag-of-tokens embedding (lexical, not semantic)."""

    is_stub = True

    def __init__(self, dimension: int = 256, model_name: str = "hashing-bow-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, n in counts.items():
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            vec[h % self._dimension] += sign * (1.0 + math.log(n))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]


def validate_vectors(vectors: np.ndarray, expected_rows: int) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise RetrievalError(
            f"embedder returned shape {vectors.shape}, expected ({expected_rows}, dim)"
        )
    if not np.isfinite(vectors).all():
        raise RetrievalError("embedder returned non-finite values")
