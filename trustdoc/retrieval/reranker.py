"""Second-stage rerankers (query + candidate text -> relevance score).

``Reranker`` is replaceable: anything with ``model_name``, ``is_stub`` and
``score(query, texts)`` works. Only the fused candidate set is ever scored,
never the whole knowledge base.

Score scale depends on the model and library version (probabilities in [0, 1]
or raw logits), which is why the relevance threshold must be calibrated per
model rather than assumed.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np

from trustdoc.exceptions import RetrievalError
from trustdoc.retrieval.embedding import _import_sentence_transformers, _load_cached
from trustdoc.retrieval.text import tokenize


class Reranker(Protocol):
    model_name: str
    is_stub: bool

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text (higher = more relevant), same order."""


class CrossEncoderReranker:
    is_stub = False

    def __init__(
        self, model_name: str, device: str = "cpu", batch_size: int = 16, max_length: int = 512
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def model(self):
        def load():
            st = _import_sentence_transformers()
            return st.CrossEncoder(self.model_name, device=self._device, max_length=self._max_length)

        return _load_cached("reranker", self.model_name, self._device, load)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        raw = self.model.predict(
            [(query, t) for t in texts], batch_size=self._batch_size, show_progress_bar=False
        )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.shape != (len(texts),):
            raise RetrievalError(
                f"cross-encoder returned shape {arr.shape}; expected one score per candidate"
            )
        return [float(x) if math.isfinite(x) else float("-inf") for x in arr]


class OverlapReranker:
    """Deterministic stand-in: fraction of query tokens present in the text (0..1).

    Lexical only - NOT a language model. For tests and offline smoke runs.
    """

    is_stub = True

    def __init__(self, model_name: str = "token-overlap-v1") -> None:
        self.model_name = model_name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(tokenize(t))) / len(q) for t in texts]
