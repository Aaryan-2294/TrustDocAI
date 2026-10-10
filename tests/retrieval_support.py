"""Shared helpers for the retrieval tests: a small corpus and controllable stand-ins."""

from __future__ import annotations

from trustdoc.retrieval.embedding import HashingEmbedder
from trustdoc.retrieval.reranker import OverlapReranker


class CountingEmbedder(HashingEmbedder):
    """Hashing embedder that records how much it was asked to embed."""

    def __init__(self, model_name: str = "hashing-bow-v1") -> None:
        super().__init__(model_name=model_name)
        self.documents_embedded = 0
        self.queries_embedded = 0

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return super().embed_documents(texts)

    def embed_query(self, query):
        self.queries_embedded += 1
        return HashingEmbedder.embed_documents(self, [query])[0]   # bypass the document counter


class RecordingReranker(OverlapReranker):
    """Overlap reranker that records how many candidates it scored per call."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def score(self, query, texts):
        self.batch_sizes.append(len(texts))
        return super().score(query, texts)


class MarkerReranker:
    """Scores a text by the first marker substring it contains (test control)."""

    is_stub = True
    model_name = "marker-reranker"

    def __init__(self, scores: dict[str, float], default: float = 0.0) -> None:
        self._scores, self._default = scores, default

    def score(self, query, texts):
        out = []
        for text in texts:
            out.append(next((v for k, v in self._scores.items() if k in text), self._default))
        return out

from corpus_support import corpus_files  # noqa: E402,F401  (re-exported for the tests)
