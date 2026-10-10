"""Hybrid retrieval engine (Part 3).

Light imports (types, tokenizer, RRF, threshold) work on a core install. The
rest needs the ``retrieval`` extra and is imported lazily on first access, so
``import trustdoc.retrieval`` never pulls in numpy/faiss/torch by itself.
"""

from __future__ import annotations

import importlib

from trustdoc.retrieval.fusion import reciprocal_rank_fusion
from trustdoc.retrieval.text import chunk_index_text, tokenize
from trustdoc.retrieval.threshold import ThresholdDecision, apply_relevance_threshold
from trustdoc.retrieval.types import (
    Citation,
    Contribution,
    Evidence,
    FusedHit,
    RankedHit,
    RankedList,
    RetrievalMode,
    SearchResult,
)

_LAZY = {
    "Retriever": "retriever",
    "create_retriever": "factory",
    "create_embedder": "factory",
    "create_reranker": "factory",
    "IndexManager": "index_manager",
    "IndexState": "index_manager",
    "IndexStatus": "index_manager",
    "IndexManifest": "index_manager",
    "ProjectIndex": "index_manager",
    "FaissIndex": "faiss_index",
    "Bm25Index": "bm25_index",
    "SentenceTransformerEmbedder": "embedding",
    "HashingEmbedder": "embedding",
    "CrossEncoderReranker": "reranker",
    "OverlapReranker": "reranker",
}

__all__ = [
    "reciprocal_rank_fusion", "chunk_index_text", "tokenize", "ThresholdDecision",
    "apply_relevance_threshold", "Citation", "Contribution", "Evidence", "FusedHit",
    "RankedHit", "RankedList", "RetrievalMode", "SearchResult", *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(f"trustdoc.retrieval.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module 'trustdoc.retrieval' has no attribute {name!r}")
