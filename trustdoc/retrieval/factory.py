"""Assemble a ready-to-use Retriever from settings (models stay lazy)."""

from __future__ import annotations

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.retrieval.embedding import Embedder, SentenceTransformerEmbedder
from trustdoc.retrieval.index_manager import IndexManager
from trustdoc.retrieval.reranker import CrossEncoderReranker, Reranker
from trustdoc.retrieval.retriever import Retriever


def create_embedder(settings: Settings) -> Embedder:
    cfg = settings.retrieval
    return SentenceTransformerEmbedder(cfg.embedding_model, cfg.device, cfg.embedding_batch_size)


def create_reranker(settings: Settings) -> Reranker:
    cfg = settings.retrieval
    return CrossEncoderReranker(cfg.reranker_model, cfg.device)


def create_retriever(
    db: Database,
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    use_reranker: bool = True,
    auto_index: bool = True,
) -> Retriever:
    """Default wiring: configured Sentence-Transformer + Cross-Encoder, CPU.

    Nothing heavy is loaded here; models load on first use and are cached. Pass
    your own ``embedder``/``reranker`` to swap implementations.
    """
    embedder = embedder or create_embedder(settings)
    if reranker is None and use_reranker:
        reranker = create_reranker(settings)
    manager = IndexManager(db, settings, embedder)
    return Retriever(db, settings, manager, embedder, reranker, auto_index=auto_index)
