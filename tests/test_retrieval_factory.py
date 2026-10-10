import pytest

pytest.importorskip("numpy")

from trustdoc.retrieval import CrossEncoderReranker, SentenceTransformerEmbedder, create_retriever  # noqa: E402
from trustdoc.retrieval.embedding import _MODEL_CACHE  # noqa: E402


def test_default_wiring_uses_configuration_and_loads_nothing(db, settings):
    _MODEL_CACHE.clear()
    retriever = create_retriever(db, settings)
    emb, rer = retriever._embedder, retriever._reranker
    assert isinstance(emb, SentenceTransformerEmbedder) and isinstance(rer, CrossEncoderReranker)
    assert emb.model_name == settings.retrieval.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert rer.model_name == settings.retrieval.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert emb._device == "cpu" and emb._batch_size == settings.retrieval.embedding_batch_size
    assert not _MODEL_CACHE                                   # constructing a retriever never loads a model


def test_reranker_can_be_disabled_or_swapped(db, settings):
    assert create_retriever(db, settings, use_reranker=False)._reranker is None
    custom = object()
    assert create_retriever(db, settings, reranker=custom)._reranker is custom
