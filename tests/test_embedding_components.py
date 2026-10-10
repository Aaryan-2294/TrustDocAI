import sys
import types

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from trustdoc.exceptions import MissingDependencyError, RetrievalError  # noqa: E402
from trustdoc.retrieval import (  # noqa: E402
    CrossEncoderReranker,
    HashingEmbedder,
    OverlapReranker,
    SentenceTransformerEmbedder,
)
from trustdoc.retrieval.embedding import clear_model_cache, l2_normalize  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def fake_st(monkeypatch):
    """A stand-in `sentence_transformers` module recording how it is used."""
    calls = {"st_init": [], "ce_init": [], "encode": [], "predict": []}

    class FakeST:
        def __init__(self, name, device=None):
            calls["st_init"].append((name, device))

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            calls["encode"].append((list(texts), kwargs))
            return np.tile(np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))  # un-normalised

    class FakeCE:
        output = None

        def __init__(self, name, device=None, max_length=None):
            calls["ce_init"].append((name, device, max_length))

        def predict(self, pairs, **kwargs):
            calls["predict"].append((list(pairs), kwargs))
            return FakeCE.output if FakeCE.output is not None else np.arange(len(pairs), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer, module.CrossEncoder = FakeST, FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    calls["CE"] = FakeCE
    return calls


# ---------------------------------------------------------------- hashing stub
def test_hashing_embedder_is_deterministic_normalised_and_shaped():
    e = HashingEmbedder(dimension=64)
    a, b = e.embed_documents(["load config file", "load config file"]), e.embed_documents(["other words"])
    assert a.shape == (2, 64) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a[0], a[1]) and not np.array_equal(a[0], b[0])
    assert e.embed_documents([]).shape == (0, 64) and e.embed_query("x").shape == (64,)
    assert HashingEmbedder.is_stub is True


def test_l2_normalize_handles_zero_vectors():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert out[0].tolist() == [0.0, 0.0] and np.allclose(out[1], [0.6, 0.8])


# ----------------------------------------------------- sentence-transformers wrapper
def test_sentence_transformer_embedder_uses_config_and_normalises(fake_st):
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu", batch_size=8)
    assert e.is_stub is False and e.dimension == 4
    vecs = e.embed_documents(["a", "b"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6) and vecs.dtype == np.float32
    _, kwargs = fake_st["encode"][-1]
    assert kwargs["batch_size"] == 8 and kwargs["normalize_embeddings"] is True and kwargs["show_progress_bar"] is False
    assert fake_st["st_init"] == [("sentence-transformers/all-MiniLM-L6-v2", "cpu")]
    assert e.embed_documents([]).shape[0] == 0


def test_model_is_loaded_once_and_shared_across_instances_and_calls(fake_st):
    first = SentenceTransformerEmbedder("m", "cpu")
    second = SentenceTransformerEmbedder("m", "cpu")
    for _ in range(3):
        first.embed_query("q")
    second.embed_documents(["x"])
    assert len(fake_st["st_init"]) == 1                      # not reloaded per query
    SentenceTransformerEmbedder("other", "cpu").embed_query("q")
    assert len(fake_st["st_init"]) == 2                      # different model -> separate load


def test_missing_sentence_transformers_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependencyError, match=r"\.\[retrieval\]"):
        SentenceTransformerEmbedder("m").embed_query("q")
    with pytest.raises(MissingDependencyError):
        CrossEncoderReranker("m").score("q", ["t"])


# ------------------------------------------------------------- cross-encoder wrapper
def test_cross_encoder_scores_pairs_in_order(fake_st):
    r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", batch_size=4)
    scores = r.score("my query", ["first", "second", "third"])
    assert scores == [0.0, 1.0, 2.0] and all(isinstance(s, float) for s in scores)
    pairs, kwargs = fake_st["predict"][-1]
    assert pairs == [("my query", "first"), ("my query", "second"), ("my query", "third")]
    assert kwargs["batch_size"] == 4 and kwargs["show_progress_bar"] is False
    assert fake_st["ce_init"] == [("cross-encoder/ms-marco-MiniLM-L-6-v2", "cpu", 512)]
    r.score("q2", ["x"])
    assert len(fake_st["ce_init"]) == 1                      # loaded once
    assert r.score("q", []) == [] and r.is_stub is False


def test_cross_encoder_output_shape_handling(fake_st):
    fake_st["CE"].output = np.array([[0.2], [0.9]], dtype=np.float32)
    assert CrossEncoderReranker("m").score("q", ["a", "b"]) == pytest.approx([0.2, 0.9])
    fake_st["CE"].output = np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32)   # multi-label: ambiguous
    with pytest.raises(RetrievalError):
        CrossEncoderReranker("m2").score("q", ["a", "b"])
    fake_st["CE"].output = np.array([np.nan, 0.5])
    assert CrossEncoderReranker("m3").score("q", ["a", "b"]) == [float("-inf"), 0.5]


# ------------------------------------------------------------------ overlap stub
def test_overlap_reranker_scores_fraction_of_query_tokens():
    r = OverlapReranker()
    assert r.score("retry failed", ["retry failed tasks", "retry only", "nothing here"]) == [1.0, 0.5, 0.0]
    assert r.score("", ["x"]) == [0.0] and r.is_stub is True
