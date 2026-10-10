import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

import numpy as np  # noqa: E402

from trustdoc.exceptions import IndexCorruptError, RetrievalError  # noqa: E402
from trustdoc.retrieval import FaissIndex  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def vectors():
    return np.stack([unit([1, 0, 0]), unit([0.8, 0.6, 0]), unit([0, 1, 0]), unit([0, 0, 1])])


IDS = [100, 205, 7, 42]   # deliberately not 0..n-1: positions must map back to chunk ids


def test_search_maps_positions_back_to_chunk_ids_best_first(vectors):
    idx = FaissIndex.build(IDS, vectors)
    hits = idx.search(unit([1, 0.1, 0]), 3)
    assert [h.chunk_id for h in hits] == [100, 205, 7]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_scores_are_cosine_similarity(vectors):
    idx = FaissIndex.build(IDS, vectors)
    q = unit([0.3, 0.9, 0.2])
    for h in idx.search(q, 4):
        expected = float(vectors[IDS.index(h.chunk_id)] @ q)
        assert h.score == pytest.approx(expected, abs=1e-5)
        assert -1.0001 <= h.score <= 1.0001


def test_unnormalised_inputs_are_normalised_so_metric_stays_cosine():
    idx = FaissIndex.build([1, 2], np.array([[10.0, 0, 0], [0, 0.001, 0]], dtype=np.float32))
    hits = idx.search(np.array([5.0, 0, 0], dtype=np.float32), 2)
    assert hits[0].chunk_id == 1 and hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_index_and_zero_k(vectors):
    idx = FaissIndex.build(IDS, vectors)
    assert len(idx.search(vectors[0], 50)) == 4 and idx.search(vectors[0], 0) == []
    assert len(idx) == 4 and idx.dimension == 3


def test_persistence_roundtrip_gives_identical_results(tmp_path, vectors):
    idx = FaissIndex.build(IDS, vectors)
    idx.save(tmp_path / "ix")
    loaded = FaissIndex.load(tmp_path / "ix", expected_count=4)
    q = unit([0.5, 0.5, 0.1])
    assert [(h.chunk_id, h.score) for h in idx.search(q, 4)] == [(h.chunk_id, h.score) for h in loaded.search(q, 4)]
    assert loaded.chunk_ids.tolist() == IDS


def test_load_detects_corruption(tmp_path, vectors):
    FaissIndex.build(IDS, vectors).save(tmp_path / "ix")
    with pytest.raises(IndexCorruptError):
        FaissIndex.load(tmp_path / "ix", expected_count=3)
    (tmp_path / "ix" / "faiss.index").write_bytes(b"garbage")
    with pytest.raises(IndexCorruptError):
        FaissIndex.load(tmp_path / "ix", expected_count=4)
    with pytest.raises(IndexCorruptError):
        FaissIndex.load(tmp_path / "missing")


def test_query_dimension_mismatch_is_reported(vectors):
    with pytest.raises(RetrievalError, match="rebuild"):
        FaissIndex.build(IDS, vectors).search(np.ones(5, dtype=np.float32), 2)


def test_empty_index_roundtrip(tmp_path):
    idx = FaissIndex.build([], None)
    assert len(idx) == 0 and idx.search(np.ones(3, dtype=np.float32), 5) == []
    idx.save(tmp_path / "e")
    assert len(FaissIndex.load(tmp_path / "e", 0)) == 0


def test_build_validates_inputs(vectors):
    with pytest.raises(RetrievalError):
        FaissIndex.build([1, 1, 2, 3], vectors)       # duplicate ids
    with pytest.raises(RetrievalError):
        FaissIndex.build([1, 2], vectors)             # size mismatch
