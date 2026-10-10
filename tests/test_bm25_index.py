import pytest

pytest.importorskip("rank_bm25")

from trustdoc.exceptions import IndexCorruptError  # noqa: E402
from trustdoc.retrieval import Bm25Index  # noqa: E402

DOCS = {
    11: "def load_config(path): read the json configuration file",
    12: "class WorkerPool: starts worker threads and shuts them down",
    13: "def compute_backoff(attempt): exponential retry delay with jitter",
    14: "def compact(vacuum): delete old completed tasks and reclaim space",
    15: "installation guide: pip install the package",
    16: "retry retry retry: retry failed tasks again, retry policy for retry",
    17: "def heartbeat(worker_id): report that a worker is alive",
    18: "def fetch_pending(limit): select pending tasks ordered by priority",
    19: "unrelated notes about the weather and lunch",
}


@pytest.fixture
def index():
    return Bm25Index.build(list(DOCS), list(DOCS.values()))


def test_exact_identifier_ranks_first(index):
    for ident, expected in [("compute_backoff", 13), ("load_config", 11), ("fetch_pending", 18), ("WorkerPool", 12)]:
        assert index.search(ident, 3)[0].chunk_id == expected


def test_identifier_parts_match_natural_language_query(index):
    assert index.search("load config", 1)[0].chunk_id == 11


def test_ranking_orders_by_score_with_one_based_ranks(index):
    hits = index.search("retry", 5)
    assert hits[0].chunk_id == 16                      # term-frequency-heavy document first
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_documents_without_any_query_term_are_not_returned(index):
    assert {h.chunk_id for h in index.search("heartbeat", 10)} == {17}
    assert index.search("zzzunknownterm", 5) == [] and index.search("   ", 5) == []


def test_top_k_limits_results(index):
    assert len(index.search("worker tasks", 2)) == 2 and index.search("worker", 0) == []


def test_persistence_roundtrip_gives_identical_ranking(tmp_path, index):
    index.save(tmp_path / "ix")
    loaded = Bm25Index.load(tmp_path / "ix", expected_count=len(DOCS))
    for q in ["retry", "worker tasks", "load_config"]:
        assert [(h.chunk_id, h.score) for h in index.search(q, 5)] == [(h.chunk_id, h.score) for h in loaded.search(q, 5)]


def test_load_detects_corruption(tmp_path, index):
    index.save(tmp_path / "ix")
    with pytest.raises(IndexCorruptError):
        Bm25Index.load(tmp_path / "ix", expected_count=3)
    (tmp_path / "ix" / "bm25.json").write_text("{not json")
    with pytest.raises(IndexCorruptError):
        Bm25Index.load(tmp_path / "ix")


def test_empty_corpus_and_deterministic_ties():
    assert Bm25Index.build([], []).search("anything", 3) == []
    twin = Bm25Index.build([5, 3, 4, 6, 7, 8], ["alpha beta", "alpha beta", "gamma", "delta", "epsilon", "zeta"])
    assert [h.chunk_id for h in twin.search("alpha", 2)] == [3, 5]   # equal scores -> lower id first
