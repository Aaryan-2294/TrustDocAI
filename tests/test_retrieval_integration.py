"""End to end: Part 2 ingestion -> index build -> query -> evidence with intact provenance."""

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")
pytest.importorskip("pymupdf")

from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset  # noqa: E402
from trustdoc.models import Project  # noqa: E402
from trustdoc.retrieval import HashingEmbedder, OverlapReranker, RetrievalMode, create_retriever  # noqa: E402


@pytest.fixture
def sample(db, settings, service):
    dataset = load_dataset(DEFAULT_QUERIES)
    with db.session() as s:
        pid = s.projects.create(Project(name="sample")).id
    batch = service.ingest_directory(pid, dataset.project_dir)
    assert batch.ok, batch.failures
    retriever = create_retriever(db, settings, embedder=HashingEmbedder(), reranker=OverlapReranker())
    return pid, retriever


def test_code_provenance_survives_the_whole_pipeline(db, service, sample):
    pid, retriever = sample
    res = retriever.search(pid, "compute_backoff", top_k=3)
    top = res.ranked[0]
    assert top.citation.qualified_name == "taskqueue.workers.compute_backoff"
    assert top.source_name == "taskqueue/workers.py" and top.chunk_type == "code"
    assert top.citation.label == f"taskqueue/workers.py:{top.citation.start_line}-{top.citation.end_line}"
    with db.session() as s:
        source = s.sources.get(top.source_id)
        entity = s.entities.get(top.entity_id)
    assert entity.qualified_name == top.citation.qualified_name and entity.entity_type.value == "function"
    original = service.store.read(source.stored_path).decode().split("\n")
    assert top.text == "\n".join(original[top.citation.start_line - 1 : top.citation.end_line])
    assert top.bm25_rank == 1 and top.rrf_rank is not None and top.rerank_rank == 1   # scores from every stage


def test_pdf_and_markdown_provenance(sample):
    pid, retriever = sample
    pdf = retriever.search(pid, "exponential backoff with full jitter and the 300 second cap", top_k=3).ranked[0]
    assert pdf.citation.label == "reference.pdf, p.2" and pdf.citation.page_number == 2
    assert pdf.chunk_type == "pdf_text" and pdf.citation.start_line is None
    md = retriever.search(pid, "which environment variables override the configuration file", top_k=3).ranked[0]
    assert md.citation.heading == "TaskQueue > Configuration" and md.source_name == "README.md"
    assert md.citation.label.endswith("\u00a7 TaskQueue > Configuration")


@pytest.mark.parametrize("query, source", [
    ("fetch_pending", "taskqueue/storage.py"),
    ("delay_seconds", "taskqueue/queue.py"),
    ("how long are completed tasks retained before deletion", "reference.pdf"),
])
def test_every_mode_returns_provenance_rich_evidence(sample, query, source):
    pid, retriever = sample
    for mode in RetrievalMode:
        res = retriever.search(pid, query, top_k=5, mode=mode, apply_threshold=False)
        assert res.ranked, mode
        for ev in res.ranked:
            assert ev.citation.source_name == ev.source_name and ev.citation.label.startswith(ev.source_name)
            assert ev.project_id == pid and ev.text and ev.chunk_id == ev.citation.chunk_id
    assert retriever.search(pid, query, top_k=3).ranked[0].source_name == source


def test_abstention_flow_with_a_strict_threshold(sample):
    pid, retriever = sample
    res = retriever.search(pid, "compute_backoff", top_k=3, threshold=1.5)   # unreachable for 0..1 scores
    assert res.abstained and res.evidence == [] and res.ranked
