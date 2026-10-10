import csv
import json

import pytest

from trustdoc.evaluation.dataset import EvaluationError, EvalDataset, EvalQuery, RelevantRef, load_dataset, resolve_relevance
from trustdoc.evaluation.metrics import first_relevant_rank, mean, recall_at_k, reciprocal_rank


# ----------------------------------------------------------------- metrics (hand computed)
def test_recall_at_k_counts_groups_found():
    groups = [{1, 2}, {3}, {9}]
    ranked = [5, 2, 7, 3, 8]
    assert recall_at_k(ranked, groups, 1) == 0.0
    assert recall_at_k(ranked, groups, 2) == pytest.approx(1 / 3)       # group {1,2} found via chunk 2
    assert recall_at_k(ranked, groups, 5) == pytest.approx(2 / 3)       # {9} never retrieved
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], 3)


def test_reciprocal_rank_and_first_rank():
    groups = [{4}, {6}]
    assert reciprocal_rank([1, 6, 4], groups) == 0.5 and first_relevant_rank([1, 6, 4], groups) == 2
    assert reciprocal_rank([4], groups) == 1.0
    assert reciprocal_rank([1, 2, 3], groups) == 0.0 and first_relevant_rank([1, 2, 3], groups) is None
    assert mean([1.0, 0.5, 0.0]) == 0.5 and mean([]) == 0.0


# ------------------------------------------------------------------------ dataset
def test_bundled_dataset_covers_every_required_query_type():
    ds = load_dataset()
    assert ds.illustrative is True and "ILLUSTRATIVE" in ds.description
    assert set(ds.categories()) == {
        "exact_function_name", "exact_parameter_name", "semantic_paraphrase", "class_related",
        "readme_documentation", "pdf_reference", "mixed_code_documentation"}
    assert len({q.id for q in ds.queries}) == len(ds.queries) == 16
    assert (ds.project_dir / "README.md").is_file() and (ds.project_dir / "reference.pdf").is_file()


def test_unmatched_labels_fail_loudly(db, make_project):
    pid = make_project()
    ds = EvalDataset("x", "", True, load_dataset().project_dir,
                     [EvalQuery("q1", "c", "t", (RelevantRef("billing.py", qualified_name="billing.nope"),))])
    with db.session() as s:
        chunks, sources = s.chunks.list_for_project(pid), s.sources.list_for_project(pid)
    with pytest.raises(EvaluationError, match="q1"):
        resolve_relevance(ds, chunks, {x.id: x.name for x in sources})
    ok = EvalDataset("x", "", True, ds.project_dir,
                     [EvalQuery("q2", "c", "t", (RelevantRef("billing.py", qualified_name="billing.calculate_invoice_total"),))])
    assert len(resolve_relevance(ok, chunks, {x.id: x.name for x in sources})["q2"][0]) == 1


# ------------------------------------------------------------ full evaluation run
@pytest.fixture(scope="module")
def stub_report(tmp_path_factory):
    pytest.importorskip("faiss")
    pytest.importorskip("rank_bm25")
    pytest.importorskip("pymupdf")
    from trustdoc.evaluation import run_sample_evaluation
    from trustdoc.retrieval import HashingEmbedder, OverlapReranker

    out = tmp_path_factory.mktemp("eval")
    return out, run_sample_evaluation(out, HashingEmbedder(), OverlapReranker())


def test_resolves_every_label_on_the_real_sample_corpus(db, service):
    pytest.importorskip("pymupdf")
    ds = load_dataset()
    from trustdoc.models import Project
    with db.session() as s:
        pid = s.projects.create(Project(name="s")).id
    assert service.ingest_directory(pid, ds.project_dir).ok
    with db.session() as s:
        chunks, sources = s.chunks.list_for_project(pid), s.sources.list_for_project(pid)
    resolved = resolve_relevance(ds, chunks, {x.id: x.name for x in sources})
    assert all(g for groups in resolved.values() for g in groups)            # every label matched >= 1 chunk
    assert len(resolved["pdf-2"][0]) == 1                                    # one chunk per PDF page


def test_outputs_are_written_and_machine_readable(stub_report):
    out, report = stub_report
    data = json.loads((out / "results.json").read_text())
    assert set(data["results"]) == {"bm25", "faiss", "hybrid_rrf", "hybrid_rerank"}
    meta = data["meta"]
    assert meta["ks"] == [1, 3, 5, 10] and meta["dataset"]["illustrative"] is True
    assert meta["uses_stand_in_models"] is True and meta["embedder"]["is_stub"] is True
    for pipeline in data["results"].values():
        overall = pipeline["overall"]
        assert all(0.0 <= overall[k] <= 1.0 for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr"))
        assert overall["recall@1"] <= overall["recall@3"] <= overall["recall@5"] <= overall["recall@10"]
        assert len(pipeline["queries"]) == 16 and len(pipeline["by_category"]) == 7
    rows = list(csv.DictReader((out / "per_query.csv").open()))
    assert len(rows) == 4 * 16 and {"pipeline", "query_id", "reciprocal_rank", "recall@5"} <= set(rows[0])


def test_summary_states_what_the_numbers_are_and_are_not(stub_report):
    out, _ = stub_report
    text = (out / "summary.md").read_text()
    assert "illustrative fixture" in text and "NOT the performance of the real" in text
    assert "A. BM25 only" in text and "D. Hybrid + Cross-Encoder rerank" in text and "Recall@5" in text


def test_reported_metrics_are_recomputable_from_the_saved_rankings(stub_report):
    """Guards against fabricated numbers: every aggregate must follow from the per-query rankings."""
    out, _ = stub_report
    data = json.loads((out / "results.json").read_text())
    for name, pipeline in data["results"].items():
        rr, rec5 = [], []
        for q in pipeline["queries"]:
            first = next((h["rank"] for h in q["retrieved"] if h["relevant"]), None)
            rr.append(0.0 if first is None else 1 / first)
            assert q["first_relevant_rank"] == first
        assert pipeline["overall"]["mrr"] == pytest.approx(sum(rr) / len(rr), abs=1e-4), name


def test_without_a_reranker_the_rerank_pipeline_is_skipped(tmp_path):
    pytest.importorskip("faiss"); pytest.importorskip("rank_bm25"); pytest.importorskip("pymupdf")
    from trustdoc.evaluation import run_sample_evaluation
    from trustdoc.retrieval import HashingEmbedder
    report = run_sample_evaluation(tmp_path, HashingEmbedder(), None)
    assert set(report.pipelines) == {"bm25", "faiss", "hybrid_rrf"}


def test_cli_eval_retrieval_with_stand_in_models(tmp_path, capsys, monkeypatch):
    pytest.importorskip("faiss"); pytest.importorskip("rank_bm25"); pytest.importorskip("pymupdf")
    from trustdoc.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["eval-retrieval", "--embedder", "hashing", "--reranker", "overlap", "--out", str(tmp_path / "o")]) == 0
    assert "lexical stand-in models" in capsys.readouterr().out
    assert (tmp_path / "o" / "results.json").is_file()
    assert not (tmp_path / "data" / "trustdoc.db").exists()           # the user's own database is untouched
