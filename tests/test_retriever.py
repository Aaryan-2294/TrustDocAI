import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from retrieval_support import CountingEmbedder, MarkerReranker, RecordingReranker, corpus_files  # noqa: E402

from trustdoc.config import Settings  # noqa: E402
from trustdoc.exceptions import IndexNotBuiltError, RetrievalError, StaleIndexError  # noqa: E402
from trustdoc.models import Project  # noqa: E402
from trustdoc.retrieval import IndexManager, Retriever, RetrievalMode  # noqa: E402

QUERY = "calculate_invoice_total tax_rate"


@pytest.fixture
def pid(make_project):
    return make_project()


# ----------------------------------------------------------------------- modes
def test_bm25_mode_populates_only_keyword_evidence(pid, retrieval_env):
    _, r, *_ = retrieval_env()
    res = r.search(pid, "calculate_invoice_total", top_k=3, mode=RetrievalMode.BM25)
    top = res.ranked[0]
    assert top.source_name == "billing.py" and top.method == RetrievalMode.BM25
    assert top.bm25_rank == 1 and top.bm25_score > 0
    assert top.faiss_rank is None and top.rrf_score is None and top.rerank_score is None
    assert [e.final_rank for e in res.ranked] == list(range(1, len(res.ranked) + 1))
    assert not res.threshold_applied and not res.abstained and res.evidence == res.ranked


def test_faiss_mode_populates_only_semantic_evidence(pid, retrieval_env):
    _, r, *_ = retrieval_env()
    res = r.search(pid, "send an email retrying on SMTP failures", top_k=3, mode=RetrievalMode.FAISS)
    top = res.ranked[0]
    assert top.source_name == "mailer.py" and top.faiss_rank == 1
    assert -1.0 <= top.faiss_score <= 1.0 and top.bm25_rank is None and top.rrf_score is None


def test_hybrid_mode_fuses_ranks_with_configured_rrf_k(pid, retrieval_env, settings):
    _, r, *_ = retrieval_env()
    res = r.search(pid, QUERY, top_k=5, mode=RetrievalMode.HYBRID)
    k = settings.retrieval.rrf_k
    both = next(e for e in res.ranked if e.bm25_rank and e.faiss_rank)
    assert both.rrf_score == pytest.approx(1 / (k + both.bm25_rank) + 1 / (k + both.faiss_rank))
    assert [e.rrf_rank for e in res.ranked] == list(range(1, len(res.ranked) + 1))
    scores = [e.rrf_score for e in res.ranked]
    assert scores == sorted(scores, reverse=True) and res.ranked[0].source_name == "billing.py"
    assert all(e.rerank_score is None for e in res.ranked)


def test_hybrid_rerank_preserves_retrieval_metadata_and_orders_by_reranker(pid, retrieval_env):
    _, r, *_ = retrieval_env()
    res = r.search(pid, QUERY, top_k=5)
    assert res.mode == RetrievalMode.HYBRID_RERANK
    rer = [e.rerank_score for e in res.ranked]
    assert rer == sorted(rer, reverse=True) and res.best_score == max(rer)
    top = res.ranked[0]
    assert top.source_name == "billing.py"
    assert top.rrf_rank is not None and top.rrf_score is not None and top.rerank_rank == 1
    assert top.bm25_rank is not None and top.faiss_rank is not None   # original retrieval info kept
    assert [e.final_rank for e in res.ranked] == [1, 2, 3, 4, 5][: len(res.ranked)]


def test_results_are_deterministic(pid, retrieval_env):
    _, r, *_ = retrieval_env()
    a = [(e.chunk_id, e.rerank_score) for e in r.search(pid, QUERY, top_k=6).ranked]
    b = [(e.chunk_id, e.rerank_score) for e in r.search(pid, QUERY, top_k=6).ranked]
    assert a == b


# ------------------------------------------------------- rerank only candidates
def test_reranker_scores_only_the_fused_candidate_pool(db, settings, pid, retrieval_env):
    small = Settings.from_env({"TRUSTDOC_DATA_DIR": str(settings.storage.data_dir), "TRUSTDOC_LOG_TO_FILE": "false",
                               "TRUSTDOC_CANDIDATE_K": "5", "TRUSTDOC_FINAL_K": "3"})
    _, r, _, rer = retrieval_env(settings_override=small)
    total = len(r.index_manager.ensure(pid).bm25)
    assert total > 12
    r.search(pid, "unrelated utility number value task filler")
    assert rer.batch_sizes[-1] == 5 < total                       # candidate_k, not the whole knowledge base
    r.search(pid, "unrelated utility number value task filler", top_k=8)
    assert rer.batch_sizes[-1] == 8                               # depth grows with top_k
    r.search(pid, "x", mode=RetrievalMode.HYBRID)
    r.search(pid, "x", mode=RetrievalMode.BM25)
    assert len(rer.batch_sizes) == 2                              # other modes never call the reranker


# -------------------------------------------------------------------- threshold
@pytest.fixture
def scripted(retrieval_env, pid):
    scorer = MarkerReranker({"calculate_invoice_total": 0.9, "create_user": 0.55}, default=0.1)
    _, r, *_ = retrieval_env(reranker=scorer)
    return r


def test_threshold_keeps_only_sufficiently_relevant_evidence(scripted, pid):
    res = scripted.search(pid, "calculate_invoice_total create_user", top_k=6, threshold=0.5)
    assert res.threshold_applied and res.threshold == 0.5 and not res.abstained
    assert {e.source_name for e in res.evidence} == {"billing.py", "users.py"}
    assert len(res.ranked) == 6 and len(res.evidence) < len(res.ranked)   # both views available
    assert all(e.rerank_score >= 0.5 for e in res.evidence) and res.best_score == 0.9


def test_below_threshold_abstains_but_still_exposes_ranked_candidates(scripted, pid):
    res = scripted.search(pid, "calculate_invoice_total", top_k=4, threshold=0.95)
    assert res.abstained and res.evidence == [] and res.threshold_applied
    assert len(res.ranked) == 4 and res.best_score == 0.9


def test_threshold_boundary_is_inclusive(scripted, pid):
    at = scripted.search(pid, "calculate_invoice_total", top_k=4, threshold=0.9)
    assert not at.abstained and [e.rerank_score for e in at.evidence] == [0.9]
    assert scripted.search(pid, "calculate_invoice_total", top_k=4, threshold=0.9000001).abstained


def test_threshold_can_be_disabled_and_config_default_is_used(scripted, pid):
    off = scripted.search(pid, "calculate_invoice_total", top_k=4, apply_threshold=False)
    assert off.evidence == off.ranked and not off.threshold_applied and off.threshold is None and not off.abstained
    default = scripted.search(pid, "calculate_invoice_total", top_k=4)
    assert default.threshold == 0.0 and default.threshold_applied    # the uncalibrated placeholder


def test_threshold_is_ignored_for_modes_without_a_comparable_score(scripted, pid):
    res = scripted.search(pid, "calculate_invoice_total", mode=RetrievalMode.HYBRID, threshold=999)
    assert not res.threshold_applied and not res.abstained and res.evidence == res.ranked


def test_placeholder_threshold_warning_is_logged_once(pid, retrieval_env, caplog):
    import logging
    _, r, *_ = retrieval_env()
    with caplog.at_level(logging.WARNING, logger="trustdoc.retrieval"):
        r.search(pid, QUERY)
        r.search(pid, QUERY)
    assert sum("uncalibrated placeholder" in m for m in caplog.messages) == 1


# ----------------------------------------------------------------- input errors
def test_input_validation(db, settings, pid, retrieval_env):
    _, r, *_ = retrieval_env()
    for bad in ("", "   "):
        with pytest.raises(RetrievalError):
            r.search(pid, bad)
    with pytest.raises(ValueError):
        r.search(pid, "x", top_k=0)
    emb = CountingEmbedder()
    no_rerank = Retriever(db, settings, IndexManager(db, settings, emb), emb, None)
    assert no_rerank.search(pid, QUERY, mode=RetrievalMode.HYBRID).ranked
    with pytest.raises(RetrievalError, match="reranker"):
        no_rerank.search(pid, QUERY)


# ------------------------------------------------------- provenance / citations
def test_citations_and_text_match_the_database_and_stored_original(db, settings, pid, retrieval_env, service):
    _, r, *_ = retrieval_env()
    res = r.search(pid, "render an invoice as a PDF document InvoiceRenderer render_pdf", top_k=3)
    top = res.ranked[0]
    assert top.citation.qualified_name == "billing.InvoiceRenderer.render_pdf"
    with db.session() as s:
        chunk, source = s.chunks.get(top.chunk_id), s.sources.get(top.source_id)
    assert top.text == chunk.text and top.citation.label == chunk.location_label(source.name)
    assert (top.citation.start_line, top.citation.end_line) == (chunk.start_line, chunk.end_line)
    assert top.citation.source_name == "billing.py" and top.citation.chunk_id == chunk.id
    original = service.store.read(source.stored_path).decode().split("\n")
    assert top.text == "\n".join(original[chunk.start_line - 1 : chunk.end_line])   # line range is exact
    assert top.entity_id == chunk.entity_id and top.metadata["entity_type"] == "method"

    md = r.search(pid, "Installation pip install modules", top_k=3, mode=RetrievalMode.BM25).ranked[0]
    assert md.citation.heading == "Shop > Installation" and md.citation.label.startswith("README.md:")
    assert md.citation.label.endswith("\u00a7 Shop > Installation")


def test_physical_paths_never_appear_in_results(db, settings, pid, retrieval_env):
    _, r, *_ = retrieval_env()
    res = r.search(pid, QUERY, top_k=6)
    blob = json.dumps(res.to_dict())
    with db.session() as s:
        stored = [src.stored_path for src in s.sources.list_for_project(pid)]
        hashes = [src.content_hash for src in s.sources.list_for_project(pid)]
    assert stored and not any(p in blob for p in stored) and not any(h in blob for h in hashes)
    assert str(settings.storage.data_dir) not in blob and "uploads" not in blob
    assert "stored_path" not in blob


# ------------------------------------------------------------ project isolation
def test_search_never_returns_another_projects_chunks(db, make_project, retrieval_env):
    shared = {"shared.py": b'def shared_helper(x):\n    """Shared helper for both projects."""\n    return x\n'}
    a = make_project(files={**corpus_files(), **shared})
    b = make_project(files={**{f"b{i}.py": f"def b_fn_{i}(): ...\n".encode() for i in range(8)}, **shared})
    _, r, *_ = retrieval_env()
    with db.session() as s:
        ids = {pid_: {c.id for c in s.chunks.list_for_project(pid_)} for pid_ in (a, b)}
    for query in ("shared_helper", "calculate_invoice_total", "b_fn_3"):
        for mode in RetrievalMode:
            for pid_ in (a, b):
                res = r.search(pid_, query, top_k=10, mode=mode, apply_threshold=False)
                assert {e.chunk_id for e in res.ranked} <= ids[pid_], (query, mode, pid_)
                assert {e.project_id for e in res.ranked} <= {pid_}
    assert not r.search(b, "calculate_invoice_total", mode=RetrievalMode.BM25).ranked


# ---------------------------------------------------------- index lifecycle use
def test_search_builds_missing_index_but_never_reembeds_per_query(pid, retrieval_env):
    manager, r, emb, _ = retrieval_env()
    assert manager.status(pid).state.value == "missing"
    r.search(pid, QUERY)
    assert manager.status(pid).is_fresh
    embedded = emb.documents_embedded
    for q in ("tax rate", "send email", "lru cache", "delete user", "circle area"):
        r.search(pid, q)
    assert emb.documents_embedded == embedded                      # indexing happened exactly once
    assert emb.queries_embedded == 6                               # only the queries were embedded


def test_auto_index_rebuilds_when_stale_and_manual_mode_refuses(db, pid, make_service, retrieval_env):
    manager, r, *_ = retrieval_env()
    strict_manager, strict, *_ = retrieval_env(auto_index=False)
    with pytest.raises(IndexNotBuiltError):
        strict.search(pid, QUERY)
    r.search(pid, QUERY)
    make_service().ingest_bytes(pid, "feature.py", b"def brand_new_feature_xyz(): ...\n")
    with pytest.raises(StaleIndexError):
        strict.search(pid, "brand_new_feature_xyz")
    res = r.search(pid, "brand_new_feature_xyz", top_k=3, mode=RetrievalMode.BM25)
    assert res.ranked[0].source_name == "feature.py"


def test_empty_project_abstains(db, retrieval_env):
    with db.session() as s:
        empty = s.projects.create(Project(name="nothing")).id
    _, r, *_ = retrieval_env()
    for mode in RetrievalMode:
        res = r.search(empty, "anything", mode=mode)
        assert res.ranked == [] and res.evidence == [] and res.abstained


def test_result_has_timings_and_is_json_serialisable(pid, retrieval_env):
    _, r, *_ = retrieval_env()
    res = r.search(pid, QUERY)
    assert {"index_ms", "bm25_ms", "faiss_ms", "rrf_ms", "rerank_ms", "total_ms"} <= set(res.timings_ms)
    data = json.loads(json.dumps(res.to_dict()))
    assert data["mode"] == "hybrid_rerank" and data["ranked"][0]["method"] == "hybrid_rerank"
