import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from retrieval_support import CountingEmbedder, corpus_files  # noqa: E402

from trustdoc.exceptions import (  # noqa: E402
    IndexCorruptError,
    IndexNotBuiltError,
    NotFoundError,
    StaleIndexError,
)
from trustdoc.models import Project  # noqa: E402
from trustdoc.retrieval import IndexManager, IndexState  # noqa: E402


def db_chunk_ids(db, pid):
    with db.session() as s:
        return {c.id for c in s.chunks.list_for_project(pid)}


def test_missing_then_built_with_metadata(db, make_project, retrieval_env):
    pid = make_project()
    manager, _, emb, _ = retrieval_env()
    st = manager.status(pid)
    assert st.state == IndexState.MISSING and manager.is_stale(pid) and manager.metadata(pid) is None
    with pytest.raises(IndexNotBuiltError):
        manager.load(pid)

    m = manager.build(pid)
    assert m.project_id == pid and m.chunk_count == len(db_chunk_ids(db, pid)) > 10
    assert m.embedding_model == "hashing-bow-v1" and m.embedding_dimension == 256 and m.source_count == len(corpus_files())
    assert "cosine" in m.faiss_metric and m.bm25["library"].startswith("rank_bm25")
    assert emb.documents_embedded == m.chunk_count
    assert manager.status(pid).state == IndexState.FRESH and manager.metadata(pid).build_id == m.build_id
    d = manager.index_dir(pid)
    assert {p.name for p in d.iterdir()} == {"manifest.json", "faiss.index", "faiss_ids.npy", "bm25.json"}


def test_chunk_to_vector_mapping_matches_database(db, make_project, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    index = manager.load(pid)
    expected = db_chunk_ids(db, pid)
    assert set(index.faiss.chunk_ids.tolist()) == expected and set(index.bm25.chunk_ids) == expected
    assert len(index.faiss) == len(expected) == len(index.bm25)


def test_build_is_a_noop_when_fresh_and_force_rebuilds(make_project, retrieval_env):
    pid = make_project()
    manager, _, emb, _ = retrieval_env()
    first = manager.build(pid)
    embedded = emb.documents_embedded
    assert manager.build(pid).build_id == first.build_id and emb.documents_embedded == embedded
    again = manager.rebuild(pid)
    assert again.build_id != first.build_id and emb.documents_embedded == 2 * embedded


def test_staleness_when_sources_change(db, make_project, make_service, retrieval_env):
    pid = make_project()
    manager, _, emb, _ = retrieval_env()
    manager.build(pid)
    svc = make_service()

    svc.ingest_bytes(pid, "new_module.py", b"def brand_new(): ...\n")
    st = manager.status(pid)
    assert st.state == IndexState.STALE and any("changed" in r for r in st.reasons)
    with pytest.raises(StaleIndexError):
        manager.load(pid)
    assert manager.load(pid, allow_stale=True) is not None
    manager.ensure(pid)
    assert manager.status(pid).is_fresh

    svc.ingest_bytes(pid, "users.py", b"def create_user(name): ...\n")        # existing file, new content
    assert manager.status(pid).state == IndexState.STALE
    manager.ensure(pid)
    with db.session() as s:
        src = s.sources.get_by_name(pid, "cache.py")
    svc.remove_source(src.id)                                                  # removed file
    assert manager.status(pid).state == IndexState.STALE
    manager.ensure(pid)
    assert manager.status(pid).is_fresh and emb.documents_embedded > 0


def test_unchanged_reingest_does_not_make_the_index_stale(make_project, make_service, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    make_service().ingest_bytes(pid, "users.py", corpus_files()["users.py"])   # identical -> UNCHANGED
    assert manager.status(pid).is_fresh


def test_embedding_model_change_marks_stale(make_project, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    other, *_ = retrieval_env(embedder=CountingEmbedder(model_name="another-model"))
    st = other.status(pid)
    assert st.state == IndexState.STALE and any("embedding model changed" in r for r in st.reasons)


def test_reload_from_disk_does_not_reembed_and_is_cached(db, settings, make_project, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    emb2 = CountingEmbedder()
    fresh_manager = IndexManager(db, settings, emb2)           # e.g. after an application restart
    loaded = fresh_manager.load(pid)
    assert emb2.documents_embedded == 0 and loaded.manifest.chunk_count == len(loaded.faiss)
    assert fresh_manager.load(pid) is loaded                   # cached: no second disk load
    q = emb2.embed_query("calculate invoice total")
    assert loaded.faiss.search(q, 3) and loaded.bm25.search("calculate_invoice_total", 3)


def test_indexes_are_project_scoped(db, make_project, retrieval_env):
    a = make_project()
    b_files = {f"b_{i}.py": f"def b_func_{i}(): ...\n".encode() for i in range(8)}      # >= 8 chunks: BM25 IDF needs a real corpus
    b_files["other.py"] = b"def zxqunique(): ...\n"
    b = make_project(files=b_files)
    manager, *_ = retrieval_env()
    manager.build(a)
    manager.build(b)
    ia, ib = manager.load(a), manager.load(b)
    ids_a, ids_b = db_chunk_ids(db, a), db_chunk_ids(db, b)
    assert not ids_a & ids_b
    assert set(ia.faiss.chunk_ids.tolist()) == ids_a and set(ib.faiss.chunk_ids.tolist()) == ids_b
    assert set(ia.bm25.chunk_ids) == ids_a and set(ib.bm25.chunk_ids) == ids_b
    assert manager.index_dir(a) != manager.index_dir(b)
    assert not {h.chunk_id for h in ia.bm25.search("zxqunique", 5)}      # b's terms absent from a's index
    assert {h.chunk_id for h in ib.bm25.search("zxqunique", 5)}           # ...but present in b's
    assert manager.status(a).is_fresh and manager.delete(b) and manager.status(a).is_fresh


def test_corrupt_manifest_and_corrupt_files(make_project, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    (manager.index_dir(pid) / "manifest.json").write_text("{broken")
    assert manager.status(pid).state == IndexState.CORRUPT
    with pytest.raises(IndexCorruptError):
        manager.load(pid)
    assert manager.ensure(pid).manifest.chunk_count > 0        # ensure() repairs by rebuilding

    manager2, *_ = retrieval_env()
    (manager.index_dir(pid) / "faiss_ids.npy").write_bytes(b"xx")   # manifest fine, data broken
    with pytest.raises(IndexCorruptError):
        manager2.load(pid)


def test_empty_project_builds_an_empty_index(db, retrieval_env):
    with db.session() as s:
        pid = s.projects.create(Project(name="empty")).id
    manager, *_ = retrieval_env()
    m = manager.build(pid)
    assert m.chunk_count == 0 and manager.status(pid).is_fresh
    idx = manager.load(pid)
    assert len(idx.faiss) == 0 and idx.bm25.search("x", 3) == []


def test_unknown_project_is_rejected(retrieval_env):
    manager, *_ = retrieval_env()
    with pytest.raises(NotFoundError):
        manager.build(424242)


def test_failed_rebuild_keeps_previous_index_and_leaves_no_temp_dirs(monkeypatch, make_project, make_service, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    old = manager.build(pid)
    make_service().ingest_bytes(pid, "late.py", b"def late(): ...\n")

    from trustdoc.retrieval.bm25_index import Bm25Index

    def boom(self, directory):
        raise OSError("disk full")

    monkeypatch.setattr(Bm25Index, "save", boom)
    with pytest.raises(OSError):
        manager.build(pid)
    monkeypatch.undo()
    assert json.loads((manager.index_dir(pid) / "manifest.json").read_text())["build_id"] == old.build_id
    assert not [p for p in manager.index_dir(pid).parent.iterdir() if p.name.startswith((".building", ".old"))]


def test_delete_removes_index(make_project, retrieval_env):
    pid = make_project()
    manager, *_ = retrieval_env()
    manager.build(pid)
    assert manager.delete(pid) is True and manager.delete(pid) is False
    assert manager.status(pid).state == IndexState.MISSING
