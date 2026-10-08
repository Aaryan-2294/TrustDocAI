import hashlib
import os
import sys

import pytest

from trustdoc.exceptions import (
    FileTooLargeError,
    InvalidSourceNameError,
    NotFoundError,
    ParsingError,
    UnsupportedSourceError,
)
from trustdoc.ingestion import IngestionStatus
from trustdoc.models import ChunkType, EntityType, Project, SourceType

PY_A = '''"""Pkg A."""
import os


def load(path: str, strict: bool = False) -> dict:
    """Load a file."""
    return {}


class Store:
    """Stores things."""

    def put(self, key, value):
        return True
'''
README = "# Demo\nIntro.\n\n## Installation\nRun pip install demo.\n\n## Usage\nCall load().\n"


@pytest.fixture
def pid2(db):
    with db.session() as s:
        return s.projects.create(Project(name="other")).id


def stored_files(service, project_id):
    d = service.store.root / f"project_{project_id}"
    return sorted(p.name for p in d.glob("*")) if d.exists() else []


# ------------------------------------------------------------ storage + hashing
def test_ingest_bytes_registers_source_with_sha256_size_and_stored_copy(service, db, project_id):
    data = PY_A.encode()
    res = service.ingest_bytes(project_id, "pkg/a.py", data)
    assert res.status == IngestionStatus.INGESTED
    src = res.source
    assert src.content_hash == hashlib.sha256(data).hexdigest() and src.size_bytes == len(data)
    assert src.name == "pkg/a.py" and src.source_type == SourceType.PYTHON
    assert src.stored_path == f"project_{project_id}/{src.content_hash}.py"
    assert not os.path.isabs(src.stored_path)
    assert service.store.read(src.stored_path) == data  # original preserved byte-for-byte
    assert src.metadata["module_name"] == "pkg.a" and src.metadata["entity_counts"]["function"] == 1
    with db.session() as s:
        assert s.sources.get(src.id).metadata == src.metadata


def test_names_are_sanitised_and_unsafe_names_rejected_before_any_write(service, project_id):
    assert service.ingest_bytes(project_id, "a\\b\\x.py", b"x = 1\n").source.name == "a/b/x.py"
    with pytest.raises(InvalidSourceNameError):
        service.ingest_bytes(project_id, "../escape.py", b"x = 1\n")
    with pytest.raises(InvalidSourceNameError):
        service.ingest_bytes(project_id, "/abs/path.py", b"x = 1\n")
    assert len(stored_files(service, project_id)) == 1


def test_ingest_file_uses_basename_and_never_leaks_the_filesystem_path(service, tmp_path, project_id):
    f = tmp_path / "deep" / "dir" / "util.py"
    f.parent.mkdir(parents=True)
    f.write_text("def f(): ...\n")
    res = service.ingest_file(project_id, f)
    assert res.source.name == "util.py"
    assert str(tmp_path) not in res.source.name + res.source.stored_path
    assert service.ingest_file(project_id, f, name="lib/util.py").source.name == "lib/util.py"


def test_unsupported_oversized_and_unknown_project(service, make_service, project_id):
    with pytest.raises(UnsupportedSourceError):
        service.ingest_bytes(project_id, "notes.txt", b"hi")
    small = make_service()
    small._max_bytes = 10
    with pytest.raises(FileTooLargeError):
        small.ingest_bytes(project_id, "big.py", b"x" * 11)
    with pytest.raises(NotFoundError):
        service.ingest_bytes(9999, "a.py", b"x = 1\n")
    assert stored_files(service, 9999) == []


def test_explicit_source_type_override(service, project_id):
    res = service.ingest_bytes(project_id, "README", b"# T\nbody\n", source_type=SourceType.MARKDOWN)
    assert res.source.source_type == SourceType.MARKDOWN and res.chunk_count == 1


# ------------------------------------------------------------------ duplicates
def test_reingesting_identical_content_is_unchanged(service, db, project_id):
    first = service.ingest_bytes(project_id, "a.py", PY_A.encode())
    again = service.ingest_bytes(project_id, "a.py", PY_A.encode())
    assert again.status == IngestionStatus.UNCHANGED and again.source.id == first.source.id
    assert (again.entity_count, again.chunk_count) == (first.entity_count, first.chunk_count)
    with db.session() as s:
        assert len(s.sources.list_for_project(project_id)) == 1
        assert s.chunks.count_for_project(project_id) == first.chunk_count  # not duplicated


def test_force_reingests_same_content(service, project_id):
    first = service.ingest_bytes(project_id, "a.py", PY_A.encode())
    forced = service.ingest_bytes(project_id, "a.py", PY_A.encode(), force=True)
    assert forced.status == IngestionStatus.UPDATED and forced.source.id != first.source.id
    assert len(stored_files(service, project_id)) == 1  # shared file kept


def test_changed_content_replaces_source_entities_chunks_and_old_file(service, db, project_id):
    old = service.ingest_bytes(project_id, "a.py", b"def old(): ...\n")
    new = service.ingest_bytes(project_id, "a.py", b"def new_one(): ...\ndef another(): ...\n")
    assert new.status == IngestionStatus.UPDATED and new.replaced_source_id == old.source.id
    with db.session() as s:
        assert s.sources.get(old.source.id) is None
        names = {e.name for e in s.entities.list_for_project(project_id)}
        assert names == {"a", "new_one", "another"}
        assert {c.source_id for c in s.chunks.list_for_project(project_id)} == {new.source.id}
    assert stored_files(service, project_id) == [os.path.basename(new.source.stored_path)]


def test_same_content_under_two_names_shares_file_and_warns(service, project_id):
    a = service.ingest_bytes(project_id, "a.py", b"x = 1\n")
    b = service.ingest_bytes(project_id, "copy/b.py", b"x = 1\n")
    assert a.source.stored_path == b.source.stored_path and len(stored_files(service, project_id)) == 1
    assert any("identical content" in w for w in b.warnings)
    assert b.source.metadata["duplicate_of"] == "a.py"
    service.remove_source(a.source.id)
    assert service.store.read(b.source.stored_path) == b"x = 1\n"  # still referenced
    service.remove_source(b.source.id)
    assert stored_files(service, project_id) == []


def test_project_isolation(service, db, project_id, pid2):
    a = service.ingest_bytes(project_id, "a.py", b"def f(): ...\n")
    b = service.ingest_bytes(pid2, "a.py", b"def f(): ...\n")  # same name and content, other project
    assert a.status == b.status == IngestionStatus.INGESTED and a.source.id != b.source.id
    assert a.source.stored_path != b.source.stored_path
    assert a.source.stored_path.startswith(f"project_{project_id}/")
    assert b.source.stored_path.startswith(f"project_{pid2}/")
    with db.session() as s:
        assert {e.project_id for e in s.entities.list_for_project(project_id)} == {project_id}
        assert {c.project_id for c in s.chunks.list_for_project(pid2)} == {pid2}
    service.ingest_bytes(pid2, "a.py", b"def g(): ...\n")  # updating one project...
    with db.session() as s:
        assert "f" in {e.name for e in s.entities.list_for_project(project_id)}  # ...leaves the other alone


# ---------------------------------------------------------------------- atomicity
def test_malformed_python_leaves_no_partial_state(service, db, project_id):
    with pytest.raises(ParsingError):
        service.ingest_bytes(project_id, "bad.py", b"def broken(:\n")
    with db.session() as s:
        assert s.sources.list_for_project(project_id) == []
        assert sum(db.table_counts().values()) == 1  # just the project row
    assert stored_files(service, project_id) == []


def test_failed_update_keeps_previous_version(service, db, project_id):
    good = service.ingest_bytes(project_id, "a.py", b"def ok(): ...\n")
    with pytest.raises(ParsingError):
        service.ingest_bytes(project_id, "a.py", b"def nope(:\n")
    with db.session() as s:
        assert s.sources.get_by_name(project_id, "a.py").id == good.source.id
        assert {e.name for e in s.entities.list_for_project(project_id)} == {"a", "ok"}
    assert service.store.read(good.source.stored_path) == b"def ok(): ...\n"


def test_database_failure_during_persist_rolls_back_and_removes_new_file(service, db, project_id, monkeypatch):
    from trustdoc.db.repositories import ChunkRepository

    def boom(self, chunks):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ChunkRepository, "add_many", boom)
    with pytest.raises(RuntimeError):
        service.ingest_bytes(project_id, "a.py", b"def f(): ...\n")
    with db.session() as s:
        assert s.sources.list_for_project(project_id) == [] and s.entities.list_for_project(project_id) == []
    assert stored_files(service, project_id) == []


# ------------------------------------------------------------- per-type results
def test_markdown_ingestion_stores_heading_provenance(service, db, project_id):
    res = service.ingest_bytes(project_id, "README.md", README.encode())
    assert res.source.metadata["heading_count"] == 3 and res.source.metadata["title"] == "Demo"
    with db.session() as s:
        chunks = s.chunks.list_for_source(res.source.id)
    assert [c.heading for c in chunks] == ["Demo", "Demo > Installation", "Demo > Usage"]
    inst = chunks[1]
    assert (inst.start_line, inst.end_line, inst.chunk_type) == (4, 5, ChunkType.MARKDOWN)
    assert inst.location_label("README.md") == "README.md:4-5 \u00a7 Demo > Installation"


def test_pdf_ingestion_and_warnings(service, make_pdf, db, project_id):
    res = service.ingest_bytes(project_id, "docs/ref.pdf", make_pdf(["Alpha page", None, "Gamma page"]))
    assert res.source.source_type == SourceType.PDF and res.source.metadata["page_count"] == 3
    assert res.source.metadata["empty_pages"] == [2] and any("[2]" in w for w in res.warnings)
    with db.session() as s:
        assert [c.page_number for c in s.chunks.list_for_source(res.source.id)] == [1, 3]
    blank = service.ingest_bytes(project_id, "scan.pdf", make_pdf([None, None]))
    assert blank.chunk_count == 0 and blank.source.metadata["no_extractable_text"] is True
    assert any("OCR" in w for w in blank.warnings)


def test_corrupt_pdf_is_rejected_cleanly(service, db, project_id):
    pytest.importorskip("pymupdf")
    with pytest.raises(ParsingError):
        service.ingest_bytes(project_id, "x.pdf", b"%PDF-1.7 garbage")
    assert db.table_counts()["sources"] == 0 and stored_files(service, project_id) == []


# ------------------------------------------------------ sample project integration
@pytest.fixture
def sample_project(tmp_path, make_pdf):
    root = tmp_path / "sample"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text('"""Package."""\n')
    (root / "pkg" / "core.py").write_text(PY_A)
    (root / "README.md").write_text(README)
    (root / "docs").mkdir()
    (root / "docs" / "spec.pdf").write_bytes(make_pdf(["Spec page one.", "Spec page two."]))
    (root / "notes.txt").write_text("ignored")
    (root / "pkg" / "broken.py").write_text("def x(:\n")
    (root / ".git").mkdir()
    (root / ".git" / "hook.py").write_text("x = 1\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.py").write_text("x = 1\n")
    if hasattr(os, "symlink"):
        try:
            os.symlink(root / "README.md", root / "link.md")
        except OSError:
            pass
    return root


def test_ingest_sample_project_end_to_end(service, db, project_id, sample_project):
    batch = service.ingest_directory(project_id, sample_project)
    names = sorted(r.source.name for r in batch.results)
    assert names == ["README.md", "docs/spec.pdf", "pkg/__init__.py", "pkg/core.py"]
    assert [f.name for f in batch.failures] == ["pkg/broken.py"] and not batch.ok
    assert "line 1" in batch.failures[0].error and batch.skipped_unsupported == 1

    with db.session() as s:
        assert {x.name for x in s.sources.list_for_project(project_id)} == set(names)
        by_name = {x.name: x for x in s.sources.list_for_project(project_id)}
        core = by_name["pkg/core.py"]
        entities = {e.qualified_name: e for e in s.entities.list_for_source(core.id)}
        assert set(entities) == {"pkg.core", "pkg.core.load", "pkg.core.Store", "pkg.core.Store.put"}
        assert entities["pkg.core.Store.put"].parent_id == entities["pkg.core.Store"].id
        assert entities["pkg.core.Store"].parent_id == entities["pkg.core"].id
        assert entities["pkg.core"].parent_id is None

        # provenance: every code chunk points back to a real entity and to the exact stored source lines
        stored_lines = service.store.read(core.stored_path).decode().split("\n")
        chunks = s.chunks.list_for_source(core.id)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        for c in chunks:
            ent = s.entities.get(c.entity_id)
            assert ent.source_id == core.id and ent.qualified_name == c.metadata["qualified_name"]
            assert ent.start_line <= c.start_line <= c.end_line <= ent.end_line
            assert c.text == "\n".join(stored_lines[c.start_line - 1 : c.end_line])
            assert c.project_id == project_id and c.location_label(core.name).startswith("pkg/core.py:")
        load_chunk = next(c for c in chunks if c.metadata["qualified_name"] == "pkg.core.load")
        assert load_chunk.text.startswith("def load(path: str") and load_chunk.entity_id == entities["pkg.core.load"].id

        pdf_chunks = s.chunks.list_for_source(by_name["docs/spec.pdf"].id)
        assert [c.page_number for c in pdf_chunks] == [1, 2]
        assert pdf_chunks[1].location_label("docs/spec.pdf") == "docs/spec.pdf, p.2"
        md_chunks = s.chunks.list_for_source(by_name["README.md"].id)
        assert "Demo > Installation" in [c.heading for c in md_chunks]

    # second run: nothing changes, only the broken file fails again
    again = service.ingest_directory(project_id, sample_project)
    assert {r.status for r in again.results} == {IngestionStatus.UNCHANGED}
    assert len(again.failures) == 1


def test_symlinks_and_ignored_dirs_are_not_followed(service, project_id, sample_project):
    names = {r.source.name for r in service.ingest_directory(project_id, sample_project).results}
    assert "link.md" not in names and not any(n.startswith((".git", "__pycache__")) for n in names)


def test_remove_source_cascades_and_deletes_file(service, db, project_id):
    res = service.ingest_bytes(project_id, "a.py", PY_A.encode())
    assert service.remove_source(res.source.id) is True and service.remove_source(res.source.id) is False
    with db.session() as s:
        assert s.entities.list_for_project(project_id) == [] and s.chunks.list_for_project(project_id) == []
    assert stored_files(service, project_id) == []


def test_small_chunk_config_is_honoured_for_all_source_types(make_service, db, project_id, make_pdf):
    svc = make_service(CHUNK_MAX_CHARS=200, CHUNK_OVERLAP_CHARS=30)
    long_py = "def f():\n" + "".join(f"    x{i} = {i}  # filler text here\n" for i in range(40))
    long_md = "# H\n" + " ".join(f"word{i}" for i in range(150))
    long_pdf = make_pdf([" ".join(f"w{i}" for i in range(150))])
    for name, data in (("l.py", long_py), ("l.md", long_md), ("l.pdf", long_pdf)):
        res = svc.ingest_bytes(project_id, name, data if isinstance(data, bytes) else data.encode())
        assert res.chunk_count > 2, name
        with db.session() as s:
            assert all(len(c.text) <= 200 for c in s.chunks.list_for_source(res.source.id)), name
