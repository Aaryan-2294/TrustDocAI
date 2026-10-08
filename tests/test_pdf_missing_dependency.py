import sys

import pytest

from trustdoc.exceptions import MissingDependencyError
from trustdoc.parsing.pdf_reader import read_pdf


@pytest.fixture
def no_pymupdf(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", None)  # makes `import pymupdf` raise ImportError
    monkeypatch.setitem(sys.modules, "fitz", None)


def test_read_pdf_explains_how_to_install(no_pymupdf):
    with pytest.raises(MissingDependencyError, match=r"\.\[pdf\]"):
        read_pdf(b"%PDF-1.4", "x.pdf")


def test_python_and_markdown_ingestion_work_without_pymupdf(no_pymupdf, service, db, project_id):
    assert service.ingest_bytes(project_id, "a.py", b"def f(): ...\n").chunk_count == 1
    assert service.ingest_bytes(project_id, "r.md", b"# T\nbody\n").chunk_count == 1
    with pytest.raises(MissingDependencyError):
        service.ingest_bytes(project_id, "d.pdf", b"%PDF-1.4")
    assert db.table_counts()["sources"] == 2  # failed PDF left nothing behind
