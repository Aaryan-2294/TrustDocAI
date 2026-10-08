import hashlib

import pytest

from trustdoc.exceptions import InvalidSourceNameError, UnsupportedSourceError
from trustdoc.ingestion import UploadStore, detect_source_type, sanitize_source_name
from trustdoc.models import SourceType


@pytest.mark.parametrize("raw, clean", [
    ("README.md", "README.md"),
    ("src\\pkg\\a.py", "src/pkg/a.py"),
    ("./src//pkg/./a.py", "src/pkg/a.py"),
    ("  docs/guide.md  ", "docs/guide.md"),
])
def test_sanitize_normalises(raw, clean):
    assert sanitize_source_name(raw) == clean


@pytest.mark.parametrize("bad", [
    "", "   ", "../secret.py", "a/../../b.py", "..\\..\\evil.py", "/etc/passwd.py",
    "C:\\Windows\\x.py", "C:/x.py", "~/x.py", "a\x00b.py", "a\nb.py", "./", "a/" + "x" * 300 + ".py",
])
def test_sanitize_rejects_unsafe_names(bad):
    with pytest.raises(InvalidSourceNameError):
        sanitize_source_name(bad)


@pytest.mark.parametrize("name, kind", [
    ("a.py", SourceType.PYTHON), ("A.PY", SourceType.PYTHON), ("r.md", SourceType.MARKDOWN),
    ("r.markdown", SourceType.MARKDOWN), ("doc.pdf", SourceType.PDF), ("x/Doc.PDF", SourceType.PDF),
])
def test_detect_source_type(name, kind):
    assert detect_source_type(name) == kind


@pytest.mark.parametrize("name", ["a.txt", "Makefile", "script.js", "image.png"])
def test_unsupported_types(name):
    with pytest.raises(UnsupportedSourceError):
        detect_source_type(name)


def test_store_is_content_addressed_and_idempotent(tmp_path):
    store = UploadStore(tmp_path / "uploads")
    data = b"print('hi')\n"
    digest = hashlib.sha256(data).hexdigest()
    rel, created = store.save(7, digest, SourceType.PYTHON, data)
    assert rel == f"project_7/{digest}.py" and created
    assert store.read(rel) == data
    assert store.save(7, digest, SourceType.PYTHON, data) == (rel, False)
    assert not list((tmp_path / "uploads" / "project_7").glob(".upload-*"))  # no temp leftovers
    assert store.delete(rel) and not store.delete(rel)


def test_store_never_resolves_outside_root(tmp_path):
    store = UploadStore(tmp_path / "uploads")
    for bad in ("../outside.py", "project_1/../../x", "/etc/passwd"):
        with pytest.raises(InvalidSourceNameError):
            store.resolve(bad)
    with pytest.raises(ValueError):
        store.relative_path(1, "../../etc/passwd", SourceType.PYTHON)
