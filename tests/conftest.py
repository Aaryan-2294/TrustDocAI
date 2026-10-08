import pytest

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.models import Project, Source, SourceType


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env({"TRUSTDOC_DATA_DIR": str(tmp_path / "data"), "TRUSTDOC_LOG_TO_FILE": "false"})


@pytest.fixture
def db(settings):
    settings.ensure_directories()
    database = Database(settings.storage.db_path)
    database.initialize()
    return database


@pytest.fixture
def project_id(db):
    with db.session() as s:
        return s.projects.create(Project(name="demo")).id


@pytest.fixture
def py_source_id(db, project_id):
    with db.session() as s:
        return s.sources.add(
            Source(project_id=project_id, source_type=SourceType.PYTHON, name="pkg/a.py",
                   content_hash="abc", size_bytes=10)
        ).id


# ----------------------------------------------------------- Part 2 fixtures
@pytest.fixture
def make_service(tmp_path, db):
    """Factory for an IngestionService sharing the test database."""
    from trustdoc.ingestion import IngestionService

    def factory(**env):
        values = {"TRUSTDOC_DATA_DIR": str(tmp_path / "data"), "TRUSTDOC_LOG_TO_FILE": "false"}
        values.update({f"TRUSTDOC_{k}": str(v) for k, v in env.items()})
        return IngestionService(db, Settings.from_env(values))

    return factory


@pytest.fixture
def service(make_service):
    return make_service()


@pytest.fixture
def make_pdf():
    """Build small text PDFs: items are page text, or None for a blank page."""
    pymupdf = pytest.importorskip("pymupdf")

    def build(pages, **save_kwargs):
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
        data = doc.tobytes(**save_kwargs)
        doc.close()
        return data

    return build
