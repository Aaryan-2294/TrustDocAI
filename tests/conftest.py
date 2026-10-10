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


# ----------------------------------------------------------- Part 3 fixtures
import sys  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))


@pytest.fixture
def retrieval_env(db, settings):
    """Factory building (IndexManager, Retriever, embedder, reranker) on the shared test DB."""
    pytest.importorskip("faiss")
    pytest.importorskip("rank_bm25")
    from retrieval_support import CountingEmbedder, RecordingReranker
    from trustdoc.retrieval import IndexManager, Retriever

    def build(*, embedder=None, reranker=None, auto_index=True, settings_override=None):
        cfg = settings_override or settings
        embedder = embedder or CountingEmbedder()
        reranker = reranker if reranker is not None else RecordingReranker()
        manager = IndexManager(db, cfg, embedder)
        retriever = Retriever(db, cfg, manager, embedder, reranker, auto_index=auto_index)
        return manager, retriever, embedder, reranker

    return build


@pytest.fixture
def make_project(db, make_service):
    """Create a project and ingest the shared test corpus into it."""
    from corpus_support import corpus_files

    counter = {"n": 0}

    def create(files=None, name=None):
        counter["n"] += 1
        with db.session() as s:
            project = s.projects.create(Project(name=name or f"proj{counter['n']}"))
        svc = make_service()
        for fname, data in (files or corpus_files()).items():
            svc.ingest_bytes(project.id, fname, data)
        return project.id

    return create
