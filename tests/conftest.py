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
