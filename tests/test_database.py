import sqlite3

import pytest

from trustdoc.db import SCHEMA_VERSION, TABLE_NAMES, Database
from trustdoc.exceptions import DatabaseError


def test_initialize_creates_all_tables(db):
    assert db.schema_version() == SCHEMA_VERSION
    assert set(db.table_counts()) == set(TABLE_NAMES)


def test_initialize_is_idempotent(db, project_id):
    assert db.initialize() == SCHEMA_VERSION
    assert db.table_counts()["projects"] == 1  # data survived


def test_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(DatabaseError):
        Database(path).initialize()


def test_foreign_keys_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.session() as s:
            s.conn.execute(
                "INSERT INTO sources (project_id, source_type, name, content_hash, created_at) "
                "VALUES (999, 'python', 'x.py', 'h', 'now')"
            )


def test_check_constraint_rejects_unknown_enum_value(db, project_id):
    with pytest.raises(sqlite3.IntegrityError):
        with db.session() as s:
            s.conn.execute(
                "INSERT INTO sources (project_id, source_type, name, content_hash, created_at) "
                "VALUES (?, 'docx', 'x.docx', 'h', 'now')", (project_id,)
            )


def test_session_rolls_back_on_error(db):
    from trustdoc.models import Project
    with pytest.raises(RuntimeError):
        with db.session() as s:
            s.projects.create(Project(name="temp"))
            raise RuntimeError("boom")
    assert db.table_counts()["projects"] == 0
