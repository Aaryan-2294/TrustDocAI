"""SQLite connection management and the unit-of-work ``Session``."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from trustdoc.db.repositories import (
    ChunkRepository,
    ClaimRepository,
    EntityRepository,
    FindingRepository,
    ProjectRepository,
    RunRepository,
    SourceRepository,
)
from trustdoc.db.schema import MIGRATIONS, SCHEMA_VERSION, TABLE_NAMES
from trustdoc.exceptions import DatabaseError
from trustdoc.logging_setup import get_logger

log = get_logger("db")


class Session:
    """Repositories bound to one connection / one transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.projects = ProjectRepository(conn)
        self.sources = SourceRepository(conn)
        self.entities = EntityRepository(conn)
        self.chunks = ChunkRepository(conn)
        self.claims = ClaimRepository(conn)
        self.runs = RunRepository(conn)
        self.findings = FindingRepository(conn)


class Database:
    """File-backed SQLite database.

    A fresh connection is opened per ``session()`` (cheap for SQLite and safe
    when Streamlit runs callbacks on different threads).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit on success, roll back on any exception."""
        conn = self.connect()
        try:
            yield Session(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------ schema mgmt
    def schema_version(self) -> int:
        conn = self.connect()
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()

    def initialize(self) -> int:
        """Create or upgrade the schema. Idempotent. Returns the schema version."""
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise DatabaseError(
                f"Database schema v{current} is newer than this application "
                f"(supports up to v{SCHEMA_VERSION})."
            )
        if current == SCHEMA_VERSION:
            return current
        conn = self.connect()
        try:
            for version in sorted(v for v in MIGRATIONS if v > current):
                log.info("Applying schema migration v%d", version)
                migration = MIGRATIONS[version]
                if callable(migration):          # transactional Python migration (v2+)
                    migration(conn)
                else:
                    conn.executescript(migration)
                    conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()
        finally:
            conn.close()
        return SCHEMA_VERSION

    def table_counts(self) -> dict[str, int]:
        conn = self.connect()
        try:
            return {
                t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])  # noqa: S608
                for t in TABLE_NAMES
            }
        finally:
            conn.close()
