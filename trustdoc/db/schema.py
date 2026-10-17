"""SQLite schema (versioned via ``PRAGMA user_version``).

CHECK constraints are generated from the enums in ``trustdoc.models.enums`` so
the database and the Python code share one vocabulary.
"""

from __future__ import annotations

from enum import Enum

from trustdoc.models.enums import (
    ChunkType,
    ClaimOrigin,
    ClaimType,
    ClaimVerdict,
    EntityType,
    EvidenceStance,
    IssueType,
    RunStatus,
    Severity,
    SourceType,
    VerificationTier,
)


def _check(column: str, enum_cls: type[Enum], nullable: bool = False) -> str:
    values = ", ".join(f"'{m.value}'" for m in enum_cls)
    clause = f"{column} IN ({values})"
    return f"CHECK ({column} IS NULL OR {clause})" if nullable else f"CHECK ({clause})"


# --- Frozen vocabularies of schema v1. Never edit these: they define what an existing v1 database
# --- contains, and the v2 migration needs them to stay exactly as they were.
_V1_CLAIM_TYPES = ("entity_reference", "parameter", "return", "behavioral")
_V1_VERDICTS = ("supported", "contradicted", "insufficient_evidence")
_V1_SEVERITIES = ("low", "medium", "high", "critical")


def _check_values(column: str, values) -> str:
    listed = ", ".join(f"'{v}'" for v in values)
    return f"CHECK ({column} IN ({listed}))"


def _enum_values(enum_cls: type[Enum]) -> tuple[str, ...]:
    return tuple(m.value for m in enum_cls)


# The three tables whose CHECK constraints changed in v2, parameterised by their vocabularies so the
# same DDL text builds both the v1 tables and the v2 replacements.
def _claims_ddl(name: str, claim_types) -> str:
    return f"""CREATE TABLE {name} (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id      INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_id       INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    claim_type     TEXT NOT NULL {_check_values("claim_type", claim_types)},
    origin         TEXT NOT NULL {_check("origin", ClaimOrigin)},
    text           TEXT NOT NULL,
    subject_name   TEXT,
    start_line     INTEGER,
    page_number    INTEGER,
    extracted_json TEXT NOT NULL DEFAULT '{{}}',
    created_at     TEXT NOT NULL
)"""


def _claim_results_ddl(name: str, verdicts) -> str:
    return f"""CREATE TABLE {name} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    claim_id   INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    tier       TEXT NOT NULL {_check("tier", VerificationTier)},
    verdict    TEXT NOT NULL {_check_values("verdict", verdicts)},
    confidence REAL,
    rationale  TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (run_id, claim_id)
)"""


def _findings_ddl(name: str, severities) -> str:
    return f"""CREATE TABLE {name} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    tier         TEXT NOT NULL {_check("tier", VerificationTier)},
    issue_type   TEXT NOT NULL {_check("issue_type", IssueType)},
    severity     TEXT NOT NULL {_check_values("severity", severities)},
    message      TEXT NOT NULL,
    suggestion   TEXT NOT NULL DEFAULT '',
    source_id    INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    entity_id    INTEGER REFERENCES code_entities(id) ON DELETE SET NULL,
    claim_id     INTEGER REFERENCES claims(id) ON DELETE SET NULL,
    start_line   INTEGER,
    details_json TEXT NOT NULL DEFAULT '{{}}',
    created_at   TEXT NOT NULL
)"""


SCHEMA_V1 = f"""
CREATE TABLE projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE sources (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_type   TEXT NOT NULL {_check("source_type", SourceType)},
    name          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    stored_path   TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{{}}',
    created_at    TEXT NOT NULL,
    UNIQUE (project_id, name)
);
CREATE INDEX idx_sources_project ON sources(project_id);

CREATE TABLE code_entities (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id         INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    parent_id         INTEGER REFERENCES code_entities(id) ON DELETE CASCADE,
    entity_type       TEXT NOT NULL {_check("entity_type", EntityType)},
    name              TEXT NOT NULL,
    qualified_name    TEXT NOT NULL,
    parameters_json   TEXT NOT NULL DEFAULT '[]',
    return_annotation TEXT,
    docstring         TEXT,
    decorators_json   TEXT NOT NULL DEFAULT '[]',
    is_async          INTEGER NOT NULL DEFAULT 0,
    is_public         INTEGER NOT NULL DEFAULT 1,
    start_line        INTEGER,
    end_line          INTEGER,
    created_at        TEXT NOT NULL
);
CREATE INDEX idx_entities_project_qname ON code_entities(project_id, qualified_name);
CREATE INDEX idx_entities_source ON code_entities(source_id);

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    entity_id     INTEGER REFERENCES code_entities(id) ON DELETE SET NULL,
    chunk_index   INTEGER NOT NULL,
    chunk_type    TEXT NOT NULL {_check("chunk_type", ChunkType)},
    text          TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    start_line    INTEGER,
    end_line      INTEGER,
    page_number   INTEGER,
    heading       TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{{}}',
    created_at    TEXT NOT NULL,
    UNIQUE (source_id, chunk_index)
);
CREATE INDEX idx_chunks_project ON chunks(project_id);

{_claims_ddl("claims", _V1_CLAIM_TYPES)};
CREATE INDEX idx_claims_project ON claims(project_id);
CREATE INDEX idx_claims_source ON claims(source_id);

CREATE TABLE verification_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    status          TEXT NOT NULL {_check("status", RunStatus)},
    trust_score     REAL,
    config_json     TEXT NOT NULL DEFAULT '{{}}',
    summary_json    TEXT NOT NULL DEFAULT '{{}}',
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);
CREATE INDEX idx_runs_project ON verification_runs(project_id);

CREATE TABLE trust_score_components (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    name         TEXT NOT NULL,
    raw_score    REAL NOT NULL,
    weight       REAL NOT NULL,
    contribution REAL NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    UNIQUE (run_id, name)
);

{_claim_results_ddl("claim_results", _V1_VERDICTS)};

CREATE TABLE evidence_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_result_id INTEGER NOT NULL REFERENCES claim_results(id) ON DELETE CASCADE,
    chunk_id        INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    stance          TEXT NOT NULL {_check("stance", EvidenceStance)},
    relevance_score REAL,
    rank            INTEGER
);
CREATE INDEX idx_evidence_result ON evidence_links(claim_result_id);

{_findings_ddl("findings", _V1_SEVERITIES)};
CREATE INDEX idx_findings_run ON findings(run_id);
"""


def _copy_sequence(old: str, new: str) -> list[str]:
    """Keep AUTOINCREMENT ids moving forward after a table rebuild (never reuse deleted ids)."""
    return [
        f"INSERT INTO sqlite_sequence(name, seq) SELECT '{new}', seq FROM sqlite_sequence "
        f"WHERE name = '{old}' AND NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = '{new}')",
        f"UPDATE sqlite_sequence SET seq = (SELECT seq FROM sqlite_sequence WHERE name = '{old}') "
        f"WHERE name = '{new}' AND EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = '{old}')",
    ]


def _rebuild(table: str, ddl: str, columns: str, indexes: list[str]) -> list[str]:
    """SQLite's documented 12-step table rebuild (create new, copy, drop old, rename new).

    Dropping the old table first and renaming the NEW one keeps the REFERENCES clauses of other
    tables pointing at the right name. Requires PRAGMA foreign_keys=OFF (set by the migration).
    """
    new = f"{table}__v2"
    return [
        ddl.replace(f"CREATE TABLE {table} ", f"CREATE TABLE {new} ", 1),
        f"INSERT INTO {new} ({columns}) SELECT {columns} FROM {table}",
        *_copy_sequence(table, new),
        f"DROP TABLE {table}",
        f"ALTER TABLE {new} RENAME TO {table}",
        *indexes,
    ]


def v2_statements() -> list[str]:
    """Schema v2: widen the CHECK vocabularies of claims / claim_results / findings (Part 5)."""
    claim_types, verdicts = _enum_values(ClaimType), _enum_values(ClaimVerdict)
    severities = _enum_values(Severity)
    return [
        *_rebuild(
            "claims", _claims_ddl("claims", claim_types),
            "id, project_id, source_id, chunk_id, claim_type, origin, text, subject_name, "
            "start_line, page_number, extracted_json, created_at",
            ["CREATE INDEX idx_claims_project ON claims(project_id)",
             "CREATE INDEX idx_claims_source ON claims(source_id)"],
        ),
        *_rebuild(
            "claim_results", _claim_results_ddl("claim_results", verdicts),
            "id, run_id, claim_id, tier, verdict, confidence, rationale, created_at", [],
        ),
        *_rebuild(
            "findings", _findings_ddl("findings", severities),
            "id, run_id, project_id, tier, issue_type, severity, message, suggestion, source_id, "
            "entity_id, claim_id, start_line, details_json, created_at",
            ["CREATE INDEX idx_findings_run ON findings(run_id)"],
        ),
    ]


def migrate_to_v2(conn) -> None:
    """Apply schema v2 atomically. Data, ids and relationships are preserved.

    Runs with foreign keys OFF (required by the rebuild procedure), inside one explicit
    transaction, and verifies ``PRAGMA foreign_key_check`` before committing. On any failure the
    transaction is rolled back and the database is left exactly as it was.
    """
    from trustdoc.exceptions import DatabaseError

    previous = conn.isolation_level
    conn.isolation_level = None            # autocommit mode: we control BEGIN/COMMIT explicitly
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            for statement in v2_statements():
                conn.execute(statement)
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise DatabaseError(f"Migration to schema v2 would break foreign keys: {violations[:3]}")
            conn.execute("PRAGMA user_version = 2")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.isolation_level = previous


# version -> DDL script (str) or a function(conn) that upgrades the database *to* that version.
MIGRATIONS: dict = {1: SCHEMA_V1, 2: migrate_to_v2}
SCHEMA_VERSION = max(MIGRATIONS)

TABLE_NAMES = (
    "projects",
    "sources",
    "code_entities",
    "chunks",
    "claims",
    "verification_runs",
    "trust_score_components",
    "claim_results",
    "evidence_links",
    "findings",
)
