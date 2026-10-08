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

CREATE TABLE claims (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id      INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_id       INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    claim_type     TEXT NOT NULL {_check("claim_type", ClaimType)},
    origin         TEXT NOT NULL {_check("origin", ClaimOrigin)},
    text           TEXT NOT NULL,
    subject_name   TEXT,
    start_line     INTEGER,
    page_number    INTEGER,
    extracted_json TEXT NOT NULL DEFAULT '{{}}',
    created_at     TEXT NOT NULL
);
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

CREATE TABLE claim_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    claim_id   INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    tier       TEXT NOT NULL {_check("tier", VerificationTier)},
    verdict    TEXT NOT NULL {_check("verdict", ClaimVerdict)},
    confidence REAL,
    rationale  TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (run_id, claim_id)
);

CREATE TABLE evidence_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_result_id INTEGER NOT NULL REFERENCES claim_results(id) ON DELETE CASCADE,
    chunk_id        INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    stance          TEXT NOT NULL {_check("stance", EvidenceStance)},
    relevance_score REAL,
    rank            INTEGER
);
CREATE INDEX idx_evidence_result ON evidence_links(claim_result_id);

CREATE TABLE findings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    tier         TEXT NOT NULL {_check("tier", VerificationTier)},
    issue_type   TEXT NOT NULL {_check("issue_type", IssueType)},
    severity     TEXT NOT NULL {_check("severity", Severity)},
    message      TEXT NOT NULL,
    suggestion   TEXT NOT NULL DEFAULT '',
    source_id    INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    entity_id    INTEGER REFERENCES code_entities(id) ON DELETE SET NULL,
    claim_id     INTEGER REFERENCES claims(id) ON DELETE SET NULL,
    start_line   INTEGER,
    details_json TEXT NOT NULL DEFAULT '{{}}',
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_findings_run ON findings(run_id);
"""

# version -> DDL that upgrades the database *to* that version.
MIGRATIONS: dict[int, str] = {1: SCHEMA_V1}
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
