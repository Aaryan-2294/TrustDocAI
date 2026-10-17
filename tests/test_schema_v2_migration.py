"""Schema v2 (Part 5): backward-compatible migration of EXISTING v1 databases."""

import sqlite3

import pytest

from trustdoc.db import SCHEMA_VERSION, Database
from trustdoc.db import schema as schema_module
from trustdoc.db.schema import MIGRATIONS, SCHEMA_V1
from trustdoc.exceptions import DatabaseError
from trustdoc.models import ClaimType, ClaimVerdict, Severity

NOW = "2026-01-01T00:00:00+00:00"


def build_v1_database(path, *, orphan_finding=False):
    """Create a genuine v1 database (the Part 1-4 schema) containing rows in every affected table."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_V1)
    conn.execute("PRAGMA user_version = 1")
    conn.execute("PRAGMA foreign_keys = ON")
    q = conn.execute
    q("INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (1,'p','',?,?)", (NOW, NOW))
    q("INSERT INTO sources (id, project_id, source_type, name, content_hash, created_at) VALUES (1,1,'markdown','README.md','h',?)", (NOW,))
    q("INSERT INTO sources (id, project_id, source_type, name, content_hash, created_at) VALUES (2,1,'python','a.py','h2',?)", (NOW,))
    q("INSERT INTO code_entities (id, project_id, source_id, entity_type, name, qualified_name, created_at) VALUES (1,1,2,'function','f','a.f',?)", (NOW,))
    q("INSERT INTO chunks (id, project_id, source_id, chunk_index, chunk_type, text, char_count, created_at) VALUES (1,1,1,0,'markdown','x',1,?)", (NOW,))
    for cid, ctype in [(1, "entity_reference"), (2, "parameter"), (3, "behavioral")]:
        q("INSERT INTO claims (id, project_id, source_id, chunk_id, claim_type, origin, text, subject_name, start_line, extracted_json, created_at) "
          "VALUES (?,1,1,1,?,'existing_doc',?, 'f', 3, '{\"k\": 1}', ?)", (cid, ctype, f"claim {cid}", NOW))
    q("INSERT INTO verification_runs (id, project_id, status, created_at) VALUES (1,1,'completed',?)", (NOW,))
    for cid, verdict in [(1, "supported"), (2, "contradicted"), (3, "insufficient_evidence")]:
        q("INSERT INTO claim_results (id, run_id, claim_id, tier, verdict, confidence, rationale, created_at) VALUES (?,1,?,'deterministic',?,0.9,'why',?)",
          (cid, cid, verdict, NOW))
    q("INSERT INTO evidence_links (id, claim_result_id, chunk_id, stance, relevance_score, rank) VALUES (1,2,1,'contradicts',0.5,1)")
    q("INSERT INTO trust_score_components (run_id, position, name, raw_score, weight, contribution) VALUES (1,0,'accuracy',0.5,1.0,50.0)")
    q("INSERT INTO findings (id, run_id, project_id, tier, issue_type, severity, message, source_id, entity_id, claim_id, details_json, created_at) "
      "VALUES (1,1,1,'deterministic','parameter_name_mismatch','high','msg',1,1,2,'{\"a\": 1}',?)", (NOW,))
    q("INSERT INTO findings (id, run_id, project_id, tier, issue_type, severity, message, claim_id, created_at) VALUES (2,1,1,'semantic','unsupported_claim','low','m2',3,?)", (NOW,))
    conn.commit()                       # PRAGMA foreign_keys is ignored inside an open transaction
    if orphan_finding:
        q("PRAGMA foreign_keys = OFF")
        q("INSERT INTO findings (id, run_id, project_id, tier, issue_type, severity, message, created_at) VALUES (3,999,1,'semantic','unsupported_claim','low','orphan',?)", (NOW,))
    conn.commit()
    conn.close()


def rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


@pytest.fixture
def v1_path(tmp_path):
    path = tmp_path / "legacy.db"
    build_v1_database(path)
    return path


def test_v1_vocabularies_are_frozen_and_v2_is_the_current_version():
    assert SCHEMA_VERSION == 2 and sorted(MIGRATIONS) == [1, 2]
    for new_value in ("'info'", "'signature'", "'config_dependency'", "'other'", "'not_checkable'"):
        assert new_value not in SCHEMA_V1


def test_v1_database_really_rejects_the_new_values(v1_path):
    conn = sqlite3.connect(v1_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO findings (run_id, project_id, tier, issue_type, severity, message, created_at) "
                     "VALUES (1,1,'semantic','unsupported_claim','info','m',?)", (NOW,))
    conn.close()


def test_migration_upgrades_to_v2_and_is_idempotent(v1_path):
    db = Database(v1_path)
    assert db.schema_version() == 1
    assert db.initialize() == 2 and db.schema_version() == 2
    assert db.initialize() == 2


def test_all_rows_ids_and_values_survive(v1_path):
    before = {t: rows(v1_path, f"SELECT * FROM {t} ORDER BY id")
              for t in ("claims", "claim_results", "findings")}
    counts_before = Database(v1_path).table_counts()
    Database(v1_path).initialize()
    for table, old in before.items():
        assert rows(v1_path, f"SELECT * FROM {table} ORDER BY id") == old, table
    assert Database(v1_path).table_counts() == counts_before


def test_repositories_read_migrated_data_unchanged(v1_path):
    Database(v1_path).initialize()
    db = Database(v1_path)
    with db.session() as s:
        claims = s.claims.list_for_project(1)
        assert [(c.id, c.claim_type, c.extracted) for c in claims] == [
            (1, ClaimType.ENTITY_REFERENCE, {"k": 1}), (2, ClaimType.PARAMETER, {"k": 1}), (3, ClaimType.BEHAVIORAL, {"k": 1})]
        results = s.findings.list_claim_results(1)
        assert [r.verdict for r in results] == [ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED, ClaimVerdict.INSUFFICIENT_EVIDENCE]
        assert results[1].evidence[0].chunk_id == 1                      # evidence link still attached
        assert [f.severity for f in s.findings.list_for_run(1)] == [Severity.HIGH, Severity.LOW]
        assert s.runs.get_trust_breakdown(1) is None          # run has no final score, as inserted


def test_new_vocabulary_is_accepted_old_and_invalid_values_behave(v1_path):
    Database(v1_path).initialize()
    conn = sqlite3.connect(v1_path)
    conn.execute("PRAGMA foreign_keys = ON")
    for i, ctype in enumerate(["signature", "config_dependency", "other", "return"], start=10):
        conn.execute("INSERT INTO claims (id, project_id, source_id, claim_type, origin, text, created_at) VALUES (?,1,1,?,'existing_doc','t',?)", (i, ctype, NOW))
    conn.execute("INSERT INTO claim_results (run_id, claim_id, tier, verdict, created_at) VALUES (1,10,'deterministic','not_checkable',?)", (NOW,))
    conn.execute("INSERT INTO findings (run_id, project_id, tier, issue_type, severity, message, created_at) VALUES (1,1,'semantic','unsupported_claim','info','m',?)", (NOW,))
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO claims (project_id, source_id, claim_type, origin, text, created_at) VALUES (1,1,'bogus','existing_doc','t',?)", (NOW,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO findings (run_id, project_id, tier, issue_type, severity, message, created_at) VALUES (1,1,'semantic','unsupported_claim','catastrophic','m',?)", (NOW,))
    conn.close()


def test_foreign_keys_indexes_and_cascades_still_work(v1_path):
    Database(v1_path).initialize()
    assert rows(v1_path, "PRAGMA foreign_key_check") == []
    names = {r[0] for r in rows(v1_path, "SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_claims_project", "idx_claims_source", "idx_findings_run"} <= names
    refs = rows(v1_path, "SELECT sql FROM sqlite_master WHERE name='evidence_links'")[0][0]
    assert "claim_results(id)" in refs and "__v2" not in refs            # references were not rewritten to a temp name

    conn = sqlite3.connect(v1_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM claims WHERE id = 2")                      # claims -> claim_results -> evidence_links
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM claim_results WHERE claim_id = 2").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM evidence_links").fetchone()[0] == 0
    assert conn.execute("SELECT claim_id FROM findings WHERE id = 1").fetchone()[0] is None   # history kept, link nulled
    conn.execute("DELETE FROM verification_runs WHERE id = 1")           # run -> results/findings/components
    conn.commit()
    assert conn.execute("SELECT (SELECT COUNT(*) FROM claim_results), (SELECT COUNT(*) FROM findings), "
                        "(SELECT COUNT(*) FROM trust_score_components)").fetchone() == (0, 0, 0)
    conn.close()


def test_autoincrement_ids_never_go_backwards(tmp_path):
    path = tmp_path / "seq.db"
    build_v1_database(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("INSERT INTO claims (id, project_id, source_id, claim_type, origin, text, created_at) VALUES (50,1,1,'parameter','existing_doc','t',?)", (NOW,))
    conn.execute("DELETE FROM claims WHERE id = 50")                     # sequence is now 50 while max(id) is 3
    conn.commit()
    conn.close()
    Database(path).initialize()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.execute("INSERT INTO claims (project_id, source_id, claim_type, origin, text, created_at) VALUES (1,1,'signature','existing_doc','t',?)", (NOW,))
    assert cur.lastrowid == 51
    conn.close()


def test_failed_migration_rolls_back_completely(v1_path, monkeypatch):
    original = schema_module.v2_statements

    def broken():
        return original() + ["INSERT INTO table_that_does_not_exist VALUES (1)"]

    monkeypatch.setattr(schema_module, "v2_statements", broken)
    before = rows(v1_path, "SELECT * FROM findings ORDER BY id")
    db = Database(v1_path)
    with pytest.raises(sqlite3.OperationalError):
        db.initialize()
    assert db.schema_version() == 1                                      # still v1
    assert rows(v1_path, "SELECT * FROM findings ORDER BY id") == before
    assert "claims__v2" not in {r[0] for r in rows(v1_path, "SELECT name FROM sqlite_master")}
    monkeypatch.undo()
    assert Database(v1_path).initialize() == 2                           # and a later retry succeeds


def test_migration_refuses_to_commit_if_foreign_keys_would_break(tmp_path):
    path = tmp_path / "orphan.db"
    build_v1_database(path, orphan_finding=True)
    db = Database(path)
    with pytest.raises(DatabaseError, match="foreign keys"):
        db.initialize()
    assert db.schema_version() == 1 and len(rows(path, "SELECT * FROM findings")) == 3


def test_fresh_databases_are_created_at_v2_with_the_new_vocabulary(tmp_path):
    db = Database(tmp_path / "fresh.db")
    assert db.initialize() == 2
    with db.session() as s:
        from trustdoc.models import Project
        s.projects.create(Project(name="x"))
    ddl = rows(tmp_path / "fresh.db", "SELECT sql FROM sqlite_master WHERE name IN ('claims','claim_results','findings')")
    joined = " ".join(r[0] for r in ddl)
    for value in ("'info'", "'signature'", "'not_checkable'", "'config_dependency'"):
        assert value in joined
