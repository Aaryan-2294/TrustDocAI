import json

import pytest
from verify_support import PY, README

from trustdoc.cli import main
from trustdoc.config import Settings
from trustdoc.db import SCHEMA_VERSION, Database
from trustdoc.exceptions import VerificationError
from trustdoc.ingestion import IngestionService
from trustdoc.models import Project

CLEAN_README = b"The `retry_failed` function returns an integer.\n"


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRUSTDOC_LOG_TO_FILE", "false")
    monkeypatch.setenv("TRUSTDOC_LOG_LEVEL", "CRITICAL")
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    settings.ensure_directories()
    db = Database(settings.storage.db_path)
    db.initialize()
    with db.session() as s:
        pid = s.projects.create(Project(name="demo")).id
    svc = IngestionService(db, settings)
    svc.ingest_bytes(pid, "pkg/mod.py", PY)
    svc.ingest_bytes(pid, "README.md", README)
    return pid


def run(capsys, *argv):
    code = main(["verify", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_help_documents_the_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["verify", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert e.value.code == 0
    for word in ("--project-id", "--project", "--document", "--document-file", "--semantic", "--fail-on", "--json", "--show-run",
                 "--no-docstrings", "deterministic", "never ground truth", "insufficient evidence", "status 3"):
        assert word in out
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "verify" in capsys.readouterr().out


def test_a_project_or_a_run_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        main(["verify"])
    assert e.value.code == 2 and "--project-id or --project" in capsys.readouterr().err


def test_report_shows_counts_findings_score_citations_and_skipped_checks(cli_env, capsys):
    code, out, err = run(capsys, "--project-id", str(cli_env))
    assert code == 3 and err == ""                                   # documentation problems found: not a crash
    for needle in ("Claims extracted: 8", "supported 3, contradicted 3", "Trust score: 47.5 / 100", "accuracy: 50.0%", "coverage: 66.7%",
                   "grounding: 83.3%", "severity penalty: -15.0", "[HIGH] 3", "[MEDIUM] 1", "[LOW] 3", "docs: README.md:3", "code: pkg/mod.py:",
                   "suggestion:", "Skipped checks:", "INSUFFICIENT EVIDENCE means unverified"):
        assert needle in out, needle
    assert "uploads" not in out and "SECRET" not in out


@pytest.mark.parametrize("fail_on, expected", [("none", 0), ("critical", 0), ("high", 3), ("medium", 3), ("low", 3)])
def test_exit_codes_follow_fail_on(cli_env, capsys, fail_on, expected):
    assert run(capsys, "--project-id", str(cli_env), "--fail-on", fail_on)[0] == expected


def test_default_threshold_is_high_and_info_never_fails(cli_env, capsys):
    assert run(capsys, "--project-id", str(cli_env), "--no-docstrings")[0] == 3                        # still HIGH findings in README
    with Database(Settings.from_env().storage.db_path).session() as s:
        clean = s.projects.create(Project(name="clean")).id
    svc = IngestionService(Database(Settings.from_env().storage.db_path), Settings.from_env())
    svc.ingest_bytes(clean, "pkg/mod.py", PY)
    svc.ingest_bytes(clean, "README.md", CLEAN_README)
    code, out, _ = run(capsys, "--project", "clean", "--no-docstrings")
    assert code == 0 and "contradicted 0" in out                       # coverage gaps (MEDIUM/LOW) do not fail the default threshold
    assert run(capsys, "--project", "clean", "--no-docstrings", "--fail-on", "medium")[0] == 3


def test_json_output_is_structured_and_complete(cli_env, capsys):
    code, out, _ = run(capsys, "--project-id", str(cli_env), "--json", "--fail-on", "none")
    data = json.loads(out)
    assert code == 0 and data["status"] == "completed" and data["trust_score"]["final"] == 47.5 and data["counts"]["claims"] == 8
    assert {c["status"] for c in data["claims"]} == {"supported", "contradicted", "not_checkable"}
    assert data["findings"][0]["severity"] == "high" and data["findings"][0]["doc_location"] and data["skipped"]
    assert all(f["uid"] and f["tier"] in ("deterministic", "semantic") for f in data["findings"])


def test_project_by_name_and_usage_errors(cli_env, capsys):
    assert run(capsys, "--project", "demo", "--fail-on", "none")[0] == 0
    for argv, message in ((["--project", "nope"], "No project named 'nope'"), (["--project-id", "424242"], "424242"),
                          (["--project-id", "1", "--document", "missing.md"], "No source named"),
                          (["--project-id", "1", "--document", "pkg/mod.py"], "only Markdown")):
        code, out, err = run(capsys, *argv)
        assert code == 2 and message in err and out == ""


def test_document_selection_and_docstring_switch(cli_env, capsys):
    code, out, _ = run(capsys, "--project-id", str(cli_env), "--document", "README.md", "--no-docstrings", "--fail-on", "none")
    assert code == 0 and "Claims extracted: 7" in out and "Documents: README.md" in out


def test_document_file_is_ingested_then_verified(cli_env, tmp_path, capsys):
    doc = tmp_path / "NOTES.md"
    doc.write_text("The `retry_failed` function returns a string.\n")
    code, out, _ = run(capsys, "--project-id", str(cli_env), "--document-file", str(doc), "--no-docstrings")
    assert code == 3 and "Documents: NOTES.md" in out and "documented as returning `str`" in out
    assert str(tmp_path) not in out                                       # only the logical name is shown


def test_semantic_requested_without_a_model_degrades_gracefully(cli_env, capsys):
    code, out, err = run(capsys, "--project-id", str(cli_env), "--semantic")
    assert code == 3 and "Semantic verification was unavailable" in out and "Claims extracted: 8" in out and err == ""


def test_stored_runs_can_be_shown_again(cli_env, capsys):
    run(capsys, "--project-id", str(cli_env))
    code, out, _ = run(capsys, "--show-run", "1", "--fail-on", "none")
    assert code == 0 and "Verification run #1" in out and "Trust score: 47.5 / 100" in out
    assert run(capsys, "--show-run", "1", "--project-id", "999")[0] == 2
    assert run(capsys, "--show-run", "99")[0] == 2


def test_operational_failures_exit_1_and_are_distinct_from_findings(cli_env, capsys, monkeypatch):
    from trustdoc.verification import VerificationEngine

    def boom(self, *a, **k):
        raise VerificationError("Verification failed: disk full")

    monkeypatch.setattr(VerificationEngine, "verify_project", boom)
    code, out, err = run(capsys, "--project-id", str(cli_env))
    assert code == 1 and "disk full" in err and out == ""


def test_existing_commands_still_work(cli_env, capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    info = json.loads(out[out.index("{"):])
    assert info["schema_version"] == SCHEMA_VERSION == 2 and info["settings"]["verification"]["weight_accuracy"] == 0.5
    with pytest.raises(SystemExit) as e:
        main(["ask", "--help"])
    assert e.value.code == 0
    assert main(["init"]) == 0


# ------------------------------------------------- Part 5.1 hotfix: repeated document arguments
def _services():
    settings = Settings.from_env()
    return Database(settings.storage.db_path), settings


def test_repeated_document_arguments_are_deduplicated(cli_env, capsys):
    base = ("--project-id", str(cli_env), "--no-docstrings", "--fail-on", "none")
    code1, once, _ = run(capsys, *base, "--document", "README.md")
    code2, twice, err = run(capsys, *base, "--document", "README.md", "--document", "README.md")
    assert code1 == code2 == 0 and err == ""
    assert "Documents: README.md\n" in twice and "Claims extracted: 7" in twice
    assert twice.split("Verification run #")[1].split("\n", 1)[1] == once.split("Verification run #")[1].split("\n", 1)[1]   # same report body
    code3, out, _ = run(capsys, *base, "--document", "README.md", "--document", "README.md", "--json")
    assert code3 == 0 and json.loads(out)["documents"] == ["README.md"]


def test_repeated_document_file_arguments_ingest_and_verify_once(cli_env, tmp_path, capsys):
    doc = tmp_path / "NOTES.md"
    doc.write_text("The `retry_failed` function returns a string.\n")
    code, out, err = run(capsys, "--project-id", str(cli_env), "--no-docstrings", "--document-file", str(doc), "--document-file", str(doc), "--json")
    data = json.loads(out)
    assert code == 3 and err == "" and data["documents"] == ["NOTES.md"] and data["status"] == "completed"
    assert [c["category"] for c in data["claims"]] == ["return"] and data["counts"]["contradicted"] == 1
    db, _ = _services()
    with db.session() as s:
        assert [x.name for x in s.sources.list_for_project(cli_env) if x.name == "NOTES.md"] == ["NOTES.md"]
        assert len(s.runs.list_for_project(cli_env)) == 1


def test_a_source_named_by_both_document_and_document_file_is_verified_once(cli_env, tmp_path, capsys):
    doc = tmp_path / "NOTES.md"
    doc.write_text("The `retry_failed` function returns a string.\n")
    run(capsys, "--project-id", str(cli_env), "--no-docstrings", "--document-file", str(doc), "--fail-on", "none")
    code, out, _ = run(capsys, "--project-id", str(cli_env), "--no-docstrings", "--document", "NOTES.md", "--document-file", str(doc),
                       "--document", "NOTES.md", "--json")
    assert code == 3 and json.loads(out)["documents"] == ["NOTES.md"]


def test_document_order_given_on_the_command_line_is_preserved(cli_env, capsys):
    db, settings = _services()
    IngestionService(db, settings).ingest_bytes(cli_env, "docs/other.md", b"The `retry_failed` function returns an integer.\n")
    code, out, _ = run(capsys, "--project-id", str(cli_env), "--no-docstrings", "--fail-on", "none",
                       "--document", "docs/other.md", "--document", "README.md", "--document", "docs/other.md")
    assert code == 0 and "Documents: docs/other.md, README.md" in out


def test_document_errors_keep_their_exit_codes_even_with_repeats(cli_env, capsys):
    for argv, message in ((["--document", "missing.md", "--document", "missing.md"], "No source named 'missing.md'"),
                          (["--document", "README.md", "--document", "pkg/mod.py", "--document", "pkg/mod.py"], "only Markdown")):
        code, out, err = run(capsys, "--project-id", str(cli_env), *argv)
        assert code == 2 and message in err and out == ""
