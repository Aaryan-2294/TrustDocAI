"""Tests for the trustdoc generate CLI command (Part 6)."""

import json
from pathlib import Path

import pytest
from ask_support import ScriptedLLM

from trustdoc.cli import main
from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset
from trustdoc.ingestion import IngestionService
from trustdoc.llm.stub import ExtractiveStubLLM
from trustdoc.models import Project
from trustdoc.retrieval import HashingEmbedder, OverlapReranker, create_retriever
from trustdoc.workflows import GenerationService


@pytest.fixture
def cli_sample_project(tmp_path, monkeypatch):
    """Real database with ingested sample project, reachable through CLI."""
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRUSTDOC_LOG_TO_FILE", "false")
    monkeypatch.setenv("TRUSTDOC_LOG_LEVEL", "CRITICAL")
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    settings.ensure_directories()
    db = Database(settings.storage.db_path)
    db.initialize()
    with db.session() as s:
        pid = s.projects.create(Project(name="sample")).id
    ingestion = IngestionService(db, settings)
    ingestion.ingest_directory(pid, load_dataset(DEFAULT_QUERIES).project_dir)
    return pid


def wire_generation(monkeypatch, llm=None):
    """Patch create_generation_service so the CLI uses lightweight stand-in models."""
    def fake_create(db, settings, **kwargs):
        retriever = create_retriever(db, settings, embedder=HashingEmbedder(), reranker=OverlapReranker())
        backend = llm or ExtractiveStubLLM()
        return GenerationService(db, settings, retriever, backend)

    monkeypatch.setattr("trustdoc.workflows.create_generation_service", fake_create)


def test_generate_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["generate", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--kind" in out
    assert "--entity" in out
    assert "--output" in out
    assert "--fail-on" in out
    assert "--json" in out


@pytest.mark.parametrize("argv", [
    ["generate"],
    ["generate", "--kind", "readme"],  # missing project
    ["generate", "--project-id", "1"],  # missing kind
    ["generate", "--project-id", "1", "--kind", "api"],  # missing --entity for api
    ["generate", "--project-id", "1", "--project", "sample", "--kind", "readme"],  # conflicting project flags
])
def test_missing_or_conflicting_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2


def test_unknown_project_exits_2(cli_sample_project, capsys):
    code = main(["generate", "--project-id", "9999", "--kind", "readme"])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_unknown_entity_exits_2(cli_sample_project, monkeypatch, capsys):
    wire_generation(monkeypatch)
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "non_existent_fn"])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_generate_readme_cli_success(cli_sample_project, monkeypatch, capsys):
    wire_generation(monkeypatch)
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "readme"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Documentation generation: README" in out
    assert "Result: completed" in out
    assert "Sources cited:" in out
    assert "Independent Verification Report" in out


def test_generate_api_cli_success(cli_sample_project, monkeypatch, capsys):
    wire_generation(monkeypatch)
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Documentation generation: API" in out
    assert "compute_backoff" in out
    assert "Result: completed" in out
    assert "Independent Verification Report" in out


def test_generate_output_file_safe_write_and_force(cli_sample_project, monkeypatch, tmp_path, capsys):
    wire_generation(monkeypatch)
    out_file = tmp_path / "out" / "api.md"

    # 1. Successful safe write
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff",
                 "--output", str(out_file)])
    assert code == 0
    assert out_file.exists()
    content_first = out_file.read_text(encoding="utf-8")
    assert len(content_first) > 0

    # 2. Refuses to overwrite existing file without --force
    capsys.readouterr()  # clear buffer
    code2 = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff",
                  "--output", str(out_file)])
    assert code2 == 2
    err2 = capsys.readouterr().err
    assert "already exists" in err2

    # 3. Overwrites with --force
    code3 = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff",
                  "--output", str(out_file), "--force"])
    assert code3 == 0


def test_generate_json_output(cli_sample_project, monkeypatch, capsys):
    wire_generation(monkeypatch)
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["project_id"] == cli_sample_project
    assert data["kind"] == "api"
    assert data["status"] == "completed"
    assert "markdown" in data
    assert "verification_report" in data
    assert "citations" in data


def test_generate_fail_on_exits_with_status_3_on_flawed_docs(cli_sample_project, monkeypatch, capsys):
    # LLM hallucinates non-existent parameter base_delay
    flawed_doc = """
### `compute_backoff`
The `compute_backoff` function accepts `attempt` and `base_delay`, and returns an integer. [E1]
"""
    wire_generation(monkeypatch, llm=ScriptedLLM(flawed_doc))

    # default fail-on is high -> should exit with code 3 because contradiction findings are HIGH/CRITICAL
    code = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff", "--fail-on", "high"])
    assert code == 3

    # fail-on none -> should exit with code 0 despite findings
    code_none = main(["generate", "--project-id", str(cli_sample_project), "--kind", "api", "--entity", "compute_backoff", "--fail-on", "none"])
    assert code_none == 0
