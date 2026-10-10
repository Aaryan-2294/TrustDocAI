import json

import pytest
from ask_support import FakeRetriever, ScriptedLLM, make_evidence, make_search_result

from trustdoc.cli import main
from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import LLMTimeoutError, MissingDependencyError, RetrievalError
from trustdoc.models import Project
from trustdoc.workflows import AskService

EVIDENCE_TEXT = "def compute_backoff(attempt):\n    return 2 ** attempt  # SECRET-SOURCE-BODY"


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """A real (empty) database with one project, reachable through the CLI via env vars."""
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
    return pid


def wire(monkeypatch, result, llm=None):
    """Patch the service factory so the CLI uses mocks but the real settings/db/bootstrap."""
    captured = {}

    def fake_create(db, settings, **kwargs):
        captured["settings"] = settings
        retriever = FakeRetriever(result)
        captured["retriever"] = retriever
        return AskService(db, settings, retriever, llm)

    monkeypatch.setattr("trustdoc.workflows.create_ask_service", fake_create)
    return captured


def evidence(pid):
    return [make_evidence(1, EVIDENCE_TEXT, project_id=pid, source_name="taskqueue/workers.py", start=47, end=52,
                          qualified_name="taskqueue.workers.compute_backoff"),
            make_evidence(2, "Retries use backoff.", project_id=pid, source_name="reference.pdf", source_type="pdf",
                          page=2, start=None, end=None)]


def test_help_documents_the_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["ask", "--help"])
    out = " ".join(capsys.readouterr().out.split())          # argparse wraps lines
    assert e.value.code == 0
    for word in ("--project-id", "--project", "--question", "--top-k", "--provider", "--model", "--json",
                 "--show-evidence", "TRUSTDOC_LLM_PROVIDER", "not called"):
        assert word in out
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert "ask" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["ask", "--question", "q"], ["ask", "--project-id", "1"],
                                  ["ask", "--project-id", "1", "--project", "x", "--question", "q"]])
def test_missing_or_conflicting_arguments_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2


def test_answered_output_shows_answer_citations_and_disclaimer_but_not_the_prompt_or_source(cli_env, monkeypatch, capsys):
    llm = ScriptedLLM("Backoff doubles each attempt [E1]. The reference agrees [E2].")
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), llm)
    code = main(["ask", "--project-id", str(cli_env), "--question", "How does the retry mechanism work?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Question: How does the retry mechanism work?" in out and "Answer:" in out
    assert "Backoff doubles each attempt [E1]." in out and "citation check: cited" in out
    assert "[E1] taskqueue/workers.py:47-52  (taskqueue.workers.compute_backoff)" in out
    assert "[E2] reference.pdf, p.2" in out and "NOT been independently verified" in out
    assert "model: scripted-model (scripted)" in out
    assert "SECRET-SOURCE-BODY" not in out and "=== BEGIN EVIDENCE" not in out and "You are TrustDoc AI" not in out


def test_show_evidence_prints_short_snippets_only(cli_env, monkeypatch, capsys):
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), ScriptedLLM("Yes [E1]."))
    main(["ask", "--project-id", str(cli_env), "-q", "q", "--show-evidence"])
    out = capsys.readouterr().out
    assert "Evidence shown to the model (snippets):" in out and "def compute_backoff(attempt): return 2 ** attempt" in out
    assert "Retrieved evidence not cited in the answer:" in out and "[E2] reference.pdf, p.2" in out


def test_uncited_and_invalid_citations_are_visible_to_the_user(cli_env, monkeypatch, capsys):
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), ScriptedLLM("No citations here."))
    main(["ask", "--project-id", str(cli_env), "-q", "q"])
    out = capsys.readouterr().out
    assert "citation check: uncited" in out and "no valid citations" in out


def test_insufficient_evidence_is_explicit_and_exits_zero(cli_env, monkeypatch, capsys):
    llm = ScriptedLLM()
    weak = [make_evidence(1, "weak", project_id=cli_env, rerank=0.12, source_name="notes.md", source_type="markdown")]
    wire(monkeypatch, make_search_result([], project_id=cli_env, abstained=True, ranked=weak), llm)
    code = main(["ask", "--project-id", str(cli_env), "-q", "Who won the world cup?"])
    out = capsys.readouterr().out
    assert code == 0 and "INSUFFICIENT EVIDENCE (below relevance threshold)" in out
    assert "The available evidence is insufficient" in out
    assert "Closest retrieved sources (not used to answer):" in out and "notes.md:10-24  (relevance score 0.120)" in out
    assert llm.calls == [] and "Answer:" not in out


def test_json_output_is_valid_and_excludes_source_text(cli_env, monkeypatch, capsys):
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), ScriptedLLM("Yes [E1]."))
    assert main(["ask", "--project-id", str(cli_env), "-q", "q", "--json"]) == 0
    raw = capsys.readouterr().out
    data = json.loads(raw)
    assert data["status"] == "answered" and data["citations"][0]["label"] == "taskqueue/workers.py:47-52"
    assert data["claims_verified"] is False and "SECRET-SOURCE-BODY" not in raw


def test_project_can_be_selected_by_name_and_options_reach_the_service(cli_env, monkeypatch, capsys):
    captured = wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), ScriptedLLM("Yes [E1]."))
    assert main(["ask", "--project", "demo", "-q", "q", "--top-k", "7", "--provider", "OLLAMA", "--model", "tiny"]) == 0
    assert captured["retriever"].calls[0]["top_k"] == 7 and captured["retriever"].calls[0]["project_id"] == cli_env
    llm_cfg = captured["settings"].llm
    assert (llm_cfg.provider, llm_cfg.model) == ("OLLAMA", "tiny")


@pytest.mark.parametrize("argv, message", [
    (["ask", "--project", "nope", "-q", "q"], "No project named 'nope'"),
    (["ask", "--project-id", "424242", "-q", "q"], "424242"),
    (["ask", "--project-id", "1", "-q", "   "], "must not be empty"),
])
def test_usage_errors_exit_2_with_a_message_on_stderr(cli_env, monkeypatch, capsys, argv, message):
    wire(monkeypatch, make_search_result([], abstained=True, ranked=[]))
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert message in captured.err and captured.out == ""


def test_no_model_configured_exits_2_with_instructions(cli_env, monkeypatch, capsys):
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), None)   # provider "none"
    assert main(["ask", "--project-id", str(cli_env), "-q", "q"]) == 2
    captured = capsys.readouterr()
    assert "TRUSTDOC_LLM_PROVIDER" in captured.err and "not_configured" in captured.err and "Answer:" not in captured.out
    assert "taskqueue/workers.py:47-52" in captured.err              # retrieved evidence is still listed


@pytest.mark.parametrize("reply, code, text", [
    (LLMTimeoutError("did not answer within 5s"), 1, "failed (timeout)"),
    (MissingDependencyError("pip install torch"), 2, "failed (dependency_missing)"),
    (RuntimeError("weird"), 1, "failed (unexpected)"),
])
def test_generation_failures_map_to_exit_codes(cli_env, monkeypatch, capsys, reply, code, text):
    wire(monkeypatch, make_search_result(evidence(cli_env), project_id=cli_env), ScriptedLLM(reply))
    assert main(["ask", "--project-id", str(cli_env), "-q", "q"]) == code
    err = capsys.readouterr().err
    assert text in err and "no answer was produced" in err


def test_retrieval_failure_exits_1(cli_env, monkeypatch, capsys):
    wire(monkeypatch, RetrievalError("index is corrupt"), ScriptedLLM())
    assert main(["ask", "--project-id", str(cli_env), "-q", "q"]) == 1
    assert "Retrieval failed (retrieval_failed): index is corrupt" in capsys.readouterr().err


def test_existing_commands_still_work(cli_env, capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    info = json.loads(out[out.index("{"):])
    assert info["schema_version"] == 1 and "ask" in info["settings"] and info["settings"]["llm"]["provider"] == "none"
