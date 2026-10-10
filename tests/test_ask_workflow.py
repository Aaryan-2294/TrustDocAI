import json

import pytest
from ask_support import FakeRetriever, ScriptedLLM, make_evidence, make_search_result

from trustdoc.config import Settings
from trustdoc.exceptions import (
    InvalidQuestionError,
    LLMEmptyOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    MissingDependencyError,
    NotFoundError,
    RetrievalError,
)
from trustdoc.workflows import AskService, AskStatus, GroundingLevel
from trustdoc.workflows.ask import INSUFFICIENT_MESSAGE


def settings_with(base, **env):
    values = {"TRUSTDOC_DATA_DIR": str(base.storage.data_dir), "TRUSTDOC_LOG_TO_FILE": "false"}
    values.update({f"TRUSTDOC_{k}": str(v) for k, v in env.items()})
    return Settings.from_env(values)


@pytest.fixture
def evidence():
    return [
        make_evidence(1, "def compute_backoff(attempt):\n    return 2 ** attempt", source_name="taskqueue/workers.py",
                      start=47, end=52, qualified_name="taskqueue.workers.compute_backoff"),
        make_evidence(2, "Retries use exponential backoff.", source_name="reference.pdf", source_type="pdf",
                      page=2, start=None, end=None, rerank=0.8),
    ]


@pytest.fixture
def make_service(db, settings, project_id):
    def build(result, llm=None, cfg=None):
        retriever = FakeRetriever(result)
        service = AskService(db, cfg or settings, retriever, llm)
        return service, retriever

    return build


# ------------------------------------------------------------------- success
def test_grounded_answer_preserves_evidence_and_maps_citations(project_id, evidence, make_service):
    llm = ScriptedLLM("Retries back off exponentially [E1]. The reference confirms it [E2][E1].")
    service, retriever = make_service(make_search_result(evidence, project_id=project_id), llm)
    for e in evidence:
        e.project_id = project_id
    r = service.ask(project_id, "How do retries work?")

    assert r.status == AskStatus.ANSWERED and not r.abstained and r.claims_verified is False
    assert r.grounding == GroundingLevel.CITED and r.invalid_citation_ids == []
    assert r.answer == r.raw_answer == "Retries back off exponentially [E1]. The reference confirms it [E2][E1]."
    assert [c.id for c in r.citations] == ["E1", "E2"] and [c.mentions for c in r.citations] == [2, 1]
    c1, c2 = r.citations
    assert (c1.label, c1.source_name, c1.start_line, c1.end_line, c1.qualified_name) == \
        ("taskqueue/workers.py:47-52", "taskqueue/workers.py", 47, 52, "taskqueue.workers.compute_backoff")
    assert (c2.label, c2.page_number, c2.source_type) == ("reference.pdf, p.2", 2, "pdf")
    assert [e.id for e in r.evidence] == ["E1", "E2"] and r.evidence[0].evidence is evidence[0]
    assert (r.model_name, r.backend, r.is_stub_model, r.finish_reason) == ("scripted-model", "scripted", False, "stop")
    assert len(llm.calls) == 1 and "taskqueue/workers.py:47-52" in llm.calls[0]["prompt"]
    assert "How do retries work?" in llm.calls[0]["prompt"] and llm.calls[0]["system_prompt"]
    assert retriever.calls == [{"project_id": project_id, "query": "How do retries work?", "top_k": 5}]
    assert {"retrieval_ms", "generation_ms", "total_ms"} <= set(r.timings_ms)
    assert r.prompt_stats["evidence_items"] == 2
    assert set(r.prompt_stats) == {"evidence_items", "evidence_chars", "approx_prompt_tokens"}   # sizes only, no prompt text


def test_top_k_override_and_config_default_are_passed_to_retrieval(db, settings, project_id, evidence, make_service):
    cfg = settings_with(settings, ASK_MAX_EVIDENCE_ITEMS=3)
    service, retriever = make_service(make_search_result(evidence), ScriptedLLM(), cfg)
    service.ask(project_id, "q")
    service.ask(project_id, "q", top_k=9)
    assert [c["top_k"] for c in retriever.calls] == [3, 9]
    with pytest.raises(ValueError):
        service.ask(project_id, "q", top_k=0)


# ---------------------------------------------------------- citation outcomes
def test_uncited_answer_is_flagged_not_called_grounded(project_id, evidence, make_service):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM("Retries use exponential backoff."))
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.ANSWERED and r.grounding == GroundingLevel.UNCITED and r.citations == []
    assert any("no valid citations" in w for w in r.warnings)


def test_unknown_citation_ids_are_flagged_and_marked_not_replaced(project_id, evidence, make_service):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM("Yes [E1], and also [E9]."))
    r = service.ask(project_id, "q")
    assert r.grounding == GroundingLevel.CITED_WITH_INVALID_REFERENCES and r.invalid_citation_ids == ["E9"]
    assert r.answer == "Yes [E1], and also [unknown source E9]." and r.raw_answer == "Yes [E1], and also [E9]."
    assert [c.id for c in r.citations] == ["E1"] and any("E9" in w for w in r.warnings)


def test_only_unknown_ids_means_uncited(project_id, evidence, make_service):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM("Claim [E5]."))
    r = service.ask(project_id, "q")
    assert r.grounding == GroundingLevel.UNCITED and r.citations == [] and r.invalid_citation_ids == ["E5"]


def test_ids_of_evidence_dropped_by_the_budget_are_not_valid(db, settings, project_id, make_service):
    cfg = settings_with(settings, ASK_MAX_EVIDENCE_ITEMS=1)
    ev = [make_evidence(1, "first"), make_evidence(2, "second", source_name="other.py")]
    service, _ = make_service(make_search_result(ev), ScriptedLLM("see [E1] and [E2]"), cfg)
    r = service.ask(project_id, "q")
    assert [c.id for c in r.citations] == ["E1"] and r.invalid_citation_ids == ["E2"]
    assert r.dropped_evidence == ["other.py:10-24"] and any("did not fit" in w for w in r.warnings)


# ----------------------------------------------------------------- abstention
def test_retrieval_abstention_skips_the_llm(project_id, evidence, make_service):
    llm = ScriptedLLM()
    low = [make_evidence(1, "weak", rerank=0.1)]
    service, _ = make_service(make_search_result([], abstained=True, ranked=low, threshold=0.5), llm)
    r = service.ask(project_id, "unrelated?")
    assert r.status == AskStatus.INSUFFICIENT_EVIDENCE and r.abstained and r.answer == INSUFFICIENT_MESSAGE
    assert r.abstention_reason == "below_relevance_threshold" and llm.calls == []
    assert r.evidence == [] and r.citations == [] and r.grounding is None
    assert r.retrieval.abstained and r.retrieval.top_candidates == [{"label": "pkg/module.py:10-24", "rerank_score": 0.1}]
    assert "weak" not in json.dumps(r.to_dict())                          # candidate text is never exposed


def test_empty_retrieval_skips_the_llm(project_id, make_service):
    llm = ScriptedLLM()
    service, _ = make_service(make_search_result([], abstained=True, ranked=[]), llm)
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.INSUFFICIENT_EVIDENCE and r.abstention_reason == "no_evidence_retrieved" and llm.calls == []


def test_no_usable_evidence_without_abstain_flag_is_still_insufficient(project_id, make_service):
    llm = ScriptedLLM()
    service, _ = make_service(make_search_result([], abstained=False, ranked=[]), llm)
    assert service.ask(project_id, "q").status == AskStatus.INSUFFICIENT_EVIDENCE and llm.calls == []


def test_abstention_needs_no_model_but_answering_without_one_reports_not_configured(project_id, evidence, make_service):
    service, _ = make_service(make_search_result([], abstained=True, ranked=[]), None)   # provider "none"
    assert service.ask(project_id, "q").status == AskStatus.INSUFFICIENT_EVIDENCE
    service, _ = make_service(make_search_result(evidence), None)
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.GENERATION_ERROR and r.error_kind == "not_configured"
    assert "TRUSTDOC_LLM_PROVIDER" in r.error and len(r.evidence) == 2 and r.answer == ""


def test_model_declared_insufficiency_is_reported_as_insufficient(project_id, evidence, make_service):
    llm = ScriptedLLM("INSUFFICIENT_EVIDENCE\nThe evidence never mentions timeouts [E1].")
    service, _ = make_service(make_search_result(evidence), llm)
    r = service.ask(project_id, "What is the timeout?")
    assert r.status == AskStatus.INSUFFICIENT_EVIDENCE and r.abstained and r.grounding is None
    assert r.abstention_reason == "model_declared_insufficient"
    assert r.answer.startswith(INSUFFICIENT_MESSAGE) and "never mentions timeouts [E1]" in r.answer
    assert len(r.evidence) == 2 and r.raw_answer.startswith("INSUFFICIENT_EVIDENCE")


def test_placeholder_threshold_is_called_out(project_id, evidence, make_service):
    service, _ = make_service(make_search_result(evidence, threshold=0.0), ScriptedLLM())
    assert any("uncalibrated placeholder" in w for w in service.ask(project_id, "q").warnings)
    service, _ = make_service(make_search_result(evidence, threshold=0.4), ScriptedLLM())
    assert not any("uncalibrated" in w for w in service.ask(project_id, "q").warnings)


# ----------------------------------------------------------------- validation
def test_missing_project_raises_before_retrieval(db, make_service, evidence):
    service, retriever = make_service(make_search_result(evidence), ScriptedLLM())
    with pytest.raises(NotFoundError):
        service.ask(98765, "q")
    assert retriever.calls == []


@pytest.mark.parametrize("bad", ["", "   \n", None, 42, "has\x00nul"])
def test_invalid_questions_are_rejected(project_id, evidence, make_service, bad):
    service, retriever = make_service(make_search_result(evidence), ScriptedLLM())
    with pytest.raises(InvalidQuestionError):
        service.ask(project_id, bad)
    assert retriever.calls == []


def test_overlong_question_is_rejected_not_truncated(db, settings, project_id, evidence, make_service):
    cfg = settings_with(settings, ASK_MAX_QUESTION_CHARS=50)
    llm = ScriptedLLM()
    service, retriever = make_service(make_search_result(evidence), llm, cfg)
    with pytest.raises(InvalidQuestionError, match="not truncated"):
        service.ask(project_id, "x" * 51)
    exact = "y" * 50
    service.ask(project_id, exact)
    assert exact in llm.calls[0]["prompt"] and retriever.calls[0]["query"] == exact


# ------------------------------------------------------------- failure handling
@pytest.mark.parametrize("exc, kind", [
    (LLMTimeoutError("too slow"), "timeout"),
    (LLMUnavailableError("model missing"), "unavailable"),
    (MissingDependencyError("pip install x"), "dependency_missing"),
    (LLMEmptyOutputError("nothing"), "empty_output"),
    (RuntimeError("boom"), "unexpected"),
])
def test_generation_failures_become_structured_results_with_evidence_kept(project_id, evidence, make_service, exc, kind):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM(exc))
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.GENERATION_ERROR and r.error_kind == kind and r.answer == ""
    assert str(exc) in r.error and len(r.evidence) == 2 and r.citations == [] and r.grounding is None
    assert r.model_name == "scripted-model"
    if kind == "unexpected":
        assert "RuntimeError" in r.error


@pytest.mark.parametrize("reply", ["", "   \n\t ", "\x00\x07"])
def test_empty_generation_output_is_an_error_not_an_answer(project_id, evidence, make_service, reply):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM(reply))
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.GENERATION_ERROR and r.error_kind == "empty_output" and r.answer == ""


@pytest.mark.parametrize("exc, kind", [
    (RetrievalError("no index"), "retrieval_failed"),
    (MissingDependencyError("pip install faiss"), "dependency_missing"),
])
def test_retrieval_errors_are_structured_and_skip_the_llm(project_id, make_service, exc, kind):
    llm = ScriptedLLM()
    service, _ = make_service(exc, llm)
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.RETRIEVAL_ERROR and r.error_kind == kind and llm.calls == []


@pytest.mark.parametrize("finish, text", [("length", "maximum token limit"), ("time_limit", "time limit")])
def test_truncated_generation_is_warned_about(project_id, evidence, make_service, finish, text):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM("Answer [E1].", finish_reason=finish))
    r = service.ask(project_id, "q")
    assert r.status == AskStatus.ANSWERED and any(text in w for w in r.warnings)


# ------------------------------------------------------------------ isolation
def test_evidence_from_another_project_never_reaches_the_prompt_or_result(db, project_id, make_service):
    from trustdoc.models import Project
    with db.session() as s:
        other = s.projects.create(Project(name="other")).id
    mine = make_evidence(1, "my project text", project_id=project_id)
    foreign = make_evidence(2, "SECRET text from the other project", project_id=other, source_name="secret.py")
    llm = ScriptedLLM("Answer [E1].")
    service, _ = make_service(make_search_result([foreign, mine]), llm)
    r = service.ask(project_id, "q")
    assert "SECRET" not in llm.calls[0]["prompt"] and "secret.py" not in json.dumps(r.to_dict(include_evidence_text=True))
    assert [e.evidence.project_id for e in r.evidence] == [project_id]
    assert any("different project" in w for w in r.warnings)

    llm2 = ScriptedLLM()
    service, _ = make_service(make_search_result([foreign]), llm2)
    only_foreign = service.ask(project_id, "q")
    assert only_foreign.status == AskStatus.INSUFFICIENT_EVIDENCE and llm2.calls == []
    assert "secret.py" not in json.dumps(only_foreign.to_dict()) and only_foreign.retrieval.top_candidates == []


# --------------------------------------------------------------- prompt budget
def test_budget_truncation_and_neutralisation_are_reported(db, settings, project_id, make_service):
    cfg = settings_with(settings, ASK_MAX_EVIDENCE_CHARS=200)
    ev = [make_evidence(1, "z" * 400 + " [E3]")]
    service, _ = make_service(make_search_result(ev), ScriptedLLM("Answer [E1]."), cfg)
    r = service.ask(project_id, "q")
    assert r.evidence[0].truncated and any("truncated" in w for w in r.warnings)
    service, _ = make_service(make_search_result([make_evidence(1, "see [E3] here")]), ScriptedLLM("A [E1]"))
    r = service.ask(project_id, "q")
    assert r.evidence[0].sanitized and any("neutralised" in w for w in r.warnings)


# ------------------------------------------------------------- lazy LLM handling
def test_llm_factory_is_called_lazily_and_once(project_id, evidence, db, settings):
    calls = []

    def factory():
        calls.append(1)
        return ScriptedLLM("ok [E1]")

    abstain = FakeRetriever(make_search_result([], abstained=True, ranked=[]))
    service = AskService(db, settings, abstain, factory)
    service.ask(project_id, "q")
    assert calls == []                                                    # abstention never builds a model
    service = AskService(db, settings, FakeRetriever(make_search_result(evidence)), factory)
    service.ask(project_id, "q")
    service.ask(project_id, "q")
    assert calls == [1]


# ------------------------------------------------------------------ serialisation
def test_to_dict_is_json_safe_and_hides_evidence_text_by_default(project_id, evidence, make_service):
    service, _ = make_service(make_search_result(evidence), ScriptedLLM("Yes [E1]."))
    r = service.ask(project_id, "q")
    data = json.loads(json.dumps(r.to_dict()))
    assert data["status"] == "answered" and data["grounding"] == "cited" and data["claims_verified"] is False
    assert "return 2 ** attempt" not in json.dumps(data) and "text" not in data["evidence"][0]
    assert "return 2 ** attempt" in json.dumps(r.to_dict(include_evidence_text=True))
