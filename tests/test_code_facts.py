"""Unit tests for deterministic code-fact answering and router integration."""

from __future__ import annotations

import pytest
from ask_support import FakeRetriever, ScriptedLLM, make_evidence, make_search_result

from trustdoc.config import Settings
from trustdoc.ingestion import IngestionService
from trustdoc.models import CodeEntity, EntityType, Parameter, ParameterKind
from trustdoc.workflows import AskService, AskStatus, GroundingLevel
from trustdoc.workflows.code_facts import CodeFactResolver

WORKERS_CODE = '''import random


def compute_backoff(
    attempt: int,
    base_seconds: float = 0.5,
    jitter: float = 0.1,
) -> float:
    """Calculate exponential retry delay with random jitter."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base_seconds < 0 or jitter < 0:
        raise ValueError("delays must be non-negative")
    return base_seconds * (2 ** attempt) + random.uniform(0.0, jitter)


def should_retry(attempt: int, max_attempts: int = 3) -> bool:
    """Check whether another retry is allowed."""
    return 0 <= attempt < max_attempts
'''

VARIED_CODE = '''def complex_func(
    pos_only: int,
    /,
    standard: str = "default_val",
    *args: int,
    kw_only: bool = True,
    **kwargs,
):
    pass

def unannotated_func(x, y=None):
    pass
'''


@pytest.fixture
def project_with_workers(db, project_id, settings):
    ingest = IngestionService(db, settings)
    res = ingest.ingest_bytes(project_id, "workers.py", WORKERS_CODE.encode("utf-8"))
    assert res.status.value == "ingested"
    return project_id


@pytest.fixture
def project_with_varied(db, project_id, settings):
    ingest = IngestionService(db, settings)
    res = ingest.ingest_bytes(project_id, "varied.py", VARIED_CODE.encode("utf-8"))
    assert res.status.value == "ingested"
    return project_id


# ------------------------------------------------------------------ 1. Return types
def test_return_type_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_workers, "What is the return type of compute_backoff?")

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "return type of `compute_backoff` is `float`" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert "workers.py" in res.citations[0].label
    assert res.citations[0].source_name == "workers.py"
    assert res.citations[0].qualified_name == "workers.compute_backoff"
    assert llm.calls == []  # No LLM call!


def test_return_type_unannotated(db, project_with_varied, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_varied, "What is the return type of unannotated_func?")

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "has no return type annotation" in res.answer
    assert llm.calls == []


# ------------------------------------------------------------------ 1b. Parameter Type Annotations
def test_parameter_annotation_max_attempts(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What is the type annotation of max_attempts in should_retry?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "type annotation of parameter `max_attempts` in `should_retry` is `int`" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert "workers.py" in res.citations[0].label
    assert res.citations[0].source_name == "workers.py"
    assert res.citations[0].qualified_name == "workers.should_retry"
    assert llm.calls == []


def test_parameter_annotation_other_param(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What is the type annotation for jitter in compute_backoff?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "type annotation of parameter `jitter` in `compute_backoff` is `float`" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert res.citations[0].qualified_name == "workers.compute_backoff"
    assert llm.calls == []


def test_parameter_annotation_unannotated_param(db, project_with_varied, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_varied,
        "What is the type annotation of x in unannotated_func?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "parameter `x` in `unannotated_func` has no type annotation" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "varied.py"
    assert llm.calls == []


def test_parameter_annotation_absent_param(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What is the type annotation of timeout in compute_backoff?",
    )

    assert res.status == AskStatus.INSUFFICIENT_EVIDENCE
    assert res.abstained is True
    assert res.abstention_reason == "parameter_not_indexed"
    assert "does not have a `timeout` parameter" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


def test_parameter_annotation_ambiguous_entity_falls_back_to_llm(db, project_id, settings):
    ingest = IngestionService(db, settings)
    ingest.ingest_bytes(project_id, "module_a.py", b"def execute(param: int):\n    pass\n")
    ingest.ingest_bytes(project_id, "module_b.py", b"def execute(param: str):\n    pass\n")

    evidence = [make_evidence(1, "execute handles int or str", source_name="module_a.py")]
    llm = ScriptedLLM("Param is int in module_a [E1].")
    retriever = FakeRetriever(make_search_result(evidence, project_id=project_id))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_id, "What is the type annotation of param in execute?")

    # Ambiguity detected -> falls back to LLM retrieval workflow!
    assert len(llm.calls) == 1
    assert res.status == AskStatus.ANSWERED


# ------------------------------------------------------------------ 2. Defaults
def test_parameter_defaults_specific(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What are the default values for base_seconds and jitter in compute_backoff?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "`base_seconds` defaults to `0.5`" in res.answer
    assert "`jitter` defaults to `0.1`" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


def test_parameters_and_defaults_full_list(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What are the parameters of should_retry and their default values?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "`attempt`: int, required (no default)" in res.answer
    assert "`max_attempts`: int, default=3" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert res.citations[0].qualified_name == "workers.should_retry"
    assert llm.calls == []


def test_parameters_kinds_and_varied_signatures(db, project_with_varied, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_varied, "What are the parameters of complex_func?")

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "`pos_only`: int (positional-only), required (no default)" in res.answer
    assert '`standard`: str, default="default_val"' in res.answer
    assert "`args`: int (*args), required (no default)" in res.answer
    assert "`kw_only`: bool (keyword-only), default=True" in res.answer
    assert "`kwargs` (**kwargs), required (no default)" in res.answer
    assert llm.calls == []


# ------------------------------------------------------------------ 3. Nonexistent Parameter
def test_nonexistent_parameter_refutes_false_premise(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "How do you set a maximum delay cap in seconds for compute_backoff using the max_delay parameter?",
    )

    assert res.status == AskStatus.INSUFFICIENT_EVIDENCE
    assert res.abstained is True
    assert res.abstention_reason == "parameter_not_indexed"
    assert res.grounding == GroundingLevel.CITED
    assert "does not have a `max_delay` parameter" in res.answer
    assert "premise is unsupported" in res.answer
    assert "`attempt`" in res.answer and "`base_seconds`" in res.answer and "`jitter`" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


# ------------------------------------------------------------------ 4. Entity Existence
def test_unindexed_function_existence_abstains_with_reason(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_workers, "What arguments does the retry_all() function accept?")

    assert res.status == AskStatus.INSUFFICIENT_EVIDENCE
    assert res.abstained is True
    assert res.abstention_reason == "entity_not_indexed"
    assert "No function or class named `retry_all` was found in the indexed sources" in res.answer
    assert llm.calls == []


def test_ambiguous_entity_name_falls_back_to_llm(db, project_id, settings):
    # Ingest two files with same bare function name 'execute'
    ingest = IngestionService(db, settings)
    ingest.ingest_bytes(project_id, "pkg/a.py", b"def execute(x: int) -> int: return x\n")
    ingest.ingest_bytes(project_id, "pkg/b.py", b"def execute(x: str) -> str: return x\n")

    evidence = [make_evidence(1, "def execute(x: int) -> int: return x", source_name="pkg/a.py")]
    llm = ScriptedLLM("Execute handles both types [E1].")
    retriever = FakeRetriever(make_search_result(evidence, project_id=project_id))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_id, "What is the return type of execute?")

    # Ambiguity detected -> falls back to LLM retrieval workflow!
    assert len(llm.calls) == 1
    assert res.status == AskStatus.ANSWERED


# ------------------------------------------------------------------ 5. AST Facts (Imports, Exceptions, Conditions)
def test_imports_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_workers, "Which standard library module does workers.py import?")

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "`random`" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


def test_exception_raised_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res1 = service.ask(
        project_with_workers,
        "What exception is raised if attempt is negative in compute_backoff?",
    )
    assert res1.status == AskStatus.ANSWERED
    assert res1.grounding == GroundingLevel.CITED
    assert "ValueError('attempt must be non-negative')" in res1.answer
    assert "attempt < 0" in res1.answer
    assert llm.calls == []

    res2 = service.ask(
        project_with_workers,
        "What exception is raised if base_seconds or jitter is negative in compute_backoff?",
    )
    assert res2.status == AskStatus.ANSWERED
    assert res2.grounding == GroundingLevel.CITED
    assert "ValueError('delays must be non-negative')" in res2.answer
    assert "base_seconds < 0 or jitter < 0" in res2.answer
    assert llm.calls == []


def test_return_condition_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(project_with_workers, "Under what condition does should_retry return True?")

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "0 <= attempt < max_attempts" in res.answer
    assert llm.calls == []


# ------------------------------------------------------------------ 6. Fallback for broader questions
def test_fallback_to_llm_for_broader_questions(db, project_with_workers, settings):
    evidence = [make_evidence(1, "The task queue has no external dependencies.", source_name="README.md")]
    llm = ScriptedLLM("The project has no external dependencies [E1].")
    retriever = FakeRetriever(make_search_result(evidence, project_id=project_with_workers))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What is the overall architecture and high-level design of the system?",
    )

    assert len(llm.calls) == 1
    assert res.status == AskStatus.ANSWERED
    assert "The project has no external dependencies [E1]." in res.answer


# ------------------------------------------------------------------ 7. Formula and Jitter
def test_formula_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "What is the mathematical formula used to calculate the retry delay in compute_backoff?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "base_seconds * 2 ** attempt + random.uniform(0.0, jitter)" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


def test_jitter_generation_deterministic(db, project_with_workers, settings):
    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_with_workers,
        "How does compute_backoff generate the random jitter added to the delay?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "random.uniform(0.0, jitter)" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "workers.py"
    assert llm.calls == []


def test_project_dependencies_deterministic(db, project_id, settings):
    ingest = IngestionService(db, settings)
    readme_content = "# Retry Task Queue\n\n## Dependencies\n\nThe example uses only the Python standard library.\n"
    res_ingest = ingest.ingest_bytes(project_id, "README.md", readme_content.encode("utf-8"))
    assert res_ingest.status.value == "ingested"

    llm = ScriptedLLM("LLM should not be called")
    retriever = FakeRetriever(make_search_result([]))
    service = AskService(db, settings, retriever, llm)

    res = service.ask(
        project_id,
        "What external dependencies does the retry-task-queue project require?",
    )

    assert res.status == AskStatus.ANSWERED
    assert res.grounding == GroundingLevel.CITED
    assert "only the Python standard library" in res.answer
    assert "[E1]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_name == "README.md"
    assert llm.calls == []
