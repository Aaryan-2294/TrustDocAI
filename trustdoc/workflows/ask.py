"""The Ask workflow: question -> retrieval -> grounded prompt -> LLM -> checked citations.

Orchestration only. Retrieval (BM25/FAISS/RRF/Cross-Encoder, indexes, threshold)
belongs to ``trustdoc.retrieval`` and text generation to ``trustdoc.llm``.

Outcomes (``AskResult.status``):
* ``answered``              the model produced an answer (see ``grounding`` for its citation status)
* ``insufficient_evidence`` retrieval abstained / found nothing, or the model itself declared the
                            evidence insufficient. When retrieval abstains the LLM is NOT called.
* ``retrieval_error`` / ``generation_error``   operational failures; retrieved evidence is preserved
                            and no answer is fabricated.
Invalid questions and unknown projects raise (``InvalidQuestionError`` / ``NotFoundError``).

Citations are checked deterministically (IDs must refer to evidence actually shown to the
model). That is not claim verification: ``claims_verified`` is always False here.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Union

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import (
    InvalidQuestionError,
    LLMEmptyOutputError,
    LLMError,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    MissingDependencyError,
    RetrievalError,
)
from trustdoc.llm.base import LLMBackend
from trustdoc.llm.factory import create_llm
from trustdoc.logging_setup import get_logger
from trustdoc.retrieval.types import SearchResult
from trustdoc.workflows.citations import analyze_citations
from trustdoc.workflows.prompts import INSUFFICIENT_SENTINEL, PromptBundle, build_prompt
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    AskResult,
    AskStatus,
    GroundingLevel,
    RetrievalSummary,
)

log = get_logger("ask")

INSUFFICIENT_MESSAGE = "The available evidence is insufficient to answer this question."
PLACEHOLDER_THRESHOLD_WARNING = (
    "The retrieval relevance threshold is the uncalibrated placeholder 0.0, so a non-abstaining "
    "result does not by itself show that the evidence is sufficient."
)
_SENTINEL_RE = re.compile(rf"^\s*{INSUFFICIENT_SENTINEL.replace('_', '[_ ]')}\b[\s:.\-]*", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

LLMSource = Union[LLMBackend, Callable[[], LLMBackend], None]


def classify_llm_error(exc: BaseException) -> str:
    """Short machine-readable kind for an LLM failure."""
    if isinstance(exc, LLMNotConfiguredError):
        return "not_configured"
    if isinstance(exc, MissingDependencyError):
        return "dependency_missing"
    if isinstance(exc, LLMUnavailableError):
        return "unavailable"
    if isinstance(exc, LLMTimeoutError):
        return "timeout"
    if isinstance(exc, LLMEmptyOutputError):
        return "empty_output"
    if isinstance(exc, LLMResponseError):
        return "invalid_response"
    if isinstance(exc, LLMError):
        return "generation_failed"
    return "unexpected"


class AskService:
    def __init__(
        self, db: Database, settings: Settings, retriever, llm: LLMSource = None
    ) -> None:
        """``retriever`` is anything with ``search(project_id, query, top_k=...)`` (the Part 3
        ``Retriever``). ``llm`` may be a backend, a zero-argument factory, or None (built from
        settings on first use, and only if a model is actually needed)."""
        self._db = db
        self._settings = settings
        self._retriever = retriever
        self._llm_source = llm
        self._llm: LLMBackend | None = llm if isinstance(llm, LLMBackend) else None

    # ------------------------------------------------------------------ public
    def ask(self, project_id: int, question: str, *, top_k: int | None = None) -> AskResult:
        started = time.perf_counter()
        question = self._validate_question(question)
        with self._db.session() as s:
            s.projects.require(project_id)
        cfg = self._settings.ask
        k = cfg.max_evidence_items if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be positive")
        timings: dict[str, float] = {}
        warnings: list[str] = []

        # ---- 1. retrieval (Part 3 owns all of it)
        t = time.perf_counter()
        try:
            found = self._retriever.search(project_id, question, top_k=k)
        except (RetrievalError, MissingDependencyError) as exc:
            log.warning("Retrieval failed for project %d: %s", project_id, exc)
            return AskResult(
                project_id, question, AskStatus.RETRIEVAL_ERROR, answer="", error=str(exc),
                error_kind="dependency_missing" if isinstance(exc, MissingDependencyError) else "retrieval_failed",
                timings_ms={"retrieval_ms": _ms(t), "total_ms": _ms(started)},
            )
        timings["retrieval_ms"] = _ms(t)
        summary = _summarize(found, project_id)
        if found.threshold_applied and found.threshold == 0.0:
            warnings.append(PLACEHOLDER_THRESHOLD_WARNING)

        # defence in depth: project isolation is enforced by retrieval, but never trust blindly
        usable = [e for e in found.evidence if e.project_id == project_id]
        if len(usable) != len(found.evidence):
            log.error("Dropped %d evidence item(s) from another project", len(found.evidence) - len(usable))
            warnings.append("Evidence belonging to a different project was discarded.")

        # ---- 2. retrieval abstention: no LLM call
        if found.abstained or not usable:
            reason = ("no_evidence_retrieved" if not found.ranked
                      else "below_relevance_threshold" if found.threshold_applied
                      else "no_evidence_retrieved")
            timings["total_ms"] = _ms(started)
            log.info("Ask project=%d status=insufficient_evidence reason=%s", project_id, reason)
            return AskResult(
                project_id, question, AskStatus.INSUFFICIENT_EVIDENCE, answer=INSUFFICIENT_MESSAGE,
                abstained=True, abstention_reason=reason, retrieval=summary, warnings=warnings,
                timings_ms=timings,
            )

        # ---- 3. grounded prompt
        bundle = build_prompt(question, usable, max_items=cfg.max_evidence_items, max_chars=cfg.max_evidence_chars)
        shown = [AskEvidence(i.id, i.evidence, i.truncated, i.sanitized) for i in bundle.items]
        dropped_labels = [e.citation.label for e in bundle.dropped]
        warnings += _prompt_warnings(bundle, self._settings)
        stats = {"evidence_items": len(bundle.items), "evidence_chars": bundle.evidence_chars,
                 "approx_prompt_tokens": bundle.approx_tokens}

        def result(status: AskStatus, answer: str, **extra) -> AskResult:
            timings["total_ms"] = _ms(started)
            return AskResult(
                project_id, question, status, answer=answer, evidence=shown,
                dropped_evidence=dropped_labels, retrieval=summary, warnings=warnings,
                prompt_stats=stats, timings_ms=timings, **extra,
            )

        # ---- 4. generation
        t = time.perf_counter()
        llm = None
        try:
            llm = self._resolve_llm()
            generation = llm.generate(prompt=bundle.user_prompt, system_prompt=bundle.system_prompt)
        except Exception as exc:  # noqa: BLE001 - every backend failure becomes a structured result
            kind = classify_llm_error(exc)
            if kind == "unexpected":
                log.exception("Unexpected LLM failure")
            else:
                log.warning("LLM failure (%s): %s", kind, exc)
            timings["generation_ms"] = _ms(t)
            message = str(exc) if kind != "unexpected" else f"{type(exc).__name__}: {exc}"
            return result(
                AskStatus.GENERATION_ERROR, "", error=message, error_kind=kind,
                model_name=getattr(llm, "model_name", None), backend=getattr(llm, "name", None),
                is_stub_model=bool(getattr(llm, "is_stub", False)),
            )
        timings["generation_ms"] = _ms(t)
        model_info = {"model_name": generation.model, "backend": generation.backend,
                      "is_stub_model": bool(getattr(llm, "is_stub", False)),
                      "finish_reason": generation.finish_reason}

        raw = _CONTROL.sub("", generation.text or "").strip()
        if not raw:
            return result(AskStatus.GENERATION_ERROR, "", error="The model returned an empty answer.",
                          error_kind="empty_output", **model_info)
        if generation.finish_reason == "length":
            warnings.append("The model stopped at the maximum token limit; the answer may be cut off.")
        elif generation.finish_reason == "time_limit":
            warnings.append("Generation stopped at the time limit; the answer may be incomplete.")

        # ---- 5. model-declared insufficiency
        declared = _SENTINEL_RE.match(raw)
        if declared:
            note = raw[declared.end():].strip()
            analysis = analyze_citations(note, {i.id for i in bundle.items}) if note else None
            answer = INSUFFICIENT_MESSAGE + (f"\n\nModel note: {analysis.sanitized_text}" if analysis else "")
            return result(
                AskStatus.INSUFFICIENT_EVIDENCE, answer, abstained=True,
                abstention_reason="model_declared_insufficient", raw_answer=raw,
                citations=_citations(analysis, bundle) if analysis else [],
                invalid_citation_ids=analysis.invalid_ids if analysis else [], **model_info,
            )

        # ---- 6. citation validation (deterministic; not claim verification)
        analysis = analyze_citations(raw, {i.id for i in bundle.items})
        if analysis.invalid_ids and analysis.valid_ids:
            grounding = GroundingLevel.CITED_WITH_INVALID_REFERENCES
        elif analysis.valid_ids:
            grounding = GroundingLevel.CITED
        else:
            grounding = GroundingLevel.UNCITED
        if analysis.invalid_ids:
            warnings.append("The answer cited unknown evidence IDs (" + ", ".join(analysis.invalid_ids)
                            + "); they were marked as unknown sources, not replaced.")
        if grounding == GroundingLevel.UNCITED:
            warnings.append("The answer contains no valid citations, so it must not be treated as grounded.")
        log.info("Ask project=%d status=answered grounding=%s evidence=%d", project_id, grounding.value, len(bundle.items))
        return result(
            AskStatus.ANSWERED, analysis.sanitized_text, raw_answer=raw, grounding=grounding,
            citations=_citations(analysis, bundle), invalid_citation_ids=analysis.invalid_ids, **model_info,
        )

    # --------------------------------------------------------------- internals
    def _validate_question(self, question: object) -> str:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuestionError("The question must not be empty.")
        text = question.strip()
        if "\x00" in text:
            raise InvalidQuestionError("The question contains a NUL character.")
        limit = self._settings.ask.max_question_chars
        if len(text) > limit:
            raise InvalidQuestionError(
                f"The question is {len(text)} characters; the limit is {limit} "
                "(TRUSTDOC_ASK_MAX_QUESTION_CHARS). It is not truncated automatically."
            )
        return text

    def _resolve_llm(self) -> LLMBackend:
        if self._llm is None:
            source = self._llm_source
            self._llm = source() if callable(source) and not isinstance(source, LLMBackend) else create_llm(self._settings)
        return self._llm


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _summarize(found: SearchResult, project_id: int) -> RetrievalSummary:
    """Summary for the result. Only this project's candidates are ever listed (no foreign labels)."""
    ranked = [e for e in found.ranked if e.project_id == project_id]
    evidence = [e for e in found.evidence if e.project_id == project_id]
    return RetrievalSummary(
        mode=found.mode.value, abstained=found.abstained, n_ranked=len(ranked),
        n_evidence=len(evidence), best_score=found.best_score, threshold=found.threshold,
        threshold_applied=found.threshold_applied,
        top_candidates=[{"label": e.citation.label, "rerank_score": e.rerank_score} for e in ranked[:3]],
    )


def _prompt_warnings(bundle: PromptBundle, settings: Settings) -> list[str]:
    out = []
    for item in bundle.items:
        if item.truncated:
            out.append(f"Evidence {item.id} was truncated to fit the prompt budget ({item.omitted_chars} characters omitted).")
        if item.sanitized:
            out.append(f"Evidence {item.id} contained text resembling prompt delimiters or citation IDs; it was neutralised.")
    if bundle.dropped:
        out.append(f"{len(bundle.dropped)} retrieved evidence item(s) did not fit the prompt budget and were not shown to the model.")
    if bundle.approx_tokens + settings.llm.max_tokens > settings.llm.context_tokens:
        out.append("The prompt may not fit the configured context window; lower TRUSTDOC_ASK_MAX_EVIDENCE_CHARS.")
    return out


def _citations(analysis, bundle: PromptBundle) -> list[AskCitation]:
    by_id = {i.id: i.evidence for i in bundle.items}
    out = []
    for cid in analysis.valid_ids:
        e = by_id[cid]
        c = e.citation
        out.append(AskCitation(
            id=cid, label=c.label, source_name=c.source_name, source_type=c.source_type,
            source_id=c.source_id, chunk_id=c.chunk_id, mentions=analysis.mentions.get(cid, 1),
            start_line=c.start_line, end_line=c.end_line, page_number=c.page_number,
            heading=c.heading, qualified_name=c.qualified_name,
        ))
    return out


def create_ask_service(db: Database, settings: Settings, *, retriever=None, llm: LLMSource = None) -> AskService:
    """Default wiring: the configured Part 3 retriever and the configured LLM (both lazy)."""
    if retriever is None:
        from trustdoc.retrieval.factory import create_retriever  # imports numpy/faiss lazily

        retriever = create_retriever(db, settings)
    return AskService(db, settings, retriever, llm)
