"""The documentation generation workflow (Part 6).

Generates grounded README or Python API documentation using retrieved project evidence,
validates citations, and passes the generated Markdown through the Part 5 VerificationEngine.

Outcomes (``GenerationResult.status``):
* ``completed``             the model generated documentation and it was verified
* ``insufficient_evidence`` retrieval abstained or found no usable evidence for the project/entity
* ``retrieval_error`` / ``generation_error``  operational failures; no documentation is fabricated

Citations are checked deterministically against supplied evidence IDs, and the resulting Markdown
is verified against the project's code entities and claims via VerificationEngine.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Callable, Sequence, Union

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import (
    DocumentSelectionError,
    InvalidQuestionError,
    LLMError,
    MissingDependencyError,
    NotFoundError,
    RetrievalError,
    VerificationError,
)
from trustdoc.llm.base import LLMBackend
from trustdoc.llm.factory import create_llm
from trustdoc.logging_setup import get_logger
from trustdoc.models import CodeEntity, EntityType, Source
from trustdoc.retrieval.types import Citation, Evidence, RetrievalMode
from trustdoc.verification.resolver import EntityResolver, ResolutionKind

if TYPE_CHECKING:
    from trustdoc.verification.engine import VerificationEngine
from trustdoc.workflows.ask import classify_llm_error
from trustdoc.workflows.citations import analyze_citations
from trustdoc.workflows.prompts import (
    build_api_prompt,
    build_readme_prompt,
)
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    GenerationKind,
    GenerationResult,
    GenerationStatus,
    GroundingLevel,
)

log = get_logger("generation")

LLMSource = Union[LLMBackend, Callable[[], LLMBackend], None]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _build_ask_citations(
    analysis,
    evidence_items,
) -> tuple[list[AskCitation], GroundingLevel]:
    """Map valid cited IDs to AskCitation records with source metadata."""
    item_by_id = {item.id: item for item in evidence_items}
    citations: list[AskCitation] = []
    for cid in analysis.valid_ids:
        item = item_by_id.get(cid)
        if not item:
            continue
        ev = item.evidence
        cit = ev.citation
        citations.append(
            AskCitation(
                id=cid,
                label=cit.label,
                source_name=ev.source_name,
                source_type=ev.source_type.value if hasattr(ev.source_type, "value") else str(ev.source_type),
                source_id=ev.source_id,
                chunk_id=ev.chunk_id,
                mentions=analysis.mentions.get(cid, 1),
                start_line=cit.start_line,
                end_line=cit.end_line,
                page_number=cit.page_number,
                heading=cit.heading,
                qualified_name=cit.qualified_name,
            )
        )

    if analysis.invalid_ids and citations:
        grounding = GroundingLevel.CITED_WITH_INVALID_REFERENCES
    elif citations:
        grounding = GroundingLevel.CITED
    else:
        grounding = GroundingLevel.UNCITED

    return citations, grounding


class GenerationService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        retriever,
        llm: LLMSource = None,
        *,
        verification_engine: VerificationEngine | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._retriever = retriever
        self._llm_source = llm
        self._llm: LLMBackend | None = llm if isinstance(llm, LLMBackend) else None
        if verification_engine is None:
            from trustdoc.verification.engine import VerificationEngine

            self._verifier = VerificationEngine(db, settings, retriever=retriever, llm=llm)
        else:
            self._verifier = verification_engine

    def _get_llm(self) -> LLMBackend:
        if self._llm is None:
            if callable(self._llm_source):
                self._llm = self._llm_source()
            else:
                self._llm = create_llm(self._settings)
        return self._llm

    # ------------------------------------------------------------------ public
    def generate(
        self,
        project_id: int,
        kind: GenerationKind | str,
        *,
        entity: str | None = None,
        top_k: int | None = None,
        semantic: bool = False,
        run_verify: bool = True,
    ) -> GenerationResult:
        """Generate either README or API documentation."""
        kind = GenerationKind(kind)
        if kind == GenerationKind.README:
            return self.generate_readme(project_id, top_k=top_k, semantic=semantic, run_verify=run_verify)
        if kind == GenerationKind.API:
            if not entity or not entity.strip():
                raise InvalidQuestionError("API documentation generation requires an entity identifier (--entity).")
            return self.generate_api(project_id, entity.strip(), top_k=top_k, semantic=semantic, run_verify=run_verify)
        raise ValueError(f"Unknown generation kind {kind!r}")

    def generate_readme(
        self,
        project_id: int,
        *,
        top_k: int | None = None,
        semantic: bool = False,
        run_verify: bool = True,
    ) -> GenerationResult:
        """Generate project README documentation grounded in retrieved evidence."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        with self._db.session() as s:
            project = s.projects.require(project_id)
            sources = s.sources.list_for_project(project_id)

        if not sources:
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.README,
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                markdown="",
                warnings=["Project has no ingested sources."],
                timings_ms=timings,
            )

        cfg = self._settings.generate
        k = cfg.max_evidence_items if top_k is None else top_k

        # 1. Multi-angle retrieval for README
        t_ret = time.perf_counter()
        evidence_map: dict[int, Evidence] = {}
        queries = [
            "project overview purpose architecture",
            "installation setup configuration dependencies",
            "usage quickstart examples commands entrypoints",
        ]
        try:
            for q in queries:
                res = self._retriever.search(project_id, q, top_k=k)
                for ev in res.evidence:
                    if ev.project_id == project_id and ev.chunk_id not in evidence_map:
                        evidence_map[ev.chunk_id] = ev
        except (RetrievalError, MissingDependencyError) as exc:
            log.warning("Retrieval failed during README generation for project %d: %s", project_id, exc)
            timings["retrieval_ms"] = _ms(t_ret)
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.README,
                status=GenerationStatus.RETRIEVAL_ERROR,
                markdown="",
                error=str(exc),
                error_kind="dependency_missing" if isinstance(exc, MissingDependencyError) else "retrieval_failed",
                timings_ms=timings,
            )
        timings["retrieval_ms"] = _ms(t_ret)

        evidence_items = list(evidence_map.values())
        if not evidence_items:
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.README,
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                markdown="",
                warnings=["No relevant evidence could be retrieved for README generation."],
                timings_ms=timings,
            )

        # 2. Build prompt
        bundle = build_readme_prompt(
            project.name, evidence_items, max_items=cfg.max_evidence_items, max_chars=cfg.max_evidence_chars
        )
        shown_evidence = [AskEvidence(i.id, i.evidence, i.truncated, i.sanitized) for i in bundle.items]
        dropped_labels = [e.citation.label for e in bundle.dropped]
        prompt_stats = {
            "evidence_items": len(bundle.items),
            "evidence_chars": bundle.evidence_chars,
            "dropped_items": len(bundle.dropped),
            "approx_tokens": bundle.approx_tokens,
        }

        # 3. LLM generation
        t_gen = time.perf_counter()
        try:
            llm = self._get_llm()
            gen_res = llm.generate(bundle.user_prompt, bundle.system_prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generation failed for project %d: %s", project_id, exc)
            timings["generation_ms"] = _ms(t_gen)
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.README,
                status=GenerationStatus.GENERATION_ERROR,
                markdown="",
                evidence=shown_evidence,
                dropped_evidence=dropped_labels,
                error=str(exc),
                error_kind=classify_llm_error(exc),
                prompt_stats=prompt_stats,
                timings_ms=timings,
            )
        timings["generation_ms"] = _ms(t_gen)

        # 4. Citations analysis
        analysis = analyze_citations(gen_res.text, [i.id for i in bundle.items])
        citations, grounding = _build_ask_citations(analysis, bundle.items)
        if analysis.invalid_ids:
            warnings.append(
                f"Generated documentation cited {len(analysis.invalid_ids)} unknown evidence ID(s): "
                + ", ".join(analysis.invalid_ids)
            )

        markdown = analysis.sanitized_text

        # 5. Verification integration
        verification_report = None
        verification_error: str | None = None
        if run_verify:
            t_ver = time.perf_counter()
            try:
                doc_name = f"generated/README-{hashlib.sha256(markdown.encode('utf-8')).hexdigest()[:8]}.md"
                verification_report = self._verifier.verify_text(
                    project_id, markdown, name=doc_name, semantic=semantic
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Verification failed for generated README in project %d: %s", project_id, exc)
                verification_error = str(exc)
                warnings.append(f"Verification could not be completed: {exc}")
            timings["verification_ms"] = _ms(t_ver)

        timings["total_ms"] = _ms(started)

        status = GenerationStatus.VERIFICATION_ERROR if verification_error else GenerationStatus.COMPLETED

        return GenerationResult(
            project_id=project_id,
            kind=GenerationKind.README,
            status=status,
            markdown=markdown,
            grounding=grounding,
            citations=citations,
            invalid_citation_ids=analysis.invalid_ids,
            evidence=shown_evidence,
            dropped_evidence=dropped_labels,
            verification_report=verification_report,
            raw_output=gen_res.text,
            model_name=gen_res.model,
            backend=gen_res.backend,
            is_stub_model=getattr(llm, "is_stub", False),
            finish_reason=gen_res.finish_reason,
            error=verification_error,
            error_kind="verification_failed" if verification_error else None,
            warnings=warnings,
            timings_ms=timings,
            prompt_stats=prompt_stats,
        )

    def generate_api(
        self,
        project_id: int,
        entity_name: str,
        *,
        top_k: int | None = None,
        semantic: bool = False,
        run_verify: bool = True,
    ) -> GenerationResult:
        """Generate API documentation for a specific Python class, function, or method."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        # 1. Resolve entity from AST index
        with self._db.session() as s:
            project = s.projects.require(project_id)
            entities = s.entities.list_for_project(project_id)
            sources = s.sources.list_for_project(project_id)
            source_names = {src.id: src.name for src in sources}

        resolver = EntityResolver(entities)
        resolution = resolver.resolve(entity_name)

        if resolution.kind == ResolutionKind.NOT_FOUND:
            raise NotFoundError(f"Entity {entity_name!r} not found in project {project_id}: {resolution.reason}")
        if resolution.kind == ResolutionKind.AMBIGUOUS:
            raise InvalidQuestionError(f"Ambiguous entity name {entity_name!r}: {resolution.reason}")
        if resolution.kind == ResolutionKind.INVALID:
            raise InvalidQuestionError(f"Invalid entity identifier {entity_name!r}: {resolution.reason}")
        if resolution.kind in (ResolutionKind.EXTERNAL, ResolutionKind.ALIAS):
            raise NotFoundError(f"Cannot generate API documentation for {resolution.kind.value} entity {entity_name!r}: {resolution.reason}")

        target_entity = resolution.entity
        assert target_entity is not None

        constructor = None
        if target_entity.entity_type == EntityType.CLASS:
            constructor = resolver.init_of(target_entity)

        cfg = self._settings.generate
        k = cfg.max_evidence_items if top_k is None else top_k

        # 2. Evidence gathering: direct source chunks for this entity + hybrid retrieval
        t_ret = time.perf_counter()
        evidence_items: list[Evidence] = []
        seen_chunk_ids: set[int] = set()

        # (a) Direct chunks from the entity's source
        with self._db.session() as s:
            source_chunks = s.chunks.list_for_source(target_entity.source_id)
            source_rec = s.sources.get(target_entity.source_id)

        source_name = source_rec.name if source_rec else "source.py"
        source_type = source_rec.source_type if source_rec else "python"

        direct_chunks = [
            c for c in source_chunks
            if c.entity_id == target_entity.id
            or (constructor is not None and c.entity_id == constructor.id)
            or (c.start_line is not None and target_entity.start_line is not None
                and c.start_line <= (target_entity.end_line or target_entity.start_line)
                and (c.end_line or c.start_line) >= target_entity.start_line)
        ]
        for c in direct_chunks:
            if c.id is not None and c.id not in seen_chunk_ids:
                seen_chunk_ids.add(c.id)
                stype_str = source_type.value if hasattr(source_type, "value") else str(source_type)
                ctype_str = c.chunk_type.value if hasattr(c.chunk_type, "value") else str(c.chunk_type)
                evidence_items.append(
                    Evidence(
                        chunk_id=c.id,
                        source_id=c.source_id,
                        project_id=project_id,
                        text=c.text,
                        chunk_type=ctype_str,
                        source_name=source_name,
                        source_type=stype_str,
                        citation=Citation(
                            label=c.location_label(source_name),
                            source_name=source_name,
                            source_type=stype_str,
                            source_id=c.source_id,
                            chunk_id=c.id,
                            start_line=c.start_line,
                            end_line=c.end_line,
                            page_number=c.page_number,
                            heading=c.heading,
                            qualified_name=target_entity.qualified_name,
                        ),
                        method=RetrievalMode.HYBRID_RERANK,
                        entity_id=c.entity_id,
                        rerank_score=1.0,
                    )
                )

        # (b) Retrieved chunks from the retriever for context, docstrings, or usage
        try:
            query = f"{target_entity.name} {target_entity.qualified_name}"
            res = self._retriever.search(project_id, query, top_k=k)
            for ev in res.evidence:
                if ev.project_id == project_id and ev.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(ev.chunk_id)
                    evidence_items.append(ev)
        except (RetrievalError, MissingDependencyError) as exc:
            log.warning("Retrieval failed during API generation for %s: %s", target_entity.qualified_name, exc)
            if not evidence_items:
                timings["retrieval_ms"] = _ms(t_ret)
                timings["total_ms"] = _ms(started)
                return GenerationResult(
                    project_id=project_id,
                    kind=GenerationKind.API,
                    status=GenerationStatus.RETRIEVAL_ERROR,
                    markdown="",
                    target_entity=target_entity.qualified_name,
                    error=str(exc),
                    error_kind="dependency_missing" if isinstance(exc, MissingDependencyError) else "retrieval_failed",
                    timings_ms=timings,
                )
            warnings.append(f"Additional retrieval failed ({type(exc).__name__}); proceeding with direct AST source evidence.")
        timings["retrieval_ms"] = _ms(t_ret)

        if not evidence_items:
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.API,
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                markdown="",
                target_entity=target_entity.qualified_name,
                warnings=[f"No code or documentation evidence found for entity {target_entity.qualified_name}."],
                timings_ms=timings,
            )

        # 3. Build API prompt
        bundle = build_api_prompt(
            target_entity,
            evidence_items,
            constructor=constructor,
            max_items=cfg.max_evidence_items,
            max_chars=cfg.max_evidence_chars,
        )
        shown_evidence = [AskEvidence(i.id, i.evidence, i.truncated, i.sanitized) for i in bundle.items]
        dropped_labels = [e.citation.label for e in bundle.dropped]
        prompt_stats = {
            "evidence_items": len(bundle.items),
            "evidence_chars": bundle.evidence_chars,
            "dropped_items": len(bundle.dropped),
            "approx_tokens": bundle.approx_tokens,
        }

        # 4. LLM generation
        t_gen = time.perf_counter()
        try:
            llm = self._get_llm()
            gen_res = llm.generate(bundle.user_prompt, bundle.system_prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generation failed for %s: %s", target_entity.qualified_name, exc)
            timings["generation_ms"] = _ms(t_gen)
            timings["total_ms"] = _ms(started)
            return GenerationResult(
                project_id=project_id,
                kind=GenerationKind.API,
                status=GenerationStatus.GENERATION_ERROR,
                markdown="",
                target_entity=target_entity.qualified_name,
                evidence=shown_evidence,
                dropped_evidence=dropped_labels,
                error=str(exc),
                error_kind=classify_llm_error(exc),
                prompt_stats=prompt_stats,
                timings_ms=timings,
            )
        timings["generation_ms"] = _ms(t_gen)

        # 5. Citations analysis
        analysis = analyze_citations(gen_res.text, [i.id for i in bundle.items])
        citations, grounding = _build_ask_citations(analysis, bundle.items)
        if analysis.invalid_ids:
            warnings.append(
                f"Generated documentation cited {len(analysis.invalid_ids)} unknown evidence ID(s): "
                + ", ".join(analysis.invalid_ids)
            )

        markdown = analysis.sanitized_text

        # 6. Verification integration
        verification_report = None
        verification_error: str | None = None
        if run_verify:
            t_ver = time.perf_counter()
            try:
                clean_name = target_entity.qualified_name.replace(".", "_")
                digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:8]
                doc_name = f"generated/api-{clean_name}-{digest}.md"
                verification_report = self._verifier.verify_text(
                    project_id, markdown, name=doc_name, semantic=semantic
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Verification failed for generated API doc %s: %s", target_entity.qualified_name, exc)
                verification_error = str(exc)
                warnings.append(f"Verification could not be completed: {exc}")
            timings["verification_ms"] = _ms(t_ver)

        timings["total_ms"] = _ms(started)

        status = GenerationStatus.VERIFICATION_ERROR if verification_error else GenerationStatus.COMPLETED

        return GenerationResult(
            project_id=project_id,
            kind=GenerationKind.API,
            status=status,
            markdown=markdown,
            target_entity=target_entity.qualified_name,
            grounding=grounding,
            citations=citations,
            invalid_citation_ids=analysis.invalid_ids,
            evidence=shown_evidence,
            dropped_evidence=dropped_labels,
            verification_report=verification_report,
            raw_output=gen_res.text,
            model_name=gen_res.model,
            backend=gen_res.backend,
            is_stub_model=getattr(llm, "is_stub", False),
            finish_reason=gen_res.finish_reason,
            error=verification_error,
            error_kind="verification_failed" if verification_error else None,
            warnings=warnings,
            timings_ms=timings,
            prompt_stats=prompt_stats,
        )


def create_generation_service(
    db: Database,
    settings: Settings,
    *,
    retriever=None,
    llm: LLMSource = None,
    verification_engine: VerificationEngine | None = None,
) -> GenerationService:
    """Wire a ready-to-use GenerationService from database and settings."""
    if retriever is None:
        from trustdoc.retrieval.factory import create_retriever  # lazy import: keeps workflows lightweight

        retriever = create_retriever(db, settings)
    return GenerationService(db, settings, retriever, llm, verification_engine=verification_engine)
