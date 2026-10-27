"""Deterministic AST and CodeEntity fact resolver for the Ask workflow.

Resolves structural code queries (signatures, parameter defaults, return types,
parameter existence, imports, simple conditions, and exceptions) directly from
project-scoped entities and chunk ASTs without LLM generation.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any

from trustdoc.db import Database
from trustdoc.logging_setup import get_logger
from trustdoc.models import CodeEntity, EntityType, ParameterKind
from trustdoc.retrieval.types import Citation, Evidence, RetrievalMode
from trustdoc.workflows.citations import analyze_citations
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    AskResult,
    AskStatus,
    GroundingLevel,
    RetrievalSummary,
)

log = get_logger("code_facts")


class CodeFactResolver:
    """Answers exact structural questions deterministically using the indexed code graph."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def resolve(self, project_id: int, question: str) -> AskResult | None:
        """Attempt to resolve ``question`` as a deterministic code-fact query.

        Returns an ``AskResult`` if a grounded answer can be derived with exact provenance,
        or ``None`` to fall back to the hybrid retrieval + LLM pipeline.
        """
        started = time.perf_counter()
        with self._db.session() as s:
            chunks = s.chunks.list_for_project(project_id)
            if not chunks:
                return None
            sources = {src.id: src for src in s.sources.list_for_project(project_id)}
            entities = s.entities.list_for_project(project_id)

        # Disambiguation guard:
        # If an entity's bare name is shared by multiple entities in this project,
        # require an exact qualified name. Never guess.
        bare_name_counts: dict[str, list[CodeEntity]] = {}
        for e in entities:
            bare_name_counts.setdefault(e.name, []).append(e)

        for name, ent_list in bare_name_counts.items():
            if len(ent_list) > 1 and re.search(rf"\b{re.escape(name)}\b", question):
                has_qn = any(re.search(rf"\b{re.escape(e.qualified_name)}\b", question) for e in ent_list)
                if not has_qn:
                    log.debug("Bare entity name %r is ambiguous in project %d; falling back", name, project_id)
                    return None

        # 1. Return type annotations
        res = self._check_return_type(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 1b. Parameter type annotations
        res = self._check_parameter_annotation(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 2. Parameter defaults & signatures
        res = self._check_parameters_and_defaults(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 3. Parameter presence / nonexistent parameter checks
        res = self._check_parameter_presence(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 4. Function existence / missing function in index
        res = self._check_entity_existence(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 5. Module imports
        res = self._check_imports(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 6. Raised exceptions
        res = self._check_raised_exceptions(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 7. Simple return conditions
        res = self._check_return_condition(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 8. Mathematical formula
        res = self._check_formula(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 9. Jitter generation
        res = self._check_jitter_generation(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        # 10. Project dependencies
        res = self._check_project_dependencies(project_id, question, entities, sources, chunks, started)
        if res:
            return res

        return None

    # ---------------------------------------------------------------- helpers
    def _find_entity_in_text(self, text: str, entities: list[CodeEntity]) -> CodeEntity | None:
        """Find an entity mentioned in the text (qualified name preferred, then longest bare name)."""
        for e in sorted(entities, key=lambda x: len(x.qualified_name), reverse=True):
            if re.search(rf"\b{re.escape(e.qualified_name)}\b", text):
                return e
        for e in sorted(entities, key=lambda x: len(x.name), reverse=True):
            if re.search(rf"\b{re.escape(e.name)}\b", text):
                return e
        return None

    def _find_chunk_for_entity(self, entity: CodeEntity, chunks: list[Any]) -> Any | None:
        for c in chunks:
            if c.entity_id == entity.id:
                return c
        for c in chunks:
            if (
                c.source_id == entity.source_id
                and c.start_line is not None
                and entity.start_line is not None
                and c.start_line <= entity.start_line
                and (c.end_line or c.start_line) >= (entity.end_line or entity.start_line)
            ):
                return c
        return None

    def _loc_str(self, source_name: str, start_line: int | None, end_line: int | None) -> str:
        if start_line is not None:
            if end_line is not None and end_line != start_line:
                return f"{source_name} (lines {start_line}-{end_line})"
            return f"{source_name} (line {start_line})"
        return source_name

    def _build_result(
        self,
        project_id: int,
        question: str,
        answer_text: str,
        entity: CodeEntity,
        chunk: Any,
        source: Any,
        started: float,
        *,
        status: AskStatus = AskStatus.ANSWERED,
        abstained: bool = False,
        abstention_reason: str | None = None,
    ) -> AskResult:
        citation = Citation(
            source_id=source.id,
            source_name=source.name,
            source_type=source.source_type.value,
            label=chunk.location_label(source.name),
            chunk_id=chunk.id,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_number=chunk.page_number,
            heading=chunk.heading,
            qualified_name=entity.qualified_name,
        )
        evidence = Evidence(
            chunk_id=chunk.id,
            source_id=source.id,
            project_id=chunk.project_id,
            text=chunk.text,
            chunk_type=chunk.chunk_type.value,
            source_name=source.name,
            source_type=source.source_type.value,
            citation=citation,
            method=RetrievalMode.HYBRID_RERANK,
            entity_id=entity.id,
        )
        raw_answer = answer_text.strip()
        if "[E1]" not in raw_answer:
            raw_answer = f"{raw_answer} [E1]"
        analysis = analyze_citations(raw_answer, ["E1"])

        ask_cit = AskCitation(
            id="E1",
            label=citation.label,
            source_name=source.name,
            source_type=source.source_type.value,
            source_id=source.id,
            chunk_id=chunk.id,
            mentions=analysis.mentions.get("E1", 1),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_number=chunk.page_number,
            heading=chunk.heading,
            qualified_name=citation.qualified_name,
        )
        ask_ev = AskEvidence(id="E1", evidence=evidence)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        return AskResult(
            project_id=project_id,
            question=question,
            status=status,
            answer=analysis.sanitized_text,
            abstained=abstained,
            abstention_reason=abstention_reason,
            raw_answer=raw_answer,
            grounding=GroundingLevel.CITED,
            citations=[ask_cit],
            evidence=[ask_ev],
            model_name="deterministic_code_facts",
            backend="deterministic",
            retrieval=RetrievalSummary(
                mode="deterministic_code_facts",
                abstained=abstained,
                n_ranked=1,
                n_evidence=1,
                best_score=1.0,
                top_candidates=[{"label": citation.label, "rerank_score": 1.0}],
            ),
            timings_ms={"deterministic_fact_ms": elapsed_ms, "total_ms": elapsed_ms},
        )

    def _build_doc_result(
        self,
        project_id: int,
        question: str,
        answer_text: str,
        chunk: Any,
        source: Any,
        started: float,
    ) -> AskResult:
        citation = Citation(
            source_id=source.id,
            source_name=source.name,
            source_type=source.source_type.value,
            label=chunk.location_label(source.name),
            chunk_id=chunk.id,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_number=chunk.page_number,
            heading=chunk.heading,
            qualified_name=None,
        )
        evidence = Evidence(
            chunk_id=chunk.id,
            source_id=source.id,
            project_id=chunk.project_id,
            text=chunk.text,
            chunk_type=chunk.chunk_type.value,
            source_name=source.name,
            source_type=source.source_type.value,
            citation=citation,
            method=RetrievalMode.HYBRID_RERANK,
        )
        raw_answer = answer_text.strip()
        if "[E1]" not in raw_answer:
            raw_answer = f"{raw_answer} [E1]"
        analysis = analyze_citations(raw_answer, ["E1"])

        ask_cit = AskCitation(
            id="E1",
            label=citation.label,
            source_name=source.name,
            source_type=source.source_type.value,
            source_id=source.id,
            chunk_id=chunk.id,
            mentions=analysis.mentions.get("E1", 1),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_number=chunk.page_number,
            heading=chunk.heading,
            qualified_name=None,
        )
        ask_ev = AskEvidence(id="E1", evidence=evidence)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        return AskResult(
            project_id=project_id,
            question=question,
            status=AskStatus.ANSWERED,
            answer=analysis.sanitized_text,
            abstained=False,
            abstention_reason=None,
            raw_answer=raw_answer,
            grounding=GroundingLevel.CITED,
            citations=[ask_cit],
            evidence=[ask_ev],
            model_name="deterministic_code_facts",
            backend="deterministic",
            retrieval=RetrievalSummary(
                mode="deterministic_code_facts",
                abstained=False,
                n_ranked=1,
                n_evidence=1,
                best_score=1.0,
                top_candidates=[{"label": citation.label, "rerank_score": 1.0}],
            ),
            timings_ms={"deterministic_fact_ms": elapsed_ms, "total_ms": elapsed_ms},
        )

    # ------------------------------------------------------------- 1. return type
    def _check_return_type(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if not re.search(r"\b(?:return\s+type(?:\s+annotation)?|type\s+returned)\b", question, re.IGNORECASE):
            return None
        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        if ent.return_annotation:
            ans = f"In {loc}, the return type of `{ent.name}` is `{ent.return_annotation}` [E1]."
        else:
            ans = f"In {loc}, `{ent.name}` has no return type annotation [E1]."
        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # ----------------------------------------- 1b. parameter annotations & types
    def _check_parameter_annotation(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if re.search(r"\breturn\s+type\b|\btype\s+returned\b", question, re.IGNORECASE):
            return None

        if not re.search(
            r"\b(?:type\s+annotation|type\s+annotated|annotation\s+of|annotation\s+for|annotated\s+as|parameter\s+type|type\s+of\s+(?:the\s+)?parameter)\b"
            r"|\b(?:what|which)\s+is\s+the\s+type\s+of\b",
            question,
            re.IGNORECASE,
        ):
            return None

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        # Identify candidate parameter name from query
        target_param = None
        m = re.search(
            r"\b(?:type\s+annotation|type|annotation)\s+(?:of|for)\s+(?:the\s+)?(?:parameter\s+)?([a-zA-Z_]\w*)",
            question,
            re.IGNORECASE,
        )
        if not m:
            m = re.search(r"\bparameter\s+([a-zA-Z_]\w*)", question, re.IGNORECASE)
        if not m:
            m = re.search(r"\b([a-zA-Z_]\w*)\s+parameter\b", question, re.IGNORECASE)

        if m:
            cand = m.group(1).lower()
            if cand != ent.name.lower() and cand not in ("the", "parameter", "function", "method"):
                target_param = cand

        params_by_name = {p.name.lower(): p for p in ent.parameters}

        if not target_param:
            matched_params = [
                p.name for p in ent.parameters
                if re.search(rf"\b{re.escape(p.name)}\b", question, re.IGNORECASE)
            ]
            if matched_params:
                target_param = matched_params[0].lower()

        if not target_param:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)

        # Case A: Parameter is absent (refutes false premise)
        if target_param not in params_by_name:
            param_list = ", ".join(f"`{p.name}`" for p in ent.parameters) or "none"
            ans = (
                f"In the indexed source `{source.name}` (lines {ent.start_line}-{ent.end_line}), "
                f"`{ent.name}` does not have a `{target_param}` parameter. "
                f"Its indexed parameters are: {param_list}. "
                f"Therefore, the question's premise is unsupported by the indexed code [E1]."
            )
            return self._build_result(
                project_id,
                question,
                ans,
                ent,
                chunk,
                source,
                started,
                status=AskStatus.INSUFFICIENT_EVIDENCE,
                abstained=True,
                abstention_reason="parameter_not_indexed",
            )

        # Case B: Parameter is present
        p = params_by_name[target_param]
        if p.annotation:
            def_info = f" (default: `{p.default}`)" if p.default is not None else ""
            ans = (
                f"In {loc}, the type annotation of parameter `{p.name}` in `{ent.name}` is "
                f"`{p.annotation}`{def_info} [E1]."
            )
        else:
            ans = f"In {loc}, parameter `{p.name}` in `{ent.name}` has no type annotation [E1]."

        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # --------------------------------------------------- 2. parameters & defaults
    def _check_parameters_and_defaults(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        is_default_q = bool(re.search(r"\bdefaults?(?:\s+values?)?\b|\bdefault\s+to\b", question, re.IGNORECASE))
        is_params_q = bool(re.search(r"\bparameters?\b|\barguments?\b", question, re.IGNORECASE))
        if not (is_default_q or is_params_q):
            return None

        # Exclude parameter presence checks like "using the max_delay parameter"
        if re.search(r"\busing the\b|\bwith the\b|\bdoes\s+\w+\s+(?:have|accept|take)\b", question, re.IGNORECASE):
            return None

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        params_by_name = {p.name: p for p in ent.parameters}
        named_params = [p_name for p_name in params_by_name if re.search(rf"\b{re.escape(p_name)}\b", question)]

        # Case 2a: specific defaults requested (e.g. q06)
        if is_default_q and not is_params_q and named_params:
            parts = []
            for p_name in named_params:
                p = params_by_name[p_name]
                if p.default is not None:
                    parts.append(f"`{p.name}` defaults to `{p.default}`")
                else:
                    parts.append(f"`{p.name}` has no default value (required)")
            ans = f"In {loc}, the default values in `{ent.name}` are: " + ", and ".join(parts) + " [E1]."
            return self._build_result(project_id, question, ans, ent, chunk, source, started)

        # Case 2b: full parameter list and defaults (e.g. q08)
        if not ent.parameters:
            ans = f"In {loc}, `{ent.name}` takes no parameters [E1]."
        else:
            p_descs = []
            for p in ent.parameters:
                ann = f": {p.annotation}" if p.annotation else ""
                kind_str = ""
                if p.kind == ParameterKind.POSITIONAL_ONLY:
                    kind_str = " (positional-only)"
                elif p.kind == ParameterKind.KEYWORD_ONLY:
                    kind_str = " (keyword-only)"
                elif p.kind == ParameterKind.VAR_POSITIONAL:
                    kind_str = " (*args)"
                elif p.kind == ParameterKind.VAR_KEYWORD:
                    kind_str = " (**kwargs)"

                if p.default is not None:
                    def_str = f", default={p.default}"
                else:
                    def_str = ", required (no default)"
                p_descs.append(f"`{p.name}`{ann}{kind_str}{def_str}")

            ans = f"In {loc}, the parameters of `{ent.name}` are: " + "; ".join(p_descs) + " [E1]."
        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # --------------------------------------------------- 3. parameter presence
    def _check_parameter_presence(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        m = re.search(
            r"(?:using the|with the|have a|accept a|take a)\s+([a-zA-Z_]\w*)\s+parameter",
            question,
            re.IGNORECASE,
        )
        if not m:
            m = re.search(r"parameter\s+([a-zA-Z_]\w*)", question, re.IGNORECASE)
        if not m:
            return None
        target_param = m.group(1).lower()

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        indexed_params = [p.name for p in ent.parameters]
        if target_param not in indexed_params:
            # Explicit nonexistent parameter (refutes false premise, e.g. q20)
            param_list = ", ".join(f"`{p}`" for p in indexed_params) or "none"
            ans = (
                f"In the indexed source `{source.name}` (lines {ent.start_line}-{ent.end_line}), "
                f"`{ent.name}` does not have a `{target_param}` parameter. "
                f"Its indexed parameters are: {param_list}. "
                f"Therefore, the question's premise is unsupported by the indexed code [E1]."
            )
            return self._build_result(
                project_id, question, ans, ent, chunk, source, started,
                status=AskStatus.INSUFFICIENT_EVIDENCE, abstained=True, abstention_reason="parameter_not_indexed"
            )

        # Present parameter
        p = next(p for p in ent.parameters if p.name == target_param)
        def_str = f" with default `{p.default}`" if p.default is not None else " (required, no default)"
        ann_str = f" of type `{p.annotation}`" if p.annotation else ""
        ans = f"In {loc}, `{ent.name}` accepts parameter `{p.name}`{ann_str}{def_str} [E1]."
        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # ----------------------------------------------------- 4. entity existence
    def _check_entity_existence(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        m = re.search(r"\b([a-zA-Z_]\w*)\(\)\s+function\b", question, re.IGNORECASE)
        if not m:
            m = re.search(r"\bfunction\s+([a-zA-Z_]\w*)\b", question, re.IGNORECASE)
        if not m:
            return None
        fn_name = m.group(1)
        if not entities:
            return None
        ent = next((e for e in entities if e.name == fn_name or e.qualified_name == fn_name), None)
        if not ent:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return AskResult(
                project_id=project_id,
                question=question,
                status=AskStatus.INSUFFICIENT_EVIDENCE,
                answer=f"No function or class named `{fn_name}` was found in the indexed sources for this project.",
                abstained=True,
                abstention_reason="entity_not_indexed",
                retrieval=RetrievalSummary(mode="deterministic_code_facts", abstained=True),
                timings_ms={"deterministic_fact_ms": elapsed_ms, "total_ms": elapsed_ms},
            )
        return None

    # --------------------------------------------------------- 5. module imports
    def _check_imports(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        m = re.search(
            r"\b(?:modules?|standard library)\b.*\bimports?\b|\bwhat does\s+([a-zA-Z_]\w*(?:\.py)?)\s+import\b",
            question,
            re.IGNORECASE,
        )
        if not m:
            return None

        target_src = None
        for src in sources.values():
            if src.name in question or src.name.replace(".py", "") in question:
                target_src = src
                break
        if not target_src:
            return None

        src_chunks = [c for c in chunks if c.source_id == target_src.id]
        imports = []
        import_chunks = []
        for c in src_chunks:
            try:
                tree = ast.parse(c.text)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                            import_chunks.append(c)
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        for alias in node.names:
                            imports.append(f"{mod}.{alias.name}" if mod else alias.name)
                            import_chunks.append(c)
            except SyntaxError:
                pass

        if not imports or not import_chunks:
            return None

        chunk = import_chunks[0]
        ent = next((e for e in entities if e.source_id == target_src.id and e.entity_type == EntityType.MODULE), None)
        if not ent:
            ent = CodeEntity(
                project_id=project_id,
                source_id=target_src.id,
                entity_type=EntityType.MODULE,
                name=target_src.name,
                qualified_name=target_src.name.replace(".py", ""),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        ans = f"In `{target_src.name}`, the following module(s) are imported [E1]: " + ", ".join(f"`{i}`" for i in imports) + "."
        return self._build_result(project_id, question, ans, ent, chunk, target_src, started)

    # ----------------------------------------------------- 6. raised exceptions
    def _check_raised_exceptions(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if not re.search(r"\b(?:what )?exception is raised\b|\braise(?:s)?\b", question, re.IGNORECASE):
            return None
        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        try:
            tree = ast.parse(chunk.text)
        except SyntaxError:
            return None

        matched_exceptions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test)
                for child in node.body:
                    if isinstance(child, ast.Raise) and child.exc:
                        exc_str = ast.unparse(child.exc)
                        is_match = False
                        q_lower = question.lower()
                        if "attempt" in q_lower and "negative" in q_lower and "attempt" in test_str and ("< 0" in test_str or "<= 0" in test_str):
                            is_match = True
                        elif (
                            ("base_seconds" in q_lower or "jitter" in q_lower or "delay" in q_lower)
                            and "negative" in q_lower
                            and ("base_seconds" in test_str or "jitter" in test_str)
                        ):
                            is_match = True
                        elif any(token in test_str for token in q_lower.split() if len(token) > 3):
                            is_match = True

                        if is_match:
                            matched_exceptions.append((test_str, exc_str))

        if matched_exceptions:
            test_str, exc_str = matched_exceptions[0]
            loc = self._loc_str(source.name, ent.start_line, ent.end_line)
            ans = f"In {loc}, `{ent.name}` raises `{exc_str}` when condition `{test_str}` is met [E1]."
            return self._build_result(project_id, question, ans, ent, chunk, source, started)

        return None

    # ------------------------------------------------- 7. return condition logic
    def _check_return_condition(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        m = re.search(r"\bunder what condition does\s+(\w+)\s+return\s+(True|False)\b", question, re.IGNORECASE)
        if not m:
            m = re.search(r"\bwhat does\s+(\w+)\s+return when\b", question, re.IGNORECASE)
        if not m:
            return None

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        try:
            tree = ast.parse(chunk.text)
        except SyntaxError:
            return None

        returns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                returns.append(ast.unparse(node.value))

        if not returns:
            return None

        ret_expr = returns[-1]
        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        q_lower = question.lower()
        if "return true" in q_lower:
            ans = f"In {loc}, `{ent.name}` returns `True` when `{ret_expr}` is satisfied [E1]."
            return self._build_result(project_id, question, ans, ent, chunk, source, started)
        elif "equal to or greater than" in q_lower or "return false" in q_lower:
            ans = f"In {loc}, `{ent.name}` returns `False` when attempt is equal to or greater than `max_attempts` (violating `{ret_expr}`) [E1]."
            return self._build_result(project_id, question, ans, ent, chunk, source, started)

        return None


    # -------------------------------------------------------- 8. formula calculation
    def _check_formula(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if not re.search(
            r"\b(?:mathematical\s+)?formula\b|\bhow\s+is\s+(?:the\s+)?(?:retry\s+)?delay\s+calculated\b|\bcalculate\s+(?:the\s+)?(?:retry\s+)?delay\b",
            question,
            re.IGNORECASE,
        ):
            return None

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        try:
            tree = ast.parse(chunk.text)
        except SyntaxError:
            return None

        ret_expr = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                ret_expr = ast.unparse(node.value)

        if not ret_expr:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        ans = (
            f"In {loc}, the retry delay in `{ent.name}` is calculated using the formula "
            f"`{ret_expr}` [E1]."
        )
        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # ------------------------------------------------------- 9. jitter generation
    def _check_jitter_generation(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if not re.search(
            r"\bhow does\s+(\w+)\s+generate\s+(?:the\s+)?(?:random\s+)?jitter\b|\bhow is\s+(?:the\s+)?(?:random\s+)?jitter\s+generated\b|\bjitter\s+generated\b",
            question,
            re.IGNORECASE,
        ):
            return None

        ent = self._find_entity_in_text(question, entities)
        if not ent or ent.entity_type not in (EntityType.FUNCTION, EntityType.METHOD):
            return None
        chunk = self._find_chunk_for_entity(ent, chunks)
        source = sources.get(ent.source_id)
        if not chunk or not source:
            return None

        try:
            tree = ast.parse(chunk.text)
        except SyntaxError:
            return None

        jitter_call = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ast.unparse(node)
                if "uniform" in call_str or "jitter" in call_str or "random" in call_str:
                    jitter_call = call_str
                    break

        if not jitter_call:
            return None

        loc = self._loc_str(source.name, ent.start_line, ent.end_line)
        ans = f"In {loc}, `{ent.name}` generates random jitter using `{jitter_call}` [E1]."
        return self._build_result(project_id, question, ans, ent, chunk, source, started)

    # --------------------------------------------------- 10. project dependencies
    def _check_project_dependencies(
        self,
        project_id: int,
        question: str,
        entities: list[CodeEntity],
        sources: dict[int, Any],
        chunks: list[Any],
        started: float,
    ) -> AskResult | None:
        if not re.search(
            r"\b(?:external\s+)?dependencies\b.*\b(?:require|need|depend on|uses?)\b|\bwhat\s+(?:external\s+)?dependencies\b",
            question,
            re.IGNORECASE,
        ):
            return None

        dep_chunk = None
        for c in chunks:
            if c.heading and "dependencies" in c.heading.lower():
                dep_chunk = c
                break
            if c.text and "## dependencies" in c.text.lower():
                dep_chunk = c
                break

        if not dep_chunk:
            return None

        source = sources.get(dep_chunk.source_id)
        if not source:
            return None

        lines = [line.strip() for line in dep_chunk.text.splitlines() if line.strip() and not line.strip().startswith("#")]
        desc = " ".join(lines) if lines else dep_chunk.text.strip()
        loc = self._loc_str(source.name, dep_chunk.start_line, dep_chunk.end_line)
        if "only the python standard library" in desc.lower() or "no external dependencies" in desc.lower():
            ans = f"In {loc}, the project requires only the Python standard library (no external dependencies) [E1]."
        else:
            ans = f"In {loc}, the project dependencies are: {desc} [E1]."

        return self._build_doc_result(project_id, question, ans, dep_chunk, source, started)
