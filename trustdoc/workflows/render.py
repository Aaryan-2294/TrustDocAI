"""Plain-text rendering of an AskResult (used by the CLI and the examples).

Never prints prompts, secrets, or full source text; ``show_evidence`` adds short
single-line snippets only.
"""

from __future__ import annotations

from trustdoc.workflows.types import (
    AskResult,
    AskStatus,
    GenerationKind,
    GenerationResult,
    GenerationStatus,
    GroundingLevel,
)

DISCLAIMER = ("Citations show which retrieved evidence the model referenced. The claims in this "
              "answer have NOT been independently verified.")


def render_ask_result(result: AskResult, *, show_evidence: bool = False, snippet_chars: int = 160) -> str:
    lines = [f"Question: {result.question}", ""]

    if result.status in (AskStatus.RETRIEVAL_ERROR, AskStatus.GENERATION_ERROR):
        stage = "Retrieval" if result.status == AskStatus.RETRIEVAL_ERROR else "Answer generation"
        lines.append(f"{stage} failed ({result.error_kind}): {result.error}")
        if result.evidence:
            lines += ["", f"{len(result.evidence)} evidence item(s) were retrieved but no answer was produced:"]
            lines += [f"  [{e.id}] {e.evidence.citation.label}" for e in result.evidence]
        return "\n".join(lines)

    model = ""
    if result.model_name:
        stand_in = "  [STAND-IN, not a language model]" if result.is_stub_model else ""
        model = f" | model: {result.model_name} ({result.backend}){stand_in}"

    if result.status == AskStatus.INSUFFICIENT_EVIDENCE:
        lines.append(f"Result: INSUFFICIENT EVIDENCE ({(result.abstention_reason or '').replace('_', ' ')}){model}")
        lines += ["", result.answer]
        r = result.retrieval
        if r and r.top_candidates and result.abstention_reason != "model_declared_insufficient":
            lines += ["", "Closest retrieved sources (not used to answer):"]
            for c in r.top_candidates:
                score = "" if c["rerank_score"] is None else f"  (relevance score {c['rerank_score']:.3f})"
                lines.append(f"  - {c['label']}{score}")
    else:
        grounding = result.grounding.value.replace("_", " ") if result.grounding else "n/a"
        lines.append(f"Result: answered{model} | citation check: {grounding}")
        lines += ["", "Answer:", result.answer]

    if result.citations:
        lines += ["", "Sources cited:"]
        for c in result.citations:
            entity = f"  ({c.qualified_name})" if c.qualified_name else ""
            lines.append(f"  [{c.id}] {c.label}{entity}")
    cited = {c.id for c in result.citations}
    uncited = [e for e in result.evidence if e.id not in cited]
    if uncited and result.status == AskStatus.ANSWERED:
        lines += ["", "Retrieved evidence not cited in the answer:"]
        lines += [f"  [{e.id}] {e.evidence.citation.label}" for e in uncited]

    if show_evidence and result.evidence:
        lines += ["", "Evidence shown to the model (snippets):"]
        for e in result.evidence:
            snippet = " ".join(e.evidence.text.split())[:snippet_chars]
            lines.append(f"  [{e.id}] {e.evidence.citation.label}: {snippet}")

    if result.warnings:
        lines += ["", "Notes:"] + [f"  - {w}" for w in result.warnings]
    if result.status == AskStatus.ANSWERED:
        lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ---------------------------------------------------------------- generation (Part 6)
GENERATION_DISCLAIMER = (
    "Citations show which retrieved evidence the model referenced. Verification results "
    "above reflect automated static (and optional semantic) checks against the code index."
)


def render_generation_result(
    result: GenerationResult,
    *,
    show_evidence: bool = False,
    show_markdown: bool = True,
    max_findings: int = 10,
    snippet_chars: int = 160,
) -> str:
    from trustdoc.verification.render import render_report

    kind_label = "README" if result.kind == GenerationKind.README else f"API ({result.target_entity or 'unspecified'})"
    lines = [f"Documentation generation: {kind_label} (project {result.project_id})"]

    if result.status in (GenerationStatus.RETRIEVAL_ERROR, GenerationStatus.GENERATION_ERROR):
        stage = "Retrieval" if result.status == GenerationStatus.RETRIEVAL_ERROR else "Generation"
        lines.append(f"{stage} failed ({result.error_kind}): {result.error}")
        if result.evidence:
            lines += ["", f"{len(result.evidence)} evidence item(s) were retrieved:"]
            lines += [f"  [{e.id}] {e.evidence.citation.label}" for e in result.evidence]
        return "\n".join(lines)

    if result.status == GenerationStatus.INSUFFICIENT_EVIDENCE:
        lines.append("Result: INSUFFICIENT EVIDENCE (no documentation generated)")
        if result.warnings:
            lines += ["", "Notes:"] + [f"  - {w}" for w in result.warnings]
        return "\n".join(lines)

    model = ""
    if result.model_name:
        stand_in = "  [STAND-IN, not a language model]" if result.is_stub_model else ""
        model = f" | model: {result.model_name} ({result.backend}){stand_in}"

    grounding = result.grounding.value.replace("_", " ") if result.grounding else "n/a"

    if result.status == GenerationStatus.VERIFICATION_ERROR:
        lines.append(
            f"Result: VERIFICATION ERROR ({result.error_kind or 'verification_failed'}): "
            f"{result.error}{model} | citation check: {grounding}"
        )
    elif result.status == GenerationStatus.VERIFICATION_SKIPPED:
        lines.append(
            f"Result: VERIFICATION SKIPPED (unverified){model} | citation check: {grounding}"
        )
    else:
        lines.append(f"Result: {result.status.value}{model} | citation check: {grounding}")

    if show_markdown and result.markdown:
        if result.status == GenerationStatus.VERIFICATION_ERROR:
            lines += [
                "",
                "--- Generated Markdown (UNVERIFIED) ---",
                result.markdown.strip(),
                "--- End of Generated Markdown (UNVERIFIED) ---",
            ]
        elif result.status == GenerationStatus.VERIFICATION_SKIPPED:
            lines += [
                "",
                "--- Generated Markdown (UNVERIFIED - verification skipped) ---",
                result.markdown.strip(),
                "--- End of Generated Markdown (UNVERIFIED - verification skipped) ---",
            ]
        else:
            lines += ["", "--- Generated Markdown ---", result.markdown.strip(), "--- End of Generated Markdown ---"]

    if result.citations:
        lines += ["", "Sources cited:"]
        for c in result.citations:
            entity = f"  ({c.qualified_name})" if c.qualified_name else ""
            lines.append(f"  [{c.id}] {c.label}{entity}")

    if result.invalid_citation_ids:
        lines += ["", "Invalid citation IDs:"]
        lines += [f"  - [{cid}] (not in supplied evidence)" for cid in result.invalid_citation_ids]

    if show_evidence and result.evidence:
        lines += ["", "Evidence shown to the model (snippets):"]
        for e in result.evidence:
            snippet = " ".join(e.evidence.text.split())[:snippet_chars]
            lines.append(f"  [{e.id}] {e.evidence.citation.label}: {snippet}")

    if result.verification_report:
        lines += ["", "=== Independent Verification Report ==="]
        lines.append(render_report(result.verification_report, max_per_severity=max_findings))

    if result.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in result.warnings]

    if result.status in (GenerationStatus.VERIFICATION_SKIPPED, GenerationStatus.VERIFICATION_ERROR):
        lines += [
            "",
            "Citations show which retrieved evidence the model referenced. "
            "Verification was not performed or failed; this output has NOT been verified against the code index.",
        ]
    else:
        lines += ["", GENERATION_DISCLAIMER]
    return "\n".join(lines)

