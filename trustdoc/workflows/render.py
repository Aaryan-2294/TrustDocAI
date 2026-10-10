"""Plain-text rendering of an AskResult (used by the CLI and the examples).

Never prints prompts, secrets, or full source text; ``show_evidence`` adds short
single-line snippets only.
"""

from __future__ import annotations

from trustdoc.workflows.types import AskResult, AskStatus, GroundingLevel

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
