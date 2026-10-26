"""Reusable Streamlit presentation components for TrustDoc AI (Part 7)."""

from __future__ import annotations

from typing import Any, Sequence

import streamlit as st

from trustdoc.models import Severity
from trustdoc.verification.types import FindingReport, ScoreReport
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    AskStatus,
    GenerationStatus,
    GroundingLevel,
)

HEURISTIC_DISCLAIMER = (
    "The Trust Score is an automated heuristic metric derived from static code "
    "and claim verification, not a calibrated probability of truth."
)


def render_header() -> None:
    """Render the top banner and title."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>TrustDoc AI</span>
                <span style="font-size: 0.85rem; font-weight: normal; background-color: #2b313e; color: #a3b8cc; padding: 0.2rem 0.6rem; border-radius: 12px;">v0.1.0</span>
            </h1>
            <p style="color: #6c757d; font-size: 1.05rem; margin-top: 0;">
                Evidence-Grounded Documentation Generation, Verification & Hallucination Detection
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heuristic_disclaimer() -> None:
    """Prominently display the heuristic nature of the Trust Score."""
    st.caption(f"ℹ️ **Notice:** {HEURISTIC_DISCLAIMER}")


def render_status_badge(
    status: str | Any,
    is_verified: bool = False,
    error: str | None = None,
) -> None:
    """Render an informative status alert depending on the workflow outcome."""
    status_str = status.value if hasattr(status, "value") else str(status)

    if status_str == GenerationStatus.COMPLETED.value:
        if is_verified:
            st.success("✅ **Verified Documentation**: Generated and verified against project code AST.")
        else:
            st.info("📄 **Documentation Generated**: Output ready.")
    elif status_str == GenerationStatus.VERIFICATION_SKIPPED.value:
        st.warning(
            "⚠️ **Unverified Documentation**: Automated verification was deliberately skipped (`run_verify=False`)."
        )
    elif status_str == GenerationStatus.VERIFICATION_ERROR.value:
        msg = "❌ **Verification Failed**: Verification crashed or encountered an error."
        if error:
            msg += f" Details: {error}"
        st.error(msg)
    elif status_str in (
        AskStatus.INSUFFICIENT_EVIDENCE.value,
        GenerationStatus.INSUFFICIENT_EVIDENCE.value,
    ):
        st.info(
            "🛡️ **Insufficient Evidence**: The system abstained honestly without fabricating claims "
            "because the required project evidence was missing or below threshold."
        )
    elif status_str == AskStatus.ANSWERED.value:
        st.success("✅ **Answer Generated**: Grounded in retrieved project evidence.")
    elif status_str in (
        AskStatus.RETRIEVAL_ERROR.value,
        GenerationStatus.RETRIEVAL_ERROR.value,
    ):
        st.error(f"❌ **Retrieval Error**: Operational failure during evidence search: {error or ''}")
    elif status_str in (
        AskStatus.GENERATION_ERROR.value,
        GenerationStatus.GENERATION_ERROR.value,
    ):
        st.error(f"❌ **Generation Error**: Model generation failed or unavailable: {error or ''}")
    else:
        st.info(f"Status: {status_str}")


def render_trust_score_card(score: ScoreReport | float | None) -> None:
    """Display the Trust Score gauge, component breakdown, and penalties."""
    st.markdown("### 🛡️ Trust Score")

    if score is None:
        st.info(
            "**Trust Score:** *Not Measurable*\n\n"
            "A Trust Score is computed only when at least one verifiable claim exists. "
            "Coverage alone cannot verify correctness."
        )
        render_heuristic_disclaimer()
        return

    final_val: float | None = score.final if isinstance(score, ScoreReport) else score
    if final_val is None:
        st.info("**Trust Score:** *Not Measurable*")
        render_heuristic_disclaimer()
        return

    # Visual color indicator
    color = "#28a745" if final_val >= 80 else ("#ffc107" if final_val >= 60 else "#dc3545")
    label = "High Trust" if final_val >= 80 else ("Moderate Trust" if final_val >= 60 else "Low Trust / Defects Detected")

    c1, c2 = st.columns([1, 2])
    c1.markdown(
        f"""
        <div style="border: 2px solid {color}; border-radius: 8px; padding: 1rem; text-align: center; background-color: rgba(0,0,0,0.05);">
            <div style="font-size: 2.2rem; font-weight: bold; color: {color};">{final_val:.1f} / 100</div>
            <div style="font-size: 0.9rem; font-weight: 500; color: {color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if isinstance(score, ScoreReport):
        if score.base is not None:
            c2.markdown(f"**Base Score:** `{score.base:.1f}`")
        if score.components:
            breakdown = []
            for comp in score.components:
                breakdown.append(
                    f"- **{comp.name.capitalize()}**: `{comp.raw_score * 100:.1f}%` (weight: `{comp.weight:.2f}`, contribution: `{comp.contribution:+.1f} pts`)"
                )
            c2.markdown("\n".join(breakdown))

        if score.penalties:
            c2.markdown("**Contradiction Penalties Applied:**")
            for pen in score.penalties:
                c2.markdown(f"- ⚠️ `{pen.get('label', 'Penalty')}`: `-{pen.get('penalty', 0):.1f} pts` ({pen.get('detail', '')})")

    render_heuristic_disclaimer()


def render_claim_counts(counts: dict[str, int]) -> None:
    """Display the 4-quadrant claim verification outcome metrics."""
    st.markdown("### 📊 Claim Verification Outcomes")

    supported = counts.get("supported", 0)
    contradicted = counts.get("contradicted", 0)
    insufficient = counts.get("insufficient_evidence", 0) or counts.get("insufficient", 0)
    not_checked = counts.get("not_checked", 0) or counts.get("not_checkable", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="✅ Supported", value=supported)
    c2.metric(label="❌ Contradicted", value=contradicted)
    c3.metric(label="❓ Insufficient Evidence", value=insufficient)
    c4.metric(label="⚪ Not Checked", value=not_checked)


def render_findings_list(
    findings: Sequence[FindingReport | Any],
    max_per_severity: int = 15,
) -> None:
    """Group findings by severity and render actionable remediation cards."""
    st.markdown("### ⚠️ Verification Findings & Remediation")

    if not findings:
        st.success("No defect findings detected in the verified documentation.")
        return

    # Group by severity
    by_sev: dict[str, list[Any]] = {
        Severity.CRITICAL.value: [],
        Severity.HIGH.value: [],
        Severity.MEDIUM.value: [],
        Severity.LOW.value: [],
        Severity.INFO.value: [],
    }

    for f in findings:
        sev_val = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
        if sev_val in by_sev:
            by_sev[sev_val].append(f)
        else:
            by_sev.setdefault(sev_val, []).append(f)

    sev_labels = [
        (Severity.CRITICAL.value, "Critical Severity (Breaking Contradictions)", "🔴"),
        (Severity.HIGH.value, "High Severity (API Mismatches)", "🟠"),
        (Severity.MEDIUM.value, "Medium Severity (Parameter/Type Inconsistencies)", "🟡"),
        (Severity.LOW.value, "Low Severity (Documentation Gaps)", "🔵"),
        (Severity.INFO.value, "Informational (Unchecked / Non-Defect Notes)", "⚪"),
    ]

    found_any = False
    for sev_key, sev_title, icon in sev_labels:
        items = by_sev.get(sev_key, [])
        if not items:
            continue
        found_any = True
        with st.expander(f"{icon} {sev_title} ({len(items)})", expanded=(sev_key in ("critical", "high"))):
            for i, item in enumerate(items[:max_per_severity]):
                st.markdown(f"**{i+1}. {item.issue_type.upper() if hasattr(item, 'issue_type') else 'ISSUE'}**: {item.message}")
                if getattr(item, "suggestion", None):
                    st.info(f"💡 **Suggestion**: {item.suggestion}")

                loc_info = []
                doc_loc = getattr(item, "doc_location", None)
                if doc_loc and isinstance(doc_loc, dict):
                    loc_info.append(f"**Doc Location**: `{doc_loc.get('source', '')}:{doc_loc.get('start_line', '')}`")
                elif getattr(item, "start_line", None):
                    loc_info.append(f"**Line**: `{item.start_line}`")

                code_loc = getattr(item, "code_location", None)
                if code_loc and isinstance(code_loc, dict):
                    loc_info.append(f"**Code Location**: `{code_loc.get('source', '')}:{code_loc.get('start_line', '')}`")

                if loc_info:
                    st.caption(" | ".join(loc_info))

                if getattr(item, "claim_text", None):
                    st.caption(f"Claim text: *\"{item.claim_text}\"*")

                st.divider()

    if not found_any:
        st.success("No defect findings detected in the verified documentation.")


def render_citations_list(citations: Sequence[AskCitation]) -> None:
    """Render cited sources list with location anchors."""
    if not citations:
        st.caption("No valid citations used.")
        return

    st.markdown("#### 📚 Verified Citations")
    for cit in citations:
        loc = []
        if cit.start_line is not None:
            if cit.end_line is not None and cit.end_line != cit.start_line:
                loc.append(f"L{cit.start_line}–{cit.end_line}")
            else:
                loc.append(f"L{cit.start_line}")
        if cit.qualified_name:
            loc.append(f"`{cit.qualified_name}`")
        if cit.heading:
            loc.append(f"§ {cit.heading}")

        loc_str = f" ({', '.join(loc)})" if loc else ""
        st.markdown(f"- **[{cit.id}]** `{cit.source_name}`{loc_str} — *{cit.label}*")


def render_evidence_drawer(evidence: Sequence[AskEvidence]) -> None:
    """Render expander drawer for inspection of retrieved text chunks."""
    if not evidence:
        return

    with st.expander(f"🔍 Inspect Retrieved Evidence ({len(evidence)} items)"):
        for item in evidence:
            ev = item.evidence
            score_str = f" | Rerank Score: `{ev.rerank_score:.3f}`" if ev.rerank_score is not None else ""
            st.markdown(f"**[{item.id}]** `{item.evidence.source_name}` {score_str}")
            st.code(ev.text, language="python" if ev.source_name.endswith(".py") else "markdown")
            if item.truncated:
                st.caption("⚠️ Note: Snippet was truncated to fit model context limit.")
            st.divider()
