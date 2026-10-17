"""Plain-text rendering of a VerificationReport (CLI, examples, future UI)."""

from __future__ import annotations

from trustdoc.models import ClaimVerdict, Severity
from trustdoc.verification.types import FindingReport, VerificationReport

DISCLAIMER = ("Statuses are evidence-based, not proof: INSUFFICIENT EVIDENCE means unverified (not false). Findings marked "
              "model-assisted come from an LLM and are not ground truth.")


def _where(f: FindingReport) -> str:
    parts = []
    if f.doc_location and f.doc_location.get("label"):
        parts.append(f"docs: {f.doc_location['label']}")
    if f.code_location and f.code_location.get("label"):
        parts.append(f"code: {f.code_location['label']}")
    return "   ".join(parts)


def render_report(report: VerificationReport, *, max_per_severity: int = 10) -> str:
    c = report.counts
    lines = [f"Verification run #{report.run.id} (project {report.project_id}): {report.run.status.value}"]
    if report.documents:
        lines.append(f"Documents: {', '.join(report.documents)}")
    lines += ["", f"Claims extracted: {c.get('claims', 0)}  "
              f"(supported {c.get('supported', 0)}, contradicted {c.get('contradicted', 0)}, "
              f"insufficient evidence {c.get('insufficient_evidence', 0)}, not checked {c.get('not_checkable', 0)})"]

    s = report.score
    if s is None:
        lines += ["", "Trust score: not available"]
    elif s.final is None:
        lines += ["", "Trust score: not measurable (" + "; ".join(s.notes) + ")"]
    else:
        lines += ["", f"Trust score: {s.final:.1f} / 100"]
    if s is not None:
        for comp in s.components:
            if comp.name == "severity_penalty":
                lines.append(f"  - severity penalty: {comp.contribution:+.1f}  ({comp.detail})")
            elif comp.weight == 0 and comp.detail.startswith("informational"):
                lines.append(f"  - {comp.name}: {comp.raw_score * 100:.1f}%  ({comp.detail})")
            elif comp.weight == 0:
                lines.append(f"  - {comp.name}: unavailable  ({comp.detail.replace('unavailable: ', '')})")
            else:
                lines.append(f"  - {comp.name}: {comp.raw_score * 100:.1f}%  weight {comp.weight:.2f}  -> {comp.contribution:+.1f}  ({comp.detail})")
        if s.final is not None and s.base is not None:
            lines.append(f"  = base {s.base:.1f} minus penalties -> {s.final:.1f}")
        for n in s.notes:
            if s.final is not None:
                lines.append(f"  note: {n}")

    grouped = report.findings_by_severity()
    defects = [sev for sev in grouped if sev != Severity.INFO and grouped[sev]]
    lines += ["", "Findings" + ("" if defects else ": none above INFO")]
    for sev in defects:
        items = grouped[sev]
        lines.append(f"[{sev.value.upper()}] {len(items)}")
        for f in items[:max_per_severity]:
            tag = " (model-assisted)" if f.model_assisted else ""
            lines.append(f"  - {f.message}{tag}")
            if _where(f):
                lines.append(f"      {_where(f)}")
            if f.suggestion:
                lines.append(f"      suggestion: {f.suggestion}")
            for e in f.evidence[:3]:
                lines.append(f"      evidence: {e.get('label')}")
        if len(items) > max_per_severity:
            lines.append(f"  ... and {len(items) - max_per_severity} more")
    info = grouped[Severity.INFO]
    if info:
        lines += ["", f"[INFO] {len(info)} unverified item(s) (not proven defects):"]
        for f in info[:max_per_severity]:
            lines.append(f"  - {f.message}" + (f"  [{_where(f)}]" if _where(f) else ""))
        if len(info) > max_per_severity:
            lines.append(f"  ... and {len(info) - max_per_severity} more (use --json for all)")

    if report.skipped:
        lines += ["", "Skipped checks:"] + [f"  - {m}" for m in report.skipped]
    if report.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in report.warnings[:10]]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
