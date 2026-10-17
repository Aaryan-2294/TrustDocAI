"""Severity model and finding construction.

Severity rules (also documented in the README):

=======================  =========================================================================
Severity                 Used for
=======================  =========================================================================
CRITICAL                 A documented *signature* that would break callers: a nonexistent API, a
                         documented parameter that does not exist, or a required parameter omitted
                         from a signature presented as complete.
HIGH                     Other incorrect API claims: nonexistent entity or wrong/missing parameter in
                         prose, wrong default value or annotation, wrong return type, a documented
                         positional/keyword-only mismatch, a required parameter omitted from a
                         complete parameter list.
MEDIUM                   A meaningful behavioural contradiction found by the (optional, LLM-assisted)
                         semantic verifier, and missing documentation for a public class or for a
                         callable that takes arguments.
LOW                      Minor gaps: optional parameters left out of a "complete" list, parameters
                         missing from a docstring Args section, missing docs on a no-argument callable.
INFO                     Unverified claims (insufficient evidence). Never a proven defect, and never
                         raised merely because evidence is missing.
=======================  =========================================================================
"""

from __future__ import annotations

import difflib
import hashlib
from typing import Any, Iterable, Sequence

from trustdoc.models import (
    Claim,
    ClaimType,
    ClaimVerdict,
    CodeEntity,
    Finding,
    IssueType,
    Severity,
    VerificationTier,
)
from trustdoc.verification.types import UNASSIGNED, claim_uid


def severity_for(issue_type: IssueType, claim_type: ClaimType | None = None, *, optional_only: bool = False,
                 has_arguments: bool = True, is_class: bool = False) -> Severity:
    """The documented severity table (see module docstring)."""
    is_signature = claim_type == ClaimType.SIGNATURE
    if issue_type == IssueType.NONEXISTENT_ENTITY:
        return Severity.CRITICAL if is_signature else Severity.HIGH
    if issue_type == IssueType.PARAMETER_NAME_MISMATCH:
        return Severity.CRITICAL if is_signature else Severity.HIGH
    if issue_type == IssueType.PARAMETER_COUNT_MISMATCH:
        if optional_only:
            return Severity.LOW
        return Severity.CRITICAL if is_signature else Severity.HIGH
    if issue_type in (IssueType.SIGNATURE_DEFAULT_MISMATCH, IssueType.RETURN_MISMATCH):
        return Severity.HIGH
    if issue_type == IssueType.MISSING_DOCUMENTATION:
        return Severity.MEDIUM if (is_class or has_arguments) else Severity.LOW
    if issue_type == IssueType.CONTRADICTED_CLAIM:
        return Severity.MEDIUM
    return Severity.INFO


def closest(name: str, candidates: Iterable[str], cutoff: float = 0.6) -> str | None:
    matches = difflib.get_close_matches(name, sorted(set(candidates)), n=1, cutoff=cutoff)
    return matches[0] if matches else None


def finding_uid(issue_type: IssueType, claim_key: str | None, entity_key: str | None, detail_key: str) -> str:
    raw = f"{issue_type.value}|{claim_key or ''}|{entity_key or ''}|{detail_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def code_location(entity: CodeEntity | None, source_name: str | None) -> dict[str, Any] | None:
    if entity is None:
        return None
    label = f"{source_name}:{entity.start_line}" if source_name and entity.start_line else (source_name or entity.qualified_name)
    return {"source_name": source_name, "start_line": entity.start_line, "end_line": entity.end_line,
            "qualified_name": entity.qualified_name, "label": label}


def doc_location(claim: Claim) -> dict[str, Any]:
    ex = claim.extracted
    return {"source_name": ex.get("doc_source_name"), "start_line": claim.start_line,
            "label": ex.get("doc_label"), "heading": ex.get("heading")}


def build_finding(
    *,
    issue_type: IssueType,
    severity: Severity,
    message: str,
    suggestion: str,
    tier: VerificationTier,
    claim: Claim | None,
    status: ClaimVerdict | str,
    category: ClaimType | str,
    entity: CodeEntity | None = None,
    entity_source_name: str | None = None,
    detail_key: str = "",
    detail: dict[str, Any] | None = None,
    evidence: Sequence[dict[str, Any]] = (),
    model_assisted: bool = False,
    source_id: int | None = None,
    start_line: int | None = None,
) -> Finding:
    """Create a persistable Finding draft whose ``details`` snapshot everything a reader needs.

    The snapshot (claim text, locations, citations) makes the finding self-contained, so run history
    stays interpretable even after the claim or its source has been replaced.
    """
    ckey = claim_uid(claim) if claim else None
    uid = finding_uid(issue_type, ckey, entity.qualified_name if entity else None, detail_key)
    details = {
        "finding_uid": uid,
        "category": category.value if isinstance(category, ClaimType) else category,
        "status": status.value if isinstance(status, ClaimVerdict) else status,
        "claim_uid": ckey,
        "claim_text": claim.text if claim else None,
        "doc_location": doc_location(claim) if claim else None,
        "code_location": code_location(entity, entity_source_name),
        "evidence": list(evidence),
        "model_assisted": model_assisted,
        "detail": detail or {},
    }
    return Finding(
        run_id=UNASSIGNED, project_id=claim.project_id if claim else UNASSIGNED, tier=tier,
        issue_type=issue_type, severity=severity, message=message, suggestion=suggestion,
        source_id=source_id if source_id is not None else (claim.source_id if claim else None),
        entity_id=entity.id if entity else None, claim_id=None,
        start_line=start_line if start_line is not None else (claim.start_line if claim else None),
        details=details,
    )
