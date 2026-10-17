"""Tier 1: deterministic verification against the AST-extracted code entities.

No LLM, no retrieval. Decisions are conservative:

* a documented subset of parameters is never treated as an error unless the documentation claims
  completeness (a ``def``-form signature, or words like "only"/"exactly");
* bound ``self``/``cls`` is never reported;
* call-like examples (``f(a, b=1)``) may use arbitrary variable names for positional arguments, so only
  keyword names and the number of positional arguments are checked;
* a missing return annotation, an ambiguous name, an alias/re-export or a class without an explicit
  ``__init__`` gives INSUFFICIENT_EVIDENCE, never a contradiction;
* external/standard-library names and names in example code that may define their own functions are
  NOT_CHECKABLE.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping, Sequence

from trustdoc.models import (
    Claim,
    ClaimType,
    ClaimVerdict,
    CodeEntity,
    EntityType,
    EvidenceStance,
    Finding,
    IssueType,
    Parameter,
    ParameterKind,
    Severity,
    VerificationTier,
)
from trustdoc.verification.findings import build_finding, closest, severity_for
from trustdoc.verification.resolver import EntityResolver, Resolution, ResolutionKind
from trustdoc.verification.typenorm import TypeMatch, compare_types
from trustdoc.verification.types import ClaimOutcome, EvidenceDraft

# Decorators known to preserve the decorated callable's signature. Any other decorator (click.option,
# app.route, custom wrappers, ...) may change it, so parameter claims about such callables are not verified.
SIGNATURE_PRESERVING_DECORATORS = frozenset({
    "staticmethod", "classmethod", "property", "abstractmethod", "abstractproperty", "overload", "final", "override",
    "cache", "lru_cache", "cached_property", "contextmanager", "asynccontextmanager", "setter", "getter", "deleter",
})
_VARIADIC = (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD)


def unrecognised_decorators(entity: CodeEntity) -> list[str]:
    """Decorators on ``entity`` that are not known to preserve its signature."""
    out = []
    for d in entity.decorators:
        base = d.split("(", 1)[0].strip().rsplit(".", 1)[-1]
        if base not in SIGNATURE_PRESERVING_DECORATORS:
            out.append(d)
    return out
_POSITIONAL = (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)


def render_params(params: Sequence[Parameter]) -> str:
    """Render parameters as source text, e.g. ``a, b=1, *, c: int = 2, **kw``."""
    out: list[str] = []
    seen_kw_marker = False
    last_posonly = max((i for i, p in enumerate(params) if p.kind == ParameterKind.POSITIONAL_ONLY), default=-1)
    has_varpos = any(p.kind == ParameterKind.VAR_POSITIONAL for p in params)
    for i, p in enumerate(params):
        if p.kind == ParameterKind.KEYWORD_ONLY and not has_varpos and not seen_kw_marker:
            out.append("*")
            seen_kw_marker = True
        text = {"var_positional": "*", "var_keyword": "**"}.get(p.kind.value, "") + p.name
        if p.annotation:
            text += f": {p.annotation}"
        if p.default is not None:
            text += (" = " if p.annotation else "=") + p.default
        out.append(text)
        if i == last_posonly:
            out.append("/")
    return ", ".join(out)


def _expr_key(text: str) -> str:
    try:
        return ast.dump(ast.parse(text.strip(), mode="eval").body)
    except (SyntaxError, ValueError):
        return text.strip()


class StaticVerifier:
    def __init__(self, resolver: EntityResolver, source_names: Mapping[int, str],
                 entity_chunks: Mapping[int, Sequence[int]] | None = None) -> None:
        self.resolver = resolver
        self.source_names = source_names
        self.entity_chunks = entity_chunks or {}

    # ------------------------------------------------------------------------ dispatch
    def verify(self, claim: Claim) -> ClaimOutcome:
        category = claim.claim_type
        res = self.resolver.resolve(claim.extracted.get("entity_name") or claim.subject_name or "",
                                    kind_word=claim.extracted.get("kind_word"))
        if res.kind != ResolutionKind.FOUND:
            return self._unresolved(claim, res)
        entity = res.entity
        assert entity is not None
        if claim.extracted.get("context") == "fenced_code" and entity.entity_type == EntityType.METHOD:
            return ClaimOutcome(claim, VerificationTier.DETERMINISTIC, ClaimVerdict.NOT_CHECKABLE,
                                f"`{claim.extracted.get('entity_name')}` in example code only matches the method `{entity.qualified_name}`; "
                                "a top-level definition in an example is not a claim about that method", skip_reason="example_code")
        if category == ClaimType.ENTITY_REFERENCE:
            return self._entity(claim, entity)
        if category == ClaimType.RETURN:
            return self._return(claim, entity)
        return self._parameters(claim, entity)

    # ------------------------------------------------------------------- unresolved names
    def _unresolved(self, claim: Claim, res: Resolution) -> ClaimOutcome:
        ex, name = claim.extracted, res.name or claim.subject_name or "?"

        def outcome(verdict: ClaimVerdict, why: str, findings: list[Finding] | None = None, skip: str | None = None) -> ClaimOutcome:
            return ClaimOutcome(claim, VerificationTier.DETERMINISTIC, verdict, why, confidence=1.0 if verdict != ClaimVerdict.NOT_CHECKABLE else None,
                                findings=findings or [], skip_reason=skip)

        if res.kind in (ResolutionKind.INVALID, ResolutionKind.EXTERNAL):
            why = f"`{name}`: {res.reason}; outside the scope of the indexed project source"
            return outcome(ClaimVerdict.NOT_CHECKABLE, why, skip=res.kind.value)
        if res.kind == ResolutionKind.NOT_FOUND:
            if ex.get("context") == "fenced_code":
                return outcome(ClaimVerdict.NOT_CHECKABLE, f"`{name}` appears in example code, which may define its own functions", skip="example_code")
            if not self.resolver.entities:
                return outcome(ClaimVerdict.INSUFFICIENT_EVIDENCE, "no Python source has been indexed to check this against",
                               [self._unverified(claim, "No Python source is indexed for this project.", name)])
            if res.strong or ex.get("callable_hint"):
                near = closest(name.split(".")[-1], self.resolver.known_names(), 0.7)
                hint = f"Did you mean `{near}`? " if near else ""
                finding = build_finding(
                    issue_type=IssueType.NONEXISTENT_ENTITY, severity=_sev(IssueType.NONEXISTENT_ENTITY, claim),
                    message=f"`{name}` is documented but no such function, class or method exists in the indexed Python source.",
                    suggestion=f"{hint}Remove or correct the reference; if it is defined outside the indexed sources, add those sources and re-run.",
                    tier=VerificationTier.DETERMINISTIC, claim=claim, status=ClaimVerdict.CONTRADICTED, category=claim.claim_type,
                    detail_key=name.lower(), detail={"documented_name": name, "closest": near, "reason": res.reason},
                )
                why = (f"No entity matching `{name}` among {len(self.resolver.entities)} indexed entities ({res.reason}). "
                       "Absence from the index is the evidence for this verdict.")
                return outcome(ClaimVerdict.CONTRADICTED, why, [finding])
            why = f"`{name}` could not be resolved to a function, class or method ({res.reason}); it may be a variable, key or external name"
            return outcome(ClaimVerdict.INSUFFICIENT_EVIDENCE, why, [self._unverified(claim, why, name)])
        if res.kind == ResolutionKind.ALIAS:
            why = f"`{name}`: {res.reason}; its signature cannot be inspected statically"
            return outcome(ClaimVerdict.INSUFFICIENT_EVIDENCE, why, [self._unverified(claim, why, name)])
        why = f"`{name}` is ambiguous ({res.reason}); not verified"
        return outcome(ClaimVerdict.INSUFFICIENT_EVIDENCE, why, [self._unverified(claim, why, name)])

    def _unverified(self, claim: Claim, message: str, key: str, entity: CodeEntity | None = None) -> Finding:
        return build_finding(
            issue_type=IssueType.UNSUPPORTED_CLAIM, severity=Severity.INFO, message=message,
            suggestion="Review manually; the static check could not decide. This is not evidence that the claim is false.",
            tier=VerificationTier.DETERMINISTIC, claim=claim, status=ClaimVerdict.INSUFFICIENT_EVIDENCE, category=claim.claim_type,
            entity=entity, entity_source_name=self._src(entity), detail_key=key.lower(),
        )

    # --------------------------------------------------------------------------- entity
    def _entity(self, claim: Claim, entity: CodeEntity) -> ClaimOutcome:
        kind = (claim.extracted.get("kind_word") or "").lower()
        label = f"{self._src(entity)}:{entity.start_line}"
        if kind in ("class", "constructor") and entity.entity_type != EntityType.CLASS:
            finding = build_finding(
                issue_type=IssueType.NONEXISTENT_ENTITY, severity=Severity.HIGH,
                message=f"`{entity.qualified_name}` is documented as a class but is a {entity.entity_type.value} ({label}).",
                suggestion=f"Describe `{entity.name}` as a {entity.entity_type.value}.", tier=VerificationTier.DETERMINISTIC, claim=claim,
                status=ClaimVerdict.CONTRADICTED, category=claim.claim_type, entity=entity, entity_source_name=self._src(entity),
                detail_key="kind", detail={"documented_kind": kind, "actual_kind": entity.entity_type.value},
            )
            return self._done(claim, entity, ClaimVerdict.CONTRADICTED, f"Found {entity.qualified_name} ({label}) but it is a {entity.entity_type.value}, not a class.", [finding])
        return self._done(claim, entity, ClaimVerdict.SUPPORTED, f"`{entity.qualified_name}` exists ({entity.entity_type.value}, {label}).", [])

    # ---------------------------------------------------------------- parameters/signature
    def _parameters(self, claim: Claim, entity: CodeEntity) -> ClaimOutcome:
        ex = claim.extracted
        actual = self.resolver.effective_parameters(entity)
        qn = entity.qualified_name
        if actual is None:
            why = f"`{qn}` is a class without an explicit __init__ (parameters may come from a dataclass, a base class or a metaclass); not verified"
            return self._done(claim, entity, ClaimVerdict.INSUFFICIENT_EVIDENCE, why, [self._unverified(claim, why, qn, entity)])

        target = self.resolver.init_of(entity) if entity.entity_type == EntityType.CLASS else entity
        risky = unrecognised_decorators(target) if target is not None else []
        if risky:
            why = (f"`{qn}` is decorated with {', '.join(f'`@{d}`' for d in risky[:3])}; decorators can change a callable's signature, "
                   "so the documented parameters are not verified against the source `def`")
            return self._done(claim, entity, ClaimVerdict.INSUFFICIENT_EVIDENCE, why, [self._unverified(claim, why, qn, entity)])

        documented: list[dict[str, Any]] = [d for d in ex.get("documented_params", []) if d["name"] not in ("self", "cls")]
        form = ex.get("form")
        is_def = form == "def"
        complete = bool(ex.get("complete"))
        by_name = {p.name: p for p in actual}
        has_varkw = any(p.kind == ParameterKind.VAR_KEYWORD for p in actual)
        has_varpos = any(p.kind == ParameterKind.VAR_POSITIONAL for p in actual)
        sig_text = f"{entity.name}({render_params(actual)})"
        defects: list[Finding] = []
        notes: list[Finding] = []
        uncertain: list[str] = []

        def mk(issue: IssueType, msg: str, suggestion: str, key: str, *, optional_only: bool = False, detail: dict | None = None) -> Finding:
            sev = severity_for(issue, claim.claim_type, optional_only=optional_only)
            if form == "call_like" and sev == Severity.CRITICAL:
                sev = Severity.HIGH                                  # a call-like mention is less certain than a signature
            return build_finding(issue_type=issue, severity=sev, message=msg, suggestion=suggestion, tier=VerificationTier.DETERMINISTIC,
                                 claim=claim, status=ClaimVerdict.CONTRADICTED, category=claim.claim_type, entity=entity,
                                 entity_source_name=self._src(entity), detail_key=key, detail={"actual_signature": sig_text, **(detail or {})})

        if form == "call_like":
            self._check_call_like(documented, actual, has_varkw, has_varpos, mk, defects, qn, sig_text)
        else:
            documented_names = [d["name"] for d in documented]
            for d in documented:
                name, kind = d["name"], d["kind"]
                if kind == "var_positional":
                    if not has_varpos:
                        defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH, f"`{qn}` is documented with `*{name}` but accepts no variadic positional arguments.",
                                          f"Actual signature: `{sig_text}`.", f"*{name}"))
                    continue
                if kind == "var_keyword":
                    if not has_varkw:
                        defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH, f"`{qn}` is documented with `**{name}` but accepts no variadic keyword arguments.",
                                          f"Actual signature: `{sig_text}`.", f"**{name}"))
                    continue
                a = by_name.get(name)
                if a is None or a.kind in _VARIADIC:
                    if has_varkw:
                        uncertain.append(name)
                    else:
                        near = closest(name, [p.name for p in actual if p.kind not in _VARIADIC])
                        defects.append(mk(
                            IssueType.PARAMETER_NAME_MISMATCH,
                            f"`{qn}` is documented with parameter `{name}`, but its actual parameters are: "
                            f"{', '.join(f'`{p.name}`' for p in actual) or '(none)'}.",
                            (f"Did you mean `{near}`? " if near else "") + f"Actual signature: `{sig_text}`.", name,
                            detail={"documented": name, "closest": near}))
                    continue
                self._check_param_details(d, a, form, is_def, mk, defects, qn)
            if is_def:
                self._check_order(documented, actual, mk, defects, qn, sig_text)
            if complete:
                documented_set = set(documented_names)
                missing_required = [p for p in actual if p.kind not in _VARIADIC and p.default is None and p.name not in documented_set]
                missing_optional = [p for p in actual if p.kind not in _VARIADIC and p.default is not None and p.name not in documented_set]
                for p in missing_required:
                    defects.append(mk(IssueType.PARAMETER_COUNT_MISMATCH, f"`{qn}` has the required parameter `{p.name}` that the documented parameter list omits.",
                                      f"Add `{p.name}` to the documentation. Actual signature: `{sig_text}`.", f"omit:{p.name}"))
                for p in missing_optional:
                    defects.append(mk(IssueType.PARAMETER_COUNT_MISMATCH, f"`{qn}` has the optional parameter `{p.name}` that the documented parameter list omits.",
                                      f"Mention `{p.name}` (default `{p.default}`). Actual signature: `{sig_text}`.", f"omit:{p.name}", optional_only=True))
            elif ex.get("args_section"):
                documented_set = set(documented_names)
                for p in actual:
                    if p.kind not in _VARIADIC and p.name not in documented_set:
                        notes.append(build_finding(
                            issue_type=IssueType.MISSING_DOCUMENTATION, severity=Severity.LOW,
                            message=f"Parameter `{p.name}` of `{qn}` is not described in its parameters section.",
                            suggestion=f"Document `{p.name}`. Actual signature: `{sig_text}`.", tier=VerificationTier.DETERMINISTIC, claim=claim,
                            status="missing_documentation", category=claim.claim_type, entity=entity, entity_source_name=self._src(entity),
                            detail_key=f"undocumented:{p.name}"))

        if ex.get("documented_return") and (ex.get("form") in ("def",)):
            self._check_return(claim, entity, ex["documented_return"], defects, notes, mk)

        findings = defects + notes
        if defects:
            why = "; ".join(f.message for f in defects[:3])
            return self._done(claim, entity, ClaimVerdict.CONTRADICTED, why, findings)
        if uncertain:
            why = f"`{qn}` accepts **kwargs, so documented parameter(s) {', '.join(f'`{n}`' for n in uncertain)} cannot be confirmed statically"
            return self._done(claim, entity, ClaimVerdict.INSUFFICIENT_EVIDENCE, why, findings + [self._unverified(claim, why, qn, entity)])
        return self._done(claim, entity, ClaimVerdict.SUPPORTED, f"Documented parameters are consistent with `{sig_text}`.", findings)

    def _check_call_like(self, documented, actual, has_varkw, has_varpos, mk, defects, qn, sig_text) -> None:
        by_name = {p.name: p for p in actual}
        positional_capacity = sum(1 for p in actual if p.kind in _POSITIONAL)
        placeholders = [d for d in documented if d.get("placeholder")]
        for d in documented:
            if d.get("placeholder") or d["kind"] in ("var_positional", "var_keyword"):
                continue
            if d["name"] not in by_name and not has_varkw:                  # keyword argument that does not exist
                defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH,
                                  f"The documented call passes keyword `{d['name']}`, which `{qn}` does not accept.",
                                  f"Actual signature: `{sig_text}`.", d["name"]))
        if not has_varpos and len(placeholders) > positional_capacity:
            defects.append(mk(IssueType.PARAMETER_COUNT_MISMATCH,
                              f"The documented call passes {len(placeholders)} positional arguments but `{qn}` accepts at most {positional_capacity}.",
                              f"Actual signature: `{sig_text}`.", f"positional:{len(placeholders)}"))

    def _check_param_details(self, d, a: Parameter, form, is_def, mk, defects, qn) -> None:
        name = d["name"]
        if d.get("default") is not None and form != "call_like":
            if a.default is None:
                defects.append(mk(IssueType.SIGNATURE_DEFAULT_MISMATCH, f"`{qn}` documents a default of `{d['default']}` for `{name}`, but the parameter is required.",
                                  f"`{name}` has no default in the source.", f"default:{name}"))
            elif _expr_key(d["default"]) != _expr_key(a.default):
                defects.append(mk(IssueType.SIGNATURE_DEFAULT_MISMATCH, f"`{qn}` documents `{name}={d['default']}` but the source default is `{a.default}`.",
                                  f"Use `{name}={a.default}` as in the source.", f"default:{name}", detail={"documented": d["default"], "actual": a.default}))
        if d.get("annotation") and a.annotation and compare_types(d["annotation"], a.annotation) == TypeMatch.MISMATCH:
            defects.append(mk(IssueType.SIGNATURE_DEFAULT_MISMATCH, f"`{qn}` documents `{name}` as `{d['annotation']}` but the source annotation is `{a.annotation}`.",
                              f"Use `{a.annotation}` as annotated in the source.", f"type:{name}", detail={"documented": d["annotation"], "actual": a.annotation}))
        if is_def:                                                          # only break-the-caller kind conflicts
            if a.kind == ParameterKind.KEYWORD_ONLY and d["kind"] in ("positional_only", "positional_or_keyword"):
                defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH, f"`{name}` is documented as positional but `{qn}` accepts it only by keyword.",
                                  f"Call it as `{name}=...`.", f"kind:{name}"))
            elif a.kind == ParameterKind.POSITIONAL_ONLY and d["kind"] == "keyword_only":
                defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH, f"`{name}` is documented as keyword-only but `{qn}` accepts it only by position.",
                                  f"Pass `{name}` positionally.", f"kind:{name}"))

    def _check_order(self, documented, actual, mk, defects, qn, sig_text) -> None:
        actual_pos = [p.name for p in actual if p.kind in _POSITIONAL]
        doc_pos = [d["name"] for d in documented if d["kind"] in ("positional_only", "positional_or_keyword", "unknown")]
        common_doc = [n for n in doc_pos if n in actual_pos]
        common_act = [n for n in actual_pos if n in common_doc]
        if common_doc != common_act:
            defects.append(mk(IssueType.PARAMETER_NAME_MISMATCH, f"`{qn}` documents its positional parameters in a different order than the source.",
                              f"Actual signature: `{sig_text}`.", "order", detail={"documented_order": common_doc, "actual_order": common_act}))

    # ---------------------------------------------------------------------------- return
    def _return(self, claim: Claim, entity: CodeEntity) -> ClaimOutcome:
        defects: list[Finding] = []
        notes: list[Finding] = []
        sev = lambda: severity_for(IssueType.RETURN_MISMATCH, claim.claim_type)  # noqa: E731
        self._check_return(claim, entity, claim.extracted.get("documented_return"), defects, notes, None)
        if defects:
            return self._done(claim, entity, ClaimVerdict.CONTRADICTED, defects[0].message, defects + notes)
        if notes:
            return self._done(claim, entity, ClaimVerdict.INSUFFICIENT_EVIDENCE, notes[0].message, notes)
        return self._done(claim, entity, ClaimVerdict.SUPPORTED,
                          f"`{entity.qualified_name}` is annotated `-> {entity.return_annotation}`, consistent with the documented `{claim.extracted.get('documented_return')}`.", [])

    def _check_return(self, claim: Claim, entity: CodeEntity, documented: str | None, defects: list[Finding], notes: list[Finding], mk) -> None:
        qn, label = entity.qualified_name, f"{self._src(entity)}:{entity.start_line}"
        if entity.entity_type == EntityType.CLASS:
            notes.append(self._unverified(claim, f"`{qn}` is a class; a constructor call returns an instance, so a documented return type is not checked.", f"return:{qn}", entity))
            return
        actual = entity.return_annotation
        if actual is None:
            notes.append(self._unverified(claim, f"`{qn}` has no return annotation ({label}); the documented return type `{documented}` cannot be checked statically.", f"return:{qn}", entity))
            return
        match = compare_types(documented, actual)
        if match == TypeMatch.MISMATCH:
            defects.append(build_finding(
                issue_type=IssueType.RETURN_MISMATCH, severity=severity_for(IssueType.RETURN_MISMATCH, claim.claim_type),
                message=f"`{qn}` is documented as returning `{documented}` but its annotation is `-> {actual}` ({label}).",
                suggestion=f"Document the return type as `{actual}`, as annotated in the source.", tier=VerificationTier.DETERMINISTIC, claim=claim,
                status=ClaimVerdict.CONTRADICTED, category=claim.claim_type, entity=entity, entity_source_name=self._src(entity),
                detail_key=f"return:{qn}", detail={"documented": documented, "actual": actual}))
        elif match == TypeMatch.UNKNOWN:
            notes.append(self._unverified(claim, f"The documented return type `{documented}` cannot be compared with the annotation `-> {actual}` (alias, union or differing specificity).", f"return:{qn}", entity))

    # ------------------------------------------------------------------------ plumbing
    def _done(self, claim: Claim, entity: CodeEntity, verdict: ClaimVerdict, why: str, findings: list[Finding]) -> ClaimOutcome:
        stance = {ClaimVerdict.SUPPORTED: EvidenceStance.SUPPORTS, ClaimVerdict.CONTRADICTED: EvidenceStance.CONTRADICTS}.get(verdict, EvidenceStance.RELATED)
        evidence = [EvidenceDraft(cid, stance, None, rank) for rank, cid in enumerate(self.entity_chunks.get(entity.id, ()), start=1)]
        return ClaimOutcome(claim, VerificationTier.DETERMINISTIC, verdict, why, confidence=float(claim.extracted.get("confidence", 1.0)),
                            findings=findings, evidence=evidence, entity=entity)

    def _src(self, entity: CodeEntity | None) -> str | None:
        return self.source_names.get(entity.source_id) if entity else None


def _sev(issue: IssueType, claim: Claim) -> Severity:
    sev = severity_for(issue, claim.claim_type)
    return Severity.HIGH if (claim.extracted.get("form") == "call_like" and sev == Severity.CRITICAL) else sev


# ------------------------------------------------------------------------- doc coverage
def check_documentation_coverage(
    entities: Iterable[CodeEntity], resolver: EntityResolver, mentions: Iterable[str], source_names: Mapping[int, str]
) -> tuple[list[Finding], dict[str, int]]:
    """Public classes/functions/methods without a docstring AND not mentioned in the verified documents.

    Uses the analyzer's public/private rules (underscore names, dunders and nested definitions are
    not public). These are coverage findings, never claim contradictions.
    """
    public = [e for e in entities if e.is_public and e.entity_type in (EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD)]
    mentioned_ids: set[int] = set()
    for name in mentions:
        res = resolver.resolve(name)
        if res.kind == ResolutionKind.FOUND and res.entity is not None:
            mentioned_ids.add(res.entity.id)
    findings: list[Finding] = []
    documented = 0
    for e in public:
        if e.has_docstring or e.id in mentioned_ids:
            documented += 1
            continue
        params = resolver.effective_parameters(e) or []
        sig = f"{e.name}({render_params(params)})" if e.entity_type != EntityType.CLASS else f"class {e.name}"
        src = source_names.get(e.source_id)
        findings.append(build_finding(
            issue_type=IssueType.MISSING_DOCUMENTATION,
            severity=severity_for(IssueType.MISSING_DOCUMENTATION, None, has_arguments=bool(params), is_class=e.entity_type == EntityType.CLASS),
            message=f"Public {e.entity_type.value} `{e.qualified_name}` has no docstring and is not mentioned in the verified documentation ({src}:{e.start_line}).",
            suggestion=f"Add a docstring. Source signature: `{sig}`.", tier=VerificationTier.DETERMINISTIC, claim=None,
            status="missing_documentation", category="coverage", entity=e, entity_source_name=src, detail_key=e.qualified_name,
            source_id=e.source_id, start_line=e.start_line))
    return findings, {"public_entities": len(public), "documented_entities": documented}
