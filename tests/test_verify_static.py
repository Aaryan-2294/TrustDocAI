import pytest
from verify_support import K, P, check, entity, md_claims, module, only, verifier

from trustdoc.models import ClaimType, ClaimVerdict as V, EntityType, EvidenceStance, IssueType as I, Severity as S
from trustdoc.parsing.python_analyzer import analyze_python
from trustdoc.verification.extraction import extract_docstring_claims
from trustdoc.verification.resolver import EntityResolver
from trustdoc.verification.static_verifier import check_documentation_coverage, render_params


def code():
    return [
        module("pkg.mod"),
        entity("pkg.mod.compute_backoff", params=[P("attempt", annotation="int"), P("base_seconds", "2.0", "float"), P("jitter", "True")], ret="float", id=2),
        entity("pkg.mod.retry_failed", params=[P("max_attempts", "3")], ret="int", id=3),
        entity("pkg.mod.TaskQueue", type=EntityType.CLASS, id=4),
        entity("pkg.mod.TaskQueue.__init__", type=EntityType.METHOD, params=[P("self"), P("backend", "None"), P("max_size", "10000")], public=False, id=5),
        entity("pkg.mod.TaskQueue.enqueue", type=EntityType.METHOD, params=[P("self"), P("payload", annotation="dict"), P("priority", "0"), P("delay_seconds", "0.0")], ret="Task", id=6),
        entity("pkg.mod.TaskQueue.purge", type=EntityType.METHOD, params=[P("self")], ret="int", id=7),
        entity("pkg.mod.untyped", params=[P("x")], id=8),
        entity("pkg.mod.NoInit", type=EntityType.CLASS, id=9),
    ]


def findings(o, issue=None):
    return [f for f in o.findings if issue is None or f.issue_type == issue]


# ------------------------------------------------------------------------------ entities
def test_existing_entity_is_supported_with_evidence_links():
    v = verifier(code(), chunks={2: [41, 42]})
    o = v.verify(only(md_claims("See the `compute_backoff` function."), ClaimType.ENTITY_REFERENCE))
    assert o.verdict == V.SUPPORTED and o.confidence == 0.9 and not o.findings
    assert [(d.chunk_id, d.stance) for d in o.evidence] == [(41, EvidenceStance.SUPPORTS), (42, EvidenceStance.SUPPORTS)]


def test_nonexistent_function_is_a_contradiction_with_a_justified_suggestion():
    o = check("Then call `retry_all()` to requeue.", code(), ClaimType.ENTITY_REFERENCE)
    assert o.verdict == V.CONTRADICTED
    f = findings(o, I.NONEXISTENT_ENTITY)[0]
    assert f.severity == S.HIGH and "retry_failed" in f.suggestion and "retry_all" in f.message
    assert "Absence from the index" in o.rationale and f.details["detail"]["closest"] == "retry_failed"


def test_unattributable_unknown_names_are_unverified_not_false():
    o = check("`retry_all` accepts `limit`.", code(), ClaimType.PARAMETER)          # no kind word, no parens, not dotted
    assert o.verdict == V.INSUFFICIENT_EVIDENCE and [f.severity for f in o.findings] == [S.INFO]
    assert "not evidence that the claim is false" in o.findings[0].suggestion


def test_names_rooted_in_a_project_package_are_confidently_missing():
    o = check("`pkg.mod.gone` accepts `a`.", code(), ClaimType.PARAMETER)
    assert o.verdict == V.CONTRADICTED and findings(o)[0].issue_type == I.NONEXISTENT_ENTITY


def test_external_and_stdlib_names_are_not_checkable():
    for name in ("json.loads", "requests.get", "os.path.join"):
        o = check(f"The `{name}` function accepts `s`.", code(), ClaimType.PARAMETER)
        assert o.verdict == V.NOT_CHECKABLE and o.skip_reason == "external" and not o.findings and o.confidence is None


def test_aliases_and_reexports_never_produce_contradictions():
    o = check("The `exported` function accepts `a`.", code(), ClaimType.PARAMETER, aliases={"exported"})
    assert o.verdict == V.INSUFFICIENT_EVIDENCE and "import" in o.rationale


def test_ambiguous_names_are_not_guessed():
    ents = code() + [entity("pkg.other.Store.enqueue", type=EntityType.METHOD, params=[P("self"), P("item")])]
    o = check("The `enqueue` method accepts `payload`.", ents, ClaimType.PARAMETER)
    assert o.verdict == V.INSUFFICIENT_EVIDENCE and "ambiguous" in o.rationale and "pkg.other.Store.enqueue" in o.rationale


def test_class_documented_as_function_or_vice_versa_is_flagged():
    o = check("See the `compute_backoff` class.", code(), ClaimType.ENTITY_REFERENCE)
    assert o.verdict == V.CONTRADICTED and "not a class" in o.rationale


def test_no_indexed_python_source_means_unverified():
    o = check("Call `retry_all()` now.", [], ClaimType.ENTITY_REFERENCE)
    assert o.verdict == V.INSUFFICIENT_EVIDENCE and "no Python source" in o.rationale


def test_example_code_that_defines_its_own_function_is_not_checkable():
    md = "```python\ndef my_handler(task):\n    ...\n```\n"
    o = verifier(code()).verify(only(md_claims(md), ClaimType.SIGNATURE))
    assert o.verdict == V.NOT_CHECKABLE and o.skip_reason == "example_code"
    md2 = "```python\ndef enqueue(a, b, c):\n    ...\n```\n"                       # only a *method* has that name
    assert verifier(code()).verify(only(md_claims(md2), ClaimType.SIGNATURE)).skip_reason == "example_code"
    md3 = "```python\ndef compute_backoff(attempt, base_seconds=2.0, jitter=True) -> float:\n    ...\n```\n"
    assert verifier(code()).verify(only(md_claims(md3), ClaimType.SIGNATURE)).verdict == V.SUPPORTED


# ------------------------------------------------------------------------------ parameters
def test_correct_parameter_claim_is_supported_and_a_subset_is_fine():
    both = check("The `compute_backoff` function accepts `attempt` and `jitter`.", code(), ClaimType.PARAMETER)
    one = check("The `compute_backoff` function accepts `attempt`.", code(), ClaimType.PARAMETER)
    assert both.verdict == one.verdict == V.SUPPORTED and not both.findings and not one.findings


def test_wrong_parameter_name_is_contradicted_and_lists_the_actual_parameters():
    o = check("The `compute_backoff` function accepts `attempt` and `base_delay`.", code(), ClaimType.PARAMETER)
    assert o.verdict == V.CONTRADICTED
    f = findings(o)[0]
    assert f.issue_type == I.PARAMETER_NAME_MISMATCH and f.severity == S.HIGH and "base_delay" in f.message
    assert "`attempt`, `base_seconds`, `jitter`" in f.message and "compute_backoff(attempt: int, base_seconds: float = 2.0, jitter=True)" in f.suggestion


def test_close_names_get_a_did_you_mean():
    o = check("The `compute_backoff` function accepts `base_second`.", code(), ClaimType.PARAMETER)
    assert "Did you mean `base_seconds`?" in findings(o)[0].suggestion


def test_completeness_claims_check_omissions_by_severity():
    only_a = check("The `compute_backoff` function accepts only `attempt`.", code(), ClaimType.PARAMETER)
    assert only_a.verdict == V.CONTRADICTED and [f.severity for f in only_a.findings] == [S.LOW, S.LOW]    # optional params omitted
    f2 = [entity("pkg.mod.f", params=[P("a"), P("b")])]
    required = check("The `f` function accepts only `a`.", f2, ClaimType.PARAMETER)
    assert findings(required)[0].issue_type == I.PARAMETER_COUNT_MISMATCH and findings(required)[0].severity == S.HIGH


def test_documented_defaults_are_compared_after_normalisation():
    ok = check("The `compute_backoff` function accepts `base_seconds=2.00`.", code(), ClaimType.PARAMETER)
    bad = check("The `compute_backoff` function accepts `base_seconds=5`.", code(), ClaimType.PARAMETER)
    req = check("The `compute_backoff` function accepts `attempt=3`.", code(), ClaimType.PARAMETER)
    assert ok.verdict == V.SUPPORTED
    assert bad.verdict == V.CONTRADICTED and findings(bad)[0].issue_type == I.SIGNATURE_DEFAULT_MISMATCH and findings(bad)[0].severity == S.HIGH
    assert req.verdict == V.CONTRADICTED and "required" in findings(req)[0].message


def test_variadic_parameters_are_respected():
    ents = [entity("pkg.mod.f", params=[P("a"), P("args", kind=K.VAR_POSITIONAL), P("opts", kind=K.VAR_KEYWORD)]),
            entity("pkg.mod.g", params=[P("a")])]
    assert check("The `f` function accepts `a` and `*args`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED
    assert check("The `f` function accepts `a` and `extra`.", ents, ClaimType.PARAMETER).verdict == V.INSUFFICIENT_EVIDENCE   # **opts may absorb it
    missing = check("The `g` function accepts `a` and `*items`.", ents, ClaimType.PARAMETER)
    assert missing.verdict == V.CONTRADICTED and "variadic positional" in findings(missing)[0].message
    assert check("The `g` function accepts `**options`.", ents, ClaimType.PARAMETER).verdict == V.CONTRADICTED


def test_self_and_cls_are_never_reported_or_required():
    ents = [module("pkg.mod"), entity("pkg.mod.Q", type=EntityType.CLASS),
            entity("pkg.mod.Q.m", type=EntityType.METHOD, params=[P("self"), P("a"), P("b", "1")]),
            entity("pkg.mod.Q.c", type=EntityType.METHOD, params=[P("cls"), P("a")], decorators=["classmethod"]),
            entity("pkg.mod.Q.s", type=EntityType.METHOD, params=[P("a")], decorators=["staticmethod"])]
    assert check("The `Q.m` method accepts only `a` and `b`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED
    assert check("The `Q.m` method accepts `self` and `a`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED       # documenting self is harmless
    assert check("The `Q.c` method accepts only `a`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED
    assert check("The `Q.s` method accepts only `a`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED
    assert check("The `Q.m` method accepts `cls`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED


def test_constructor_parameters_come_from_init_and_missing_init_is_unverified():
    ok = check("The `TaskQueue` class accepts `backend` and `max_size`.", code(), ClaimType.PARAMETER)
    bad = check("The `TaskQueue` class accepts `capacity`.", code(), ClaimType.PARAMETER)
    none = check("The `NoInit` class accepts `value`.", code(), ClaimType.PARAMETER)
    assert ok.verdict == V.SUPPORTED and bad.verdict == V.CONTRADICTED
    assert none.verdict == V.INSUFFICIENT_EVIDENCE and "dataclass" in none.rationale


# ----------------------------------------------------------------------------- signatures
SIG = "The signature is `{}`."


def sig(text, ents=None):
    return verifier(ents or code()).verify(only(md_claims(SIG.format(text)), ClaimType.SIGNATURE))


def test_exact_def_form_signature_is_supported():
    o = sig("def compute_backoff(attempt: int, base_seconds: float = 2.0, jitter=True) -> float")
    assert o.verdict == V.SUPPORTED and not o.findings and o.confidence == 1.0


def test_signature_errors_are_critical_when_callers_would_break():
    extra = sig("def compute_backoff(attempt, base_delay)")
    kinds = {(f.issue_type, f.severity) for f in extra.findings}
    assert extra.verdict == V.CONTRADICTED and (I.PARAMETER_NAME_MISMATCH, S.CRITICAL) in kinds
    omitted = sig("def f(a)", [entity("pkg.mod.f", params=[P("a"), P("b")])])
    assert [(f.issue_type, f.severity) for f in omitted.findings] == [(I.PARAMETER_COUNT_MISMATCH, S.CRITICAL)]
    nonexistent = verifier(code()).verify(only(md_claims("It is `def pkg.mod.gone(a, b)`."), ClaimType.SIGNATURE))
    assert nonexistent.verdict == V.CONTRADICTED and nonexistent.findings[0].severity == S.CRITICAL


def test_signature_defaults_annotations_returns_and_order():
    d = sig("def compute_backoff(attempt, base_seconds=3.0, jitter=True)")
    assert [(f.issue_type, f.severity) for f in d.findings] == [(I.SIGNATURE_DEFAULT_MISMATCH, S.HIGH)] and "3.0" in d.findings[0].message
    a = sig("def compute_backoff(attempt: str, base_seconds=2.0, jitter=True)")
    assert a.findings[0].issue_type == I.SIGNATURE_DEFAULT_MISMATCH and "`str`" in a.findings[0].message
    r = sig("def compute_backoff(attempt, base_seconds=2.0, jitter=True) -> int")
    assert r.verdict == V.CONTRADICTED and r.findings[0].issue_type == I.RETURN_MISMATCH
    o = sig("def compute_backoff(base_seconds=2.0, attempt=1, jitter=True)")
    assert any("different order" in f.message for f in o.findings)


def test_keyword_only_and_positional_only_conflicts_break_callers():
    ents = [entity("pkg.mod.f", params=[P("a"), P("b", kind=K.KEYWORD_ONLY)]), entity("pkg.mod.g", params=[P("a", kind=K.POSITIONAL_ONLY), P("b")])]
    assert any("only by keyword" in f.message for f in sig("def f(a, b)", ents).findings)
    assert sig("def f(a, *, b)", ents).verdict == V.SUPPORTED
    assert any("only by position" in f.message for f in sig("def g(*, a, b)", ents).findings)
    assert sig("def f(a, *, b)", ents).verdict == V.SUPPORTED


def test_call_like_mentions_are_checked_conservatively():
    def call(text):
        return verifier(code()).verify(only(md_claims(f"Use `{text}` to enqueue."), ClaimType.SIGNATURE))
    assert call("TaskQueue.enqueue(payload, priority=1)").verdict == V.SUPPORTED
    assert call("TaskQueue.enqueue(task)").verdict == V.SUPPORTED                        # placeholder names may be any variable
    bad_kw = call("TaskQueue.enqueue(payload, urgency=1)")
    assert bad_kw.verdict == V.CONTRADICTED and bad_kw.findings[0].severity == S.HIGH      # never CRITICAL for a call-like mention
    too_many = call("TaskQueue.enqueue(a, b, c, d)")
    assert too_many.findings[0].issue_type == I.PARAMETER_COUNT_MISMATCH and "at most 3" in too_many.findings[0].message
    assert call("compute_backoff(attempt, base_seconds=2)").findings == []                # defaults in call-like forms are values, not compared


# --------------------------------------------------------------------------------- returns
@pytest.mark.parametrize("sentence, verdict", [
    ("The `compute_backoff` function returns a float.", V.SUPPORTED),
    ("The `compute_backoff` function returns an integer.", V.CONTRADICTED),
    ("The `retry_failed` function returns `int`.", V.SUPPORTED),
    ("The `retry_failed` function returns nothing.", V.CONTRADICTED),
    ("The `untyped` function returns a string.", V.INSUFFICIENT_EVIDENCE),             # no annotation: never invent a type
    ("The `TaskQueue.purge` method returns an int.", V.SUPPORTED),
    ("The `TaskQueue.enqueue` method returns a dict.", V.INSUFFICIENT_EVIDENCE),       # annotation is a user type: undecidable
    ("The `TaskQueue` class returns a list.", V.INSUFFICIENT_EVIDENCE),                # constructors return instances
])
def test_return_claims(sentence, verdict):
    assert check(sentence, code(), ClaimType.RETURN).verdict == verdict


def test_missing_return_annotation_is_explained_not_guessed():
    o = check("The `untyped` function returns a string.", code(), ClaimType.RETURN)
    assert "no return annotation" in o.findings[0].message and o.findings[0].severity == S.INFO
    assert "not evidence that the claim is false" in o.findings[0].suggestion


def test_return_contradiction_cites_the_annotation():
    o = check("The `compute_backoff` function returns an integer.", code(), ClaimType.RETURN)
    f = o.findings[0]
    assert f.issue_type == I.RETURN_MISMATCH and f.severity == S.HIGH and "-> float" in f.message and "`float`" in f.suggestion


def test_function_named_like_a_type_does_not_confuse_return_checks():
    ents = [entity("pkg.mod.f", ret="Optional[int]"), entity("pkg.mod.g", ret="list[str]")]
    assert check("The `f` function returns an integer.", ents, ClaimType.RETURN).verdict == V.INSUFFICIENT_EVIDENCE   # int vs int | None
    assert check("The `g` function returns a list.", ents, ClaimType.RETURN).verdict == V.SUPPORTED


# -------------------------------------------------------------------------------- docstrings
DOC = "Compute.\n\nArgs:\n    attempt (int): n\n    ghost: not real\n\nReturns:\n    int: the delay\n"


def test_docstring_claims_are_verified_against_the_same_entity():
    e = code()[1]
    e.docstring = DOC
    claims = extract_docstring_claims([e], {1: "pkg/mod.py"}).claims
    v = verifier(code())
    p = v.verify(next(c for c in claims if c.claim_type == ClaimType.PARAMETER))
    r = v.verify(next(c for c in claims if c.claim_type == ClaimType.RETURN))
    assert p.verdict == V.CONTRADICTED and "ghost" in findings(p, I.PARAMETER_NAME_MISMATCH)[0].message
    assert r.verdict == V.CONTRADICTED and findings(r)[0].issue_type == I.RETURN_MISMATCH
    assert findings(p, I.MISSING_DOCUMENTATION)                                    # base_seconds / jitter missing from the Args section


def test_a_parameters_section_that_omits_parameters_is_a_low_coverage_finding_not_a_contradiction():
    e = entity("pkg.mod.h", params=[P("a"), P("b")], doc="Doc.\n\nArgs:\n    a: first\n")
    claim = extract_docstring_claims([e], {1: "pkg/mod.py"}).claims[0]
    o = verifier([module("pkg.mod"), e]).verify(claim)
    assert o.verdict == V.SUPPORTED and [(f.issue_type, f.severity) for f in o.findings] == [(I.MISSING_DOCUMENTATION, S.LOW)]
    assert "`b`" in o.findings[0].message


def test_findings_are_self_contained_snapshots():
    o = check("The `compute_backoff` function accepts `base_delay`.", code(), ClaimType.PARAMETER)
    d = o.findings[0].details
    assert d["claim_text"].startswith("The `compute_backoff`") and d["doc_location"]["label"] == "README.md:1"
    assert d["code_location"] == {"source_name": "pkg/mod.py", "start_line": 10, "end_line": 20,
                                  "qualified_name": "pkg.mod.compute_backoff", "label": "pkg/mod.py:10"}
    assert d["status"] == "contradicted" and d["category"] == "parameter" and len(d["finding_uid"]) == 16 and d["model_assisted"] is False


def test_finding_ids_are_stable_across_runs():
    a = check("The `compute_backoff` function accepts `base_delay`.", code(), ClaimType.PARAMETER)
    b = check("The `compute_backoff` function accepts `base_delay`.", code(), ClaimType.PARAMETER)
    assert a.findings[0].details["finding_uid"] == b.findings[0].details["finding_uid"]


# --------------------------------------------------------------------------------- coverage
SRC = b'''class Public:
    def method(self, x): ...
    def noargs(self): ...
    def _private(self): ...
    def __init__(self): ...

def func(a): ...

def documented(a):
    """Has docs."""

def _hidden(): ...

def outer():
    """Documented."""
    def inner(): ...
'''


def analyzed():
    return [e.entity for e in analyze_python(SRC, "pkg/m.py").entities]


def test_coverage_respects_the_analyzers_public_private_and_dunder_rules():
    ents = analyzed()
    for i, e in enumerate(ents, start=1):
        e.id, e.source_id = i, 1
    r = EntityResolver(ents)
    found, stats = check_documentation_coverage(ents, r, {"func"}, {1: "pkg/m.py"})
    flagged = {f.details["code_location"]["qualified_name"] for f in found}
    assert flagged == {"pkg.m.Public", "pkg.m.Public.method", "pkg.m.Public.noargs"}
    assert stats == {"public_entities": 6, "documented_entities": 3}              # documented + outer (docstring) + func (mentioned)
    assert all(f.issue_type == I.MISSING_DOCUMENTATION and f.claim_id is None and f.details["category"] == "coverage" for f in found)


def test_missing_documentation_severity_depends_on_what_is_undocumented():
    ents = analyzed()
    for i, e in enumerate(ents, start=1):
        e.id, e.source_id = i, 1
    found, _ = check_documentation_coverage(ents, EntityResolver(ents), set(), {1: "pkg/m.py"})
    by = {f.details["code_location"]["qualified_name"]: f.severity for f in found}
    assert by["pkg.m.Public"] == S.MEDIUM and by["pkg.m.Public.method"] == S.MEDIUM and by["pkg.m.Public.noargs"] == S.LOW
    assert "Source signature: `method(x)`" in next(f for f in found if f.details["code_location"]["qualified_name"] == "pkg.m.Public.method").suggestion


def test_ambiguous_mentions_do_not_count_as_documentation():
    ents = [module("pkg.a"), entity("pkg.a.run", params=[P("x")], id=2, source_id=1), entity("pkg.b.run", params=[P("y")], id=3, source_id=1)]
    found, stats = check_documentation_coverage(ents, EntityResolver(ents), {"run"}, {1: "x.py"})
    assert stats == {"public_entities": 2, "documented_entities": 0} and len(found) == 2


def test_no_public_entities_means_no_coverage_data():
    found, stats = check_documentation_coverage([module("pkg")], EntityResolver([module("pkg")]), set(), {})
    assert found == [] and stats == {"public_entities": 0, "documented_entities": 0}


def test_render_params_round_trips_markers():
    assert render_params([P("a", kind=K.POSITIONAL_ONLY), P("b"), P("c", "1", kind=K.KEYWORD_ONLY), P("kw", kind=K.VAR_KEYWORD)]) == "a, /, b, *, c=1, **kw"


# ----------------------------------------------------------------- decorators may change signatures
def test_unknown_decorators_make_signature_claims_unverified_not_contradicted():
    ents = [module("pkg.mod"), entity("pkg.mod.cli", params=[P("ctx")], decorators=["click.command()", "click.option('--n')"], id=2)]
    o = check("The `cli` function accepts `n` and `name`.", ents, ClaimType.PARAMETER)
    assert o.verdict == V.INSUFFICIENT_EVIDENCE and "decorated with `@click.command()`" in o.rationale and o.findings[0].severity == S.INFO
    sig = verifier(ents).verify(only(md_claims("It is `def cli(n, name)`."), ClaimType.SIGNATURE))
    assert sig.verdict == V.INSUFFICIENT_EVIDENCE                                      # even a def-form signature: the decorator wins


@pytest.mark.parametrize("decorator", ["lru_cache(maxsize=1)", "functools.cache", "staticmethod", "contextmanager", "abc.abstractmethod", "x.setter"])
def test_signature_preserving_decorators_are_still_verified(decorator):
    ents = [module("pkg.mod"), entity("pkg.mod.f", params=[P("a")], decorators=[decorator], id=2)]
    assert check("The `f` function accepts `a`.", ents, ClaimType.PARAMETER).verdict == V.SUPPORTED
    assert check("The `f` function accepts `b`.", ents, ClaimType.PARAMETER).verdict == V.CONTRADICTED


def test_a_decorated_constructor_is_also_unverified():
    ents = [module("pkg.mod"), entity("pkg.mod.C", type=EntityType.CLASS, id=2),
            entity("pkg.mod.C.__init__", type=EntityType.METHOD, params=[P("self"), P("a")], decorators=["register_init"], public=False, id=3)]
    assert check("The `C` class accepts `zzz`.", ents, ClaimType.PARAMETER).verdict == V.INSUFFICIENT_EVIDENCE
