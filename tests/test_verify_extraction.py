import pytest
from verify_support import entity, md_claims, of_type, only

from trustdoc.models import ClaimOrigin, ClaimType, EntityType
from trustdoc.verification.extraction import (
    extract_docstring_claims,
    extract_markdown_claims,
    parse_docstring,
    split_sentences,
)
from trustdoc.verification.types import claim_uid

SPEC = """The `compute_backoff` function accepts `attempt` and `base_delay` and returns a float.

The queue retries failed jobs three times.

The `TaskQueue` class persists jobs to disk.
"""


def test_the_specs_own_examples_are_extracted():
    claims = md_claims(SPEC)
    p = only(claims, ClaimType.PARAMETER)
    assert p.subject_name == "compute_backoff" and [d["name"] for d in p.extracted["documented_params"]] == ["attempt", "base_delay"]
    assert p.extracted["complete"] is False and p.extracted["form"] == "prose"
    r = only(claims, ClaimType.RETURN)
    assert r.subject_name == "compute_backoff" and r.extracted["documented_return"] == "float"
    behaviors = of_type(claims, ClaimType.BEHAVIORAL)
    assert {c.text for c in behaviors} == {"The queue retries failed jobs three times.", "The `TaskQueue` class persists jobs to disk."}
    persists = next(c for c in behaviors if "persists" in c.text)
    assert persists.subject_name == "TaskQueue"
    assert only(claims, ClaimType.ENTITY_REFERENCE).extracted["entity_name"] == "TaskQueue"      # "the `TaskQueue` class"


def test_every_claim_records_origin_method_confidence_and_location():
    for c in md_claims(SPEC, "docs/guide.md"):
        assert c.origin == ClaimOrigin.EXISTING_DOC and c.source_id == 0          # ids are stamped later by the engine
        assert c.extracted["method"].split(":")[0] in ("rule", "ast", "docstring")
        assert 0 < c.extracted["confidence"] <= 1 and c.extracted["doc_source_name"] == "docs/guide.md"
        assert c.extracted["doc_label"] == f"docs/guide.md:{c.start_line}" and claim_uid(c)


def test_generated_origin_can_be_requested():
    claims = extract_markdown_claims("The queue retries failed jobs.", "gen.md", origin=ClaimOrigin.GENERATED).claims
    assert claims and all(c.origin == ClaimOrigin.GENERATED for c in claims)


def test_source_lines_are_exact_for_paragraphs_lists_and_headings():
    md = "# T\n\nIntro text that says nothing checkable.\nStill intro.\n\nThe queue retries failed jobs\nthree times.\n\n- The `Store` class persists jobs to disk.\n"
    claims = md_claims(md)
    assert {c.text: c.start_line for c in claims if c.claim_type == ClaimType.BEHAVIORAL} == {
        "The queue retries failed jobs three times.": 6, "The `Store` class persists jobs to disk.": 9}


def test_ids_are_reproducible_and_independent_of_line_numbers():
    a = md_claims(SPEC)
    b = md_claims(SPEC)
    assert [claim_uid(c) for c in a] == [claim_uid(c) for c in b]
    shifted = md_claims("# New heading\n\nSome intro.\n\n" + SPEC)
    assert sorted(claim_uid(c) for c in a) == sorted(claim_uid(c) for c in shifted)           # lines moved, identities did not
    other = md_claims(SPEC, "other.md")
    assert not set(claim_uid(c) for c in a) & set(claim_uid(c) for c in other)               # different document, different identity


def test_identical_sentences_get_distinct_ids():
    twice = md_claims("The queue retries failed jobs.\n\nThe queue retries failed jobs.\n")
    uids = [claim_uid(c) for c in twice]
    assert len(uids) == 2 and len(set(uids)) == 2


@pytest.mark.parametrize("sentence", [
    "Install the library with `pip install taskqueue-lite`.",          # imperative
    "You can retry failed tasks manually.",                            # second person
    "The queue may retry failed tasks.",                               # hedged
    "Does the queue retry failed tasks?",                              # question
    "When a task fails, it is retried.",                               # conditional
    "Call `retry_failed` to requeue them.",                            # imperative
    "See the docs for details.",                                       # no behaviour verb
    "It is fast.",                                                     # too short / no verb
    "Retries three times.",                                            # no subject
])
def test_non_claims_are_ignored(sentence):
    assert [c for c in md_claims(sentence) if c.claim_type == ClaimType.BEHAVIORAL] == []


def test_config_dependency_claims():
    claims = md_claims("`TASKQ_WORKER_COUNT` sets the number of workers.\n\nThe environment variable overrides the file.\n")
    cfg = of_type(claims, ClaimType.CONFIG_DEPENDENCY)
    assert len(cfg) >= 1 and "TASKQ_WORKER_COUNT" in cfg[0].text


def test_code_blocks_tables_and_headings_are_not_prose_claims():
    md = ("# The queue retries failed jobs\n\n```text\nThe queue retries failed jobs three times.\n```\n\n"
          "| Name | Description |\n|---|---|\n| a | The queue retries failed jobs. |\n")
    assert md_claims(md) == []


def test_front_matter_is_ignored():
    md = "---\ntitle: The queue retries failed jobs\n---\nPlain intro.\n"
    assert md_claims(md) == []


def test_sentence_splitting_respects_code_spans_and_abbreviations():
    parts = [s for _, s in split_sentences("Use `a. b` e.g. like so. Then `c.d()` runs. It ends.")]
    assert parts == ["Use `a. b` e.g. like so.", "Then `c.d()` runs.", "It ends."]


# ---------------------------------------------------------------------------- signatures
def test_def_form_signature_is_complete_and_parsed():
    c = only(md_claims("The signature is `def compute_backoff(attempt, base=2.0, *, jitter=True) -> float`."), ClaimType.SIGNATURE)
    ex = c.extracted
    assert ex["complete"] is True and ex["form"] == "def" and ex["documented_return"] == "float" and ex["confidence"] == 1.0
    assert [(p["name"], p["kind"], p["default"]) for p in ex["documented_params"]] == [
        ("attempt", "positional_or_keyword", None), ("base", "positional_or_keyword", "2.0"), ("jitter", "keyword_only", "True")]


def test_call_like_signature_marks_positional_placeholders():
    c = only(md_claims("Add work with `enqueue(payload, priority=0)`."), ClaimType.SIGNATURE)
    params = {p["name"]: p for p in c.extracted["documented_params"]}
    assert c.extracted["form"] == "call_like" and c.extracted["complete"] is False and c.extracted["confidence"] < 1
    assert params["payload"]["placeholder"] is True and params["priority"]["placeholder"] is False


@pytest.mark.parametrize("span", ["enqueue({'a': 1})", 'print("x")', "enqueue(3, 4)", "f(...)", "x = f(a)"])
def test_calls_with_values_or_placeholders_are_not_signatures(span):
    assert of_type(md_claims(f"Example: `{span}`."), ClaimType.SIGNATURE) == []


def test_empty_parentheses_and_kind_words_make_entity_claims_only_when_attributable():
    claims = md_claims("Call `retry_all()` to requeue. The `Store` class holds data. Run `pip` now. Set `max_size` higher.")
    names = {c.extracted["entity_name"] for c in of_type(claims, ClaimType.ENTITY_REFERENCE)}
    assert names == {"retry_all", "Store"}                          # bare `max_size` / `pip` are not entity claims


def test_fenced_code_definitions_become_signature_claims_but_not_nested_ones():
    md = ("```python\ndef heartbeat(worker_id, interval_seconds=30.0) -> float:\n    ...\n\n"
          "class Mine(Base):\n    def override(self, a):\n        ...\n```\n")
    sigs = of_type(md_claims(md), ClaimType.SIGNATURE)
    assert [c.extracted["entity_name"] for c in sigs] == ["heartbeat"]
    assert sigs[0].extracted["context"] == "fenced_code" and sigs[0].start_line == 2


@pytest.mark.parametrize("fence", ["```javascript\ndef f(a): pass\n```", "```python\ndef broken(:\n```", "```python\nx = 1\n```"])
def test_non_python_or_unparseable_blocks_yield_nothing(fence):
    assert md_claims(fence) == []


def test_unclosed_fence_runs_to_end_of_document_like_markdown_renderers_do():
    claims = md_claims("```python\ndef f(a): ...\n")
    assert [c.extracted["entity_name"] for c in claims] == ["f"]        # and, importantly, no exception


def test_parameters_section_lists_and_tables_under_an_entity_heading():
    md = ("### `compute_backoff`\n\n#### Parameters\n\n- `attempt` (int): the attempt\n- base_seconds: base delay\n\n"
          "### `heartbeat`\n\n#### Arguments\n\n| Name | Type |\n|---|---|\n| `worker_id` | str |\n")
    claims = of_type(md_claims(md), ClaimType.PARAMETER)
    by_entity = {c.extracted["entity_name"]: [p["name"] for p in c.extracted["documented_params"]] for c in claims}
    assert by_entity == {"compute_backoff": ["attempt", "base_seconds"], "heartbeat": ["worker_id"]}
    assert all(c.extracted["args_section"] and not c.extracted["complete"] for c in claims)


def test_plain_word_headings_do_not_become_entities():
    md = "### Installation\n\n#### Parameters\n\n- `attempt`: x\n"
    assert md_claims(md) == []


def test_complete_marker_words_are_detected():
    c = only(md_claims("The `enqueue` method accepts only `payload` and `priority`."), ClaimType.PARAMETER)
    assert c.extracted["complete"] is True


def test_requires_needs_a_parameter_word():
    assert of_type(md_claims("The `Store` class requires `numpy` and `pandas`."), ClaimType.PARAMETER) == []
    assert len(of_type(md_claims("The `Store` class requires the `path` argument."), ClaimType.PARAMETER)) == 1


# ------------------------------------------------------------------------ robustness
@pytest.mark.parametrize("text", ["", "   \n\n  ", "\x00\x01\x02", "#", "```", "| | |\n|-|-|", "`", "`` ` ``", "\u202e\u200b" * 50])
def test_degenerate_documents_never_crash_and_never_invent_claims(text):
    result = extract_markdown_claims(text, "x.md")
    assert result.claims == [] and result.warnings and "x.md" in result.warnings[0]


def test_a_document_without_checkable_content_warns():
    r = extract_markdown_claims("# Title\n\nJust a friendly introduction.\n", "x.md")
    assert r.claims == [] and "no verifiable claims" in r.warnings[0]


def test_mentions_collect_code_names_for_coverage():
    r = extract_markdown_claims("The `compute_backoff` function accepts `attempt`. See `taskqueue.workers.WorkerPool` and `pip`.", "x.md")
    assert {"compute_backoff", "taskqueue.workers.WorkerPool"} <= r.mentions and "pip" not in r.mentions


# -------------------------------------------------------------------------- docstrings
GOOGLE = '''Compute the delay.

    Args:
        attempt (int): The attempt number.
        base_seconds: Base delay.
        *args: Extra.

    Returns:
        float: The delay in seconds.
    '''
NUMPY = '''Compute.

    Parameters
    ----------
    attempt : int
        The attempt.
    base_seconds : float, optional
        Base.

    Returns
    -------
    float
        The delay.
    '''
REST = ''':param int attempt: the attempt
    :param base_seconds: base delay
    :type base_seconds: float
    :returns: the delay
    :rtype: float
    '''


@pytest.mark.parametrize("doc, names, rtype", [
    (GOOGLE, ["attempt", "base_seconds", "args"], "float"),
    (NUMPY, ["attempt", "base_seconds"], "float"),
    (REST, ["attempt", "base_seconds"], "float"),
])
def test_docstring_styles(doc, names, rtype):
    parsed = parse_docstring(doc)
    assert [p["name"] for p in parsed["params"]] == names and parsed["return_type"] == rtype and parsed["has_params_section"]


def test_docstring_param_types_are_captured_when_valid():
    parsed = parse_docstring(GOOGLE)
    assert parsed["params"][0]["annotation"] == "int" and parsed["params"][1]["annotation"] is None


def test_docstring_claims_use_entity_provenance():
    e = entity("pkg.mod.compute_backoff", doc=GOOGLE, source_id=7, start=47)
    result = extract_docstring_claims([e], {7: "pkg/mod.py"})
    p, r = only(result.claims, ClaimType.PARAMETER), only(result.claims, ClaimType.RETURN)
    assert p.extracted["entity_name"] == "pkg.mod.compute_backoff" and p.extracted["exact_entity"] and p.extracted["args_section"]
    assert p.extracted["doc_label"] == "pkg/mod.py:47 (docstring of pkg.mod.compute_backoff)" and p.start_line == 47
    assert r.extracted["documented_return"] == "float" and p.extracted["context"] == "docstring"


def test_docstring_extraction_skips_modules_undocumented_and_unknown_sources():
    ents = [entity("pkg.mod", type=EntityType.MODULE, doc="Args:\n    a: x"), entity("pkg.mod.f", doc=None),
            entity("pkg.mod.g", doc="Just prose, no sections."), entity("pkg.mod.h", doc=GOOGLE, source_id=99)]
    assert extract_docstring_claims(ents, {1: "pkg/mod.py"}).claims == []
