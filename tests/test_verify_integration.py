"""Ingestion (Part 2) -> retrieval (Part 3, stand-in models) -> verification (Part 5), end to end.

Retrieval uses deterministic lexical stand-ins and the LLM is a scripted mock, so nothing here measures
real-model behaviour or downloads anything.
"""

import json

import pytest
from ask_support import ScriptedLLM

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")
pytest.importorskip("pymupdf")

from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset  # noqa: E402
from trustdoc.models import ClaimOrigin, ClaimType, ClaimVerdict as V, Project, Severity as S  # noqa: E402
from trustdoc.retrieval import HashingEmbedder, OverlapReranker, create_retriever  # noqa: E402
from trustdoc.verification import VerificationEngine, render_report  # noqa: E402

GOOD = """The `compute_backoff` function accepts `attempt`, `base_seconds` and `jitter`, and returns a float.

Use `enqueue(payload, priority=0, delay_seconds=0.0)` to add a task.

The `TaskQueue` class accepts `backend` and `max_size`.

The `heartbeat` function returns a float.

The `SqliteBackend.compact` method accepts `vacuum` and returns an integer.

The signature is `def fetch_pending(limit: int = 100) -> list[dict]`.
"""
BAD = """The `compute_backoff` function accepts `attempt` and `base_delay`, and returns an integer.

Call `retry_all()` to requeue everything.

The signature is `def enqueue(payload, priority=1)`.
"""


def reply(verdict="supported", ids=("E1",), explanation="Stated in the cited code."):
    return json.dumps({"verdict": verdict, "explanation": explanation, "evidence_ids": list(ids), "suggested_correction": None})


@pytest.fixture
def sample(db, settings, service):
    with db.session() as s:
        pid = s.projects.create(Project(name="sample")).id
    assert service.ingest_directory(pid, load_dataset(DEFAULT_QUERIES).project_dir).ok
    return pid


@pytest.fixture
def engine(db, settings):
    return VerificationEngine(db, settings)


def semantic_engine(db, settings, llm):
    retriever = create_retriever(db, settings, embedder=HashingEmbedder(), reranker=OverlapReranker())
    return VerificationEngine(db, settings, retriever=retriever, llm=llm)


# ----------------------------------------------------------------- false-positive guards
def test_the_bundled_accurate_readme_produces_no_defects_and_is_not_given_a_score(sample, engine):
    r = engine.verify_project(sample)
    assert r.max_severity() is None and r.counts["contradicted"] == 0
    assert all(c.status == V.NOT_CHECKABLE for c in r.claims) and r.counts["claims"] == r.counts["not_checkable"] > 0
    assert r.score.final is None and any("No claim was checked" in n for n in r.score.notes)    # never "100/100" for unchecked docs
    cov = next(c for c in r.score.components if c.name == "coverage")
    assert cov.weight == 0 and cov.raw_score == 1.0 and "informational" in cov.detail


def test_accurate_documentation_is_fully_supported_with_no_false_positives(sample, engine):
    r = engine.verify_text(sample, GOOD, name="generated/good.md")
    assert r.counts["claims"] == 8 and r.counts["supported"] == 8 and r.findings == []
    assert r.score.final == 100.0 and all(c.evidence for c in r.claims) and all(c.origin == ClaimOrigin.GENERATED.value for c in r.claims)


def test_flawed_documentation_is_caught_with_actionable_findings(sample, engine):
    r = engine.verify_text(sample, BAD, name="generated/bad.md")
    assert r.counts["contradicted"] == 4 and r.counts["supported"] == 0
    kinds = {(f.issue_type, f.severity) for f in r.findings}
    assert {("signature_default_mismatch", S.HIGH), ("nonexistent_entity", S.HIGH), ("parameter_name_mismatch", S.HIGH),
            ("return_mismatch", S.HIGH), ("parameter_count_mismatch", S.LOW)} == kinds
    assert next(f for f in r.findings if f.issue_type == "nonexistent_entity").suggestion.startswith("Did you mean `retry_failed`?")
    assert r.score.final == pytest.approx(5.0)                       # accuracy 0, grounding 3/4, 4 HIGH contradictions (-20)
    text = render_report(r)
    assert "[HIGH] 4" in text and "documented as returning `int`" in text and "Trust score: 5.0 / 100" in text


# --------------------------------------------------------------------------- semantic tier
def test_semantic_support_uses_code_evidence_and_never_the_claims_own_document(db, settings, sample):
    llm = ScriptedLLM(reply("supported", ["E1"]))
    r = semantic_engine(db, settings, llm).verify_text(sample, "The `SqliteBackend` class persists tasks to disk.\n",
                                                       name="generated/claim.md", semantic=True)
    behavior = next(c for c in r.claims if c.category == ClaimType.BEHAVIORAL)
    assert behavior.status == V.SUPPORTED and behavior.model_assisted and behavior.tier.value == "semantic" and behavior.confidence is None
    assert behavior.explanation.startswith("[model-assisted] ")
    labels = [e.label for e in behavior.evidence]
    assert labels and not any(lbl.startswith("generated/claim.md") or lbl.startswith("README.md") for lbl in labels)   # no documentation as evidence
    assert any(lbl.startswith("taskqueue/storage.py") for lbl in labels) and any(e.stance == "supports" for e in behavior.evidence)
    prompt = llm.calls[0]["prompt"]
    assert "BEGIN CLAIM" in prompt and "BEGIN EVIDENCE E1" in prompt and "persists tasks to disk" in prompt


def test_semantic_contradictions_are_flagged_as_model_assisted_and_not_penalised_like_static_ones(db, settings, sample):
    llm = ScriptedLLM(reply("contradicted", ["E1"], "The code writes to SQLite, not memory."))
    r = semantic_engine(db, settings, llm).verify_text(sample, "The `SqliteBackend` class stores tasks only in memory.\n",
                                                       name="generated/mem.md", semantic=True)
    f = next(x for x in r.findings if x.issue_type == "contradicted_claim")
    assert f.severity == S.MEDIUM and f.model_assisted and f.tier.value == "semantic" and "LLM-assisted" in f.message
    assert r.counts["contradicted"] == 1 and r.score.penalties == []                    # penalties are for CRITICAL/HIGH only
    assert "(model-assisted)" in render_report(r) and "not ground truth" in render_report(r)


def test_fabricated_or_malformed_model_output_never_becomes_a_verdict(db, settings, sample):
    for output in (reply("supported", ["E9"]), "I am sure it is fine.", reply("maybe")):
        r = semantic_engine(db, settings, ScriptedLLM(output)).verify_text(
            sample, "The `SqliteBackend` class persists tasks to disk.\n", name="generated/x.md", semantic=True)
        c = next(x for x in r.claims if x.category == ClaimType.BEHAVIORAL)
        assert c.status == V.INSUFFICIENT_EVIDENCE and "rejected" in c.explanation
        assert all(e.stance == "related" for e in c.evidence)


def test_pdf_reference_material_can_serve_as_evidence(db, settings, sample):
    llm = ScriptedLLM(reply("supported", ["E1"]))
    r = semantic_engine(db, settings, llm).verify_text(
        sample, "Completed tasks are retained for seven days before compaction deletes them.\n", name="generated/pdf.md", semantic=True)
    c = next(x for x in r.claims if x.category in (ClaimType.BEHAVIORAL, ClaimType.CONFIG_DEPENDENCY))
    assert any(e.label == "reference.pdf, p.3" for e in c.evidence)


def test_semantic_evidence_never_crosses_projects(db, settings, service, sample):
    with db.session() as s:
        other = s.projects.create(Project(name="other")).id
    for i in range(8):
        service.ingest_bytes(other, f"o{i}.py", f"def other_fn_{i}(x):\n    '''Does thing {i}.'''\n    return x\n".encode())
    llm = ScriptedLLM(reply("insufficient_evidence", [], "Nothing relevant."))
    r = semantic_engine(db, settings, llm).verify_text(other, "The `SqliteBackend` class persists tasks to disk.\n", name="generated/q.md", semantic=True)
    assert all("taskqueue" not in call["prompt"] and "compute_backoff" not in call["prompt"] for call in llm.calls)
    c = next(x for x in r.claims if x.category == ClaimType.BEHAVIORAL)
    assert c.status == V.INSUFFICIENT_EVIDENCE and all(not e.label.startswith("taskqueue/") for e in c.evidence)


def test_a_missing_model_leaves_the_deterministic_tier_fully_usable(db, settings, sample):
    engine = semantic_engine(db, settings, None)                 # provider "none"
    r = engine.verify_text(sample, BAD + "\nThe `SqliteBackend` class persists tasks to disk.\n", name="generated/mix.md", semantic=True)
    assert r.counts["contradicted"] == 4 and r.counts["not_checkable"] == 1 and r.run.status.value == "completed"
    assert any("unavailable" in s.lower() for s in r.skipped) and r.score.final is not None


def test_history_and_reports_are_consistent_through_the_whole_stack(db, settings, sample, engine):
    a = engine.verify_text(sample, BAD, name="generated/bad.md")
    b = engine.verify_text(sample, GOOD, name="generated/good.md")
    assert [r.id for r in engine.list_runs(sample)] == [b.run.id, a.run.id]
    assert engine.load_report(a.run.id).to_dict() == a.to_dict() and engine.load_report(b.run.id).to_dict() == b.to_dict()
    assert a.score.final < b.score.final
