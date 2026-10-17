import json
import re

import pytest
from ask_support import FakeRetriever, ScriptedLLM, make_search_result

from trustdoc.exceptions import DocumentSelectionError, NotFoundError, VerificationError
from trustdoc.models import ClaimOrigin, ClaimType, ClaimVerdict as V, Project, RunStatus, Severity as S
from trustdoc.verification import VerificationEngine, render_report
from trustdoc.verification.types import claim_uid

from verify_support import PY, README  # noqa: E402


@pytest.fixture
def corpus(db, project_id, make_service):
    svc = make_service()
    svc.ingest_bytes(project_id, "pkg/mod.py", PY)
    svc.ingest_bytes(project_id, "README.md", README)
    return project_id


@pytest.fixture
def engine(db, settings):
    return VerificationEngine(db, settings)


def statuses(report):
    return {c.status for c in report.claims}


def by_text(report, fragment, category=None):
    """First claim containing the text (optionally of one category; one sentence can yield several claims)."""
    return next(c for c in report.claims if fragment in c.text and (category is None or c.category == category))


# --------------------------------------------------------------------------- end to end
def test_full_run_reports_claims_findings_and_a_hand_checked_score(corpus, engine):
    r = engine.verify_project(corpus)
    c = r.counts
    assert (c["claims"], c["supported"], c["contradicted"], c["insufficient_evidence"], c["not_checkable"]) == (8, 3, 3, 0, 2)
    assert r.run.status == RunStatus.COMPLETED and r.documents == ["README.md"]
    assert by_text(r, "base_delay", ClaimType.PARAMETER).status == V.CONTRADICTED and by_text(r, "retry_all").status == V.CONTRADICTED
    assert by_text(r, "returns a float", ClaimType.RETURN).status == V.SUPPORTED and by_text(r, "integer").status == V.SUPPORTED
    behaviors = [x for x in r.claims if x.category == ClaimType.BEHAVIORAL]
    assert len(behaviors) == 2 and {b.status for b in behaviors} == {V.NOT_CHECKABLE}

    by_sev = r.findings_by_severity()
    assert [f.severity for f in r.findings][:3] == [S.HIGH, S.HIGH, S.HIGH] and len(by_sev[S.MEDIUM]) == 1 and len(by_sev[S.LOW]) == 3
    assert {f.issue_type for f in by_sev[S.MEDIUM]} == {"missing_documentation"}
    # accuracy 3/6, coverage 4/6, grounding 5/6 (the nonexistent `retry_all` has no evidence chunk); 3 HIGH contradictions
    assert r.score.inputs["checked"] == 6 and r.score.inputs["grounded"] == 5 and r.score.inputs["public_entities"] == 6
    assert r.score.base == pytest.approx(100 * (0.5 * 0.5 + 0.25 * 4 / 6 + 0.25 * 5 / 6), abs=1e-3)
    assert r.score.final == pytest.approx(62.5 - 15.0, abs=0.01)
    assert any("Semantic verification was not run" in s for s in r.skipped)


def test_findings_carry_documentation_and_code_locations_and_citations(corpus, engine, settings):
    r = engine.verify_project(corpus)
    f = next(x for x in r.findings if "base_delay" in x.message)
    assert f.doc_location["label"] == "README.md:3" and f.code_location["label"].startswith("pkg/mod.py:") and f.code_location["qualified_name"] == "pkg.mod.compute_backoff"
    assert f.claim_text.startswith("The `compute_backoff` function accepts") and f.suggestion and len(f.uid) == 16
    c = by_text(r, "base_delay", ClaimType.PARAMETER)
    assert c.evidence and c.evidence[0].label.startswith("pkg/mod.py:") and c.evidence[0].stance == "contradicts"
    blob = json.dumps(r.to_dict())
    assert "uploads" not in blob and not re.search(r"project_\d+/", blob) and str(settings.storage.data_dir) not in blob   # no physical paths
    assert f.uid in c.finding_uids


def test_docstring_checks_and_coverage_findings(corpus, engine):
    r = engine.verify_project(corpus)
    doc = next(c for c in r.claims if c.doc_label.endswith("(docstring of pkg.mod.compute_backoff)"))
    assert doc.status == V.CONTRADICTED and "ghost" in doc.explanation
    cov = [f for f in r.findings if f.category == "coverage"]
    assert {f.code_location["qualified_name"] for f in cov} == {"pkg.mod.TaskQueue.purge", "pkg.mod.undocumented"}
    assert {f.code_location["qualified_name"]: f.severity for f in cov} == {"pkg.mod.TaskQueue.purge": S.LOW, "pkg.mod.undocumented": S.MEDIUM}
    no_doc = engine.verify_project(corpus, include_docstrings=False)
    assert not any("docstring of" in c.doc_label for c in no_doc.claims) and no_doc.counts["claims"] == 7


def test_run_is_persisted_through_the_existing_repositories(corpus, engine, db):
    r = engine.verify_project(corpus)
    with db.session() as s:
        run = s.runs.require(r.run.id)
        claims = s.claims.list_for_project(corpus)
        results = s.findings.list_claim_results(run.id)
        findings = s.findings.list_for_run(run.id)
        breakdown = s.runs.get_trust_breakdown(run.id)
    assert run.status == RunStatus.COMPLETED and run.trust_score == r.score.final and run.finished_at and run.started_at
    assert len(claims) == len(results) == 8 and {c.extracted["claim_uid"] for c in claims} == {c.claim_uid for c in r.claims}
    assert [c.name for c in breakdown.components] == ["accuracy", "coverage", "grounding", "severity_penalty"] and breakdown.final_score == r.score.final
    assert len(findings) == len(r.findings) and all(f.details["finding_uid"] for f in findings)
    assert all(f.details["claim_text"] for f in findings if f.claim_id is not None)            # claim-linked findings snapshot their claim
    assert all(f.claim_id is None and f.details["category"] == "coverage" for f in findings if f.claim_id is None)
    assert run.summary["counts"]["claims"] == 8 and run.summary["score"]["final"] == r.score.final and run.config_snapshot["weights"]["accuracy"] == 0.5
    assert "api_key" not in json.dumps(run.config_snapshot).lower() and sum(len(x.evidence) for x in results) > 0


def test_reloaded_report_equals_the_returned_report(corpus, engine):
    r = engine.verify_project(corpus)
    assert engine.load_report(r.run.id).to_dict() == r.to_dict()


def test_markdown_claims_link_to_their_chunks_and_docstring_claims_to_entity_chunks(corpus, engine, db):
    engine.verify_project(corpus)
    with db.session() as s:
        for c in s.claims.list_for_project(corpus):
            assert c.chunk_id is not None
            chunk = s.chunks.get(c.chunk_id)
            if c.extracted["context"] == "docstring":
                assert chunk.entity_id is not None
            else:
                assert chunk.start_line <= c.start_line <= chunk.end_line


def test_the_text_report_is_readable(corpus, engine):
    text = render_report(engine.verify_project(corpus))
    for needle in ("Claims extracted: 8", "Trust score: 47.5 / 100", "accuracy: 50.0%", "[HIGH] 3", "Did you mean", "docs: README.md:3",
                   "Skipped checks:", "INSUFFICIENT EVIDENCE means unverified"):
        assert needle in text, needle


# ---------------------------------------------------------------------------- history
def test_every_run_is_kept_and_earlier_runs_are_not_modified(corpus, engine, db):
    first = engine.verify_project(corpus)
    with db.session() as s:
        before = (len(s.findings.list_for_run(first.run.id)), [r.verdict for r in s.findings.list_claim_results(first.run.id)])
        claim_count = len(s.claims.list_for_project(corpus))
    second = engine.verify_project(corpus)
    assert second.run.id > first.run.id and [r.id for r in engine.list_runs(corpus)] == [second.run.id, first.run.id]
    with db.session() as s:
        after = (len(s.findings.list_for_run(first.run.id)), [r.verdict for r in s.findings.list_claim_results(first.run.id)])
        assert len(s.claims.list_for_project(corpus)) == claim_count            # identical claims are reused, not duplicated
    assert before == after and engine.load_report(first.run.id).to_dict() == first.to_dict()
    assert [c.claim_uid for c in first.claims] == [c.claim_uid for c in second.claims]       # stable identity across runs
    assert {f.uid for f in first.findings} == {f.uid for f in second.findings}


def test_history_survives_when_documentation_is_replaced(corpus, engine, make_service):
    old = engine.verify_project(corpus)
    make_service().ingest_bytes(corpus, "README.md", b"# Demo\n\nThe `retry_failed` function returns a string.\n")
    new = engine.verify_project(corpus)
    assert any("string" in c.text for c in new.claims) and new.run.id != old.run.id
    reloaded = engine.load_report(old.run.id)
    assert reloaded.run.status == RunStatus.COMPLETED and reloaded.score.final == old.score.final
    # findings are self-contained snapshots, so the old run's defects stay readable even though the replaced
    # document's claim rows were removed by the existing cascade-on-replace behaviour
    assert {f.uid for f in reloaded.findings} == {f.uid for f in old.findings}
    assert all(f.claim_text for f in reloaded.findings if f.category != "coverage")


# ----------------------------------------------------------------------- selection/input
def test_document_selection_and_errors(corpus, engine, db, make_service):
    make_service().ingest_bytes(corpus, "docs/other.md", b"The `retry_failed` function returns a list.\n")
    only_other = engine.verify_project(corpus, documents=["docs/other.md"])
    assert only_other.documents == ["docs/other.md"] and only_other.counts["static_claims"] >= 1
    with pytest.raises(DocumentSelectionError, match="No source named"):
        engine.verify_project(corpus, documents=["missing.md"])
    with pytest.raises(DocumentSelectionError, match="only Markdown"):
        engine.verify_project(corpus, documents=["pkg/mod.py"])
    with pytest.raises(NotFoundError):
        engine.verify_project(98765)
    assert len(engine.list_runs(corpus)) == 1                                  # bad input never creates a run


def test_projects_are_isolated_from_each_other(db, corpus, engine, make_service):
    with db.session() as s:
        other = s.projects.create(Project(name="other")).id
    svc = make_service()
    svc.ingest_bytes(other, "NOTES.md", b"The `compute_backoff` function accepts `attempt` and `base_delay`.\n")
    svc.ingest_bytes(other, "x.py", b"def unrelated(a): ...\n")
    r = engine.verify_project(other)
    c = next(x for x in r.claims if "base_delay" in x.text and x.category == ClaimType.PARAMETER)
    assert c.status != V.SUPPORTED and "pkg.mod" not in json.dumps(r.to_dict())        # project 1's code is never consulted
    with db.session() as s:
        assert {x.project_id for x in s.claims.list_for_project(other)} == {other}


def test_empty_projects_and_code_only_projects(db, project_id, engine, make_service):
    empty = engine.verify_project(project_id)
    assert empty.counts["claims"] == 0 and empty.score.final is None and empty.run.trust_score is None
    assert any("no indexed Python entities" in w for w in empty.warnings) and "not measurable" in render_report(empty)
    with db.session() as s:
        assert s.runs.get_trust_breakdown(empty.run.id) is None
    make_service().ingest_bytes(project_id, "a.py", b"def f(x):\n    '''Doc.\n\n    Args:\n        x: value\n    '''\n\ndef g(y): ...\n")
    code_only = engine.verify_project(project_id)
    assert (code_only.counts["claims"], code_only.counts["supported"]) == (1, 1)             # the docstring of `f` matches its signature
    assert code_only.score.final == pytest.approx(100 * (0.5 * 1 + 0.25 * 0.5 + 0.25 * 1))   # accuracy 1/1, coverage 1/2, grounding 1/1
    gap = [f for f in code_only.findings if f.category == "coverage"]
    assert [(f.code_location["qualified_name"], f.severity) for f in gap] == [("a.g", S.MEDIUM)]


# ---------------------------------------------------------------- generated documentation
def test_generated_text_is_verified_as_a_generated_source(corpus, engine, db):
    text = "The `compute_backoff` function accepts `attempt` and `base_delay`.\n"
    r = engine.verify_text(corpus, text)
    assert r.documents[0].startswith("generated/verify-") and all(c.origin == ClaimOrigin.GENERATED.value for c in r.claims)
    assert not any("docstring of" in c.doc_label for c in r.claims)
    coverage = next(c for c in r.score.components if c.name == "coverage")
    assert coverage.weight == 0 and "unavailable" in coverage.detail                    # project-wide coverage is not part of a text check
    assert not any(f.category == "coverage" for f in r.findings)
    with db.session() as s:
        src = s.sources.get_by_name(corpus, r.documents[0])
    assert src.source_type.value == "markdown" and src.name == r.documents[0]
    with pytest.raises(DocumentSelectionError):
        engine.verify_text(corpus, "   ")


# -------------------------------------------------------------------- semantic wiring
def test_semantic_checks_are_skipped_when_not_requested_or_not_available(corpus, engine, db, settings):
    on = VerificationEngine(db, settings).verify_project(corpus, semantic=True)         # provider "none": nothing configured
    assert {c.status for c in on.claims if c.category == ClaimType.BEHAVIORAL} == {V.NOT_CHECKABLE}
    assert any("unavailable" in s.lower() for s in on.skipped)
    assert on.counts["contradicted"] == 3 and on.score.final is not None                   # Tier 1 is unaffected by the missing model


def test_semantic_claims_are_verified_when_a_retriever_and_model_are_supplied(corpus, db, settings):
    abstaining = FakeRetriever(make_search_result([], abstained=True))
    r = VerificationEngine(db, settings, retriever=abstaining, llm=ScriptedLLM("unused")).verify_project(corpus, semantic=True)
    behaviors = [c for c in r.claims if c.category == ClaimType.BEHAVIORAL]
    assert {c.status for c in behaviors} == {V.INSUFFICIENT_EVIDENCE} and r.counts["insufficient_evidence"] == 2     # no evidence: unverified, not false
    assert r.counts["not_checkable"] == 0 and r.semantic["checked"] == 2 and r.score.inputs["checked"] == 8
    assert all(f.severity == S.INFO for f in r.findings if f.category == "behavioral")


def test_statically_decided_claims_never_reach_the_llm(corpus, db, settings):
    llm = ScriptedLLM("{}")
    VerificationEngine(db, settings, retriever=FakeRetriever(make_search_result([], abstained=True)), llm=llm).verify_project(corpus, semantic=True)
    assert llm.calls == []                       # only retrieval abstained here; and static claims never call the model at all


def test_semantic_claim_limit_is_enforced(corpus, db, settings):
    from trustdoc.config import Settings
    cfg = Settings.from_env({"TRUSTDOC_DATA_DIR": str(settings.storage.data_dir), "TRUSTDOC_LOG_TO_FILE": "false", "TRUSTDOC_VERIFY_MAX_SEMANTIC_CLAIMS": "1"})
    r = VerificationEngine(db, cfg, retriever=FakeRetriever(make_search_result([], abstained=True)), llm=ScriptedLLM("x")).verify_project(corpus, semantic=True)
    assert r.counts["not_checkable"] == 1 and any("semantic claim limit (1)" in s for s in r.skipped)


# --------------------------------------------------------------------- atomicity
def test_a_failure_while_persisting_leaves_no_partial_results(corpus, engine, db, monkeypatch):
    from trustdoc.db.repositories import FindingRepository

    original, calls = FindingRepository.add, {"n": 0}

    def flaky(self, f):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk full")
        return original(self, f)

    monkeypatch.setattr(FindingRepository, "add", flaky)
    with pytest.raises(VerificationError, match="disk full"):
        engine.verify_project(corpus)
    counts = db.table_counts()
    assert counts["claims"] == 0 and counts["claim_results"] == 0 and counts["findings"] == 0 and counts["evidence_links"] == 0
    assert counts["trust_score_components"] == 0
    runs = engine.list_runs(corpus)
    assert [r.status for r in runs] == [RunStatus.FAILED] and "disk full" in runs[0].error_message
    monkeypatch.undo()
    assert engine.verify_project(corpus).run.status == RunStatus.COMPLETED


def test_a_failure_during_verification_is_recorded_and_earlier_runs_are_untouched(corpus, engine, db, monkeypatch):
    good = engine.verify_project(corpus)
    from trustdoc.verification.static_verifier import StaticVerifier

    monkeypatch.setattr(StaticVerifier, "verify", lambda self, claim: (_ for _ in ()).throw(ValueError("bug")))
    with pytest.raises(VerificationError, match="ValueError: bug"):
        engine.verify_project(corpus)
    runs = engine.list_runs(corpus)
    assert [r.status for r in runs] == [RunStatus.FAILED, RunStatus.COMPLETED]
    monkeypatch.undo()
    assert engine.load_report(good.run.id).to_dict() == good.to_dict()
    assert engine.load_report(runs[0].id).warnings[0].startswith("Run failed")


def test_malformed_model_output_never_corrupts_a_run(corpus, db, settings):
    from ask_support import make_evidence
    ev = [make_evidence(1, "x", source_name="pkg/mod.py", project_id=corpus)]
    r = VerificationEngine(db, settings, retriever=FakeRetriever(make_search_result(ev, project_id=corpus)), llm=ScriptedLLM("not json at all")).verify_project(corpus, semantic=True)
    assert r.run.status == RunStatus.COMPLETED and r.counts["insufficient_evidence"] == 2
    assert all("rejected" in c.explanation for c in r.claims if c.category == ClaimType.BEHAVIORAL)


def test_claim_identity_helper_matches_stored_claims(corpus, engine, db):
    engine.verify_project(corpus)
    with db.session() as s:
        assert all(claim_uid(c) for c in s.claims.list_for_project(corpus))
