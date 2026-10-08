import pytest

from trustdoc.exceptions import DuplicateError, NotFoundError
from trustdoc.models import (
    Chunk, ChunkType, Claim, ClaimOrigin, ClaimResult, ClaimType, ClaimVerdict, CodeEntity,
    EntityType, EvidenceLink, EvidenceStance, Finding, IssueType, Parameter, ParameterKind,
    Project, RunStatus, Severity, Source, SourceType, TrustScoreBreakdown, TrustScoreComponent,
    VerificationRun, VerificationTier,
)


# ---------------------------------------------------------------- projects
def test_project_crud_and_duplicate(db):
    with db.session() as s:
        p = s.projects.create(Project(name="alpha", description="first"))
        assert p.id is not None
        with pytest.raises(DuplicateError):
            s.projects.create(Project(name="alpha"))
    with db.session() as s:
        p = s.projects.require(p.id)
        p.description = "changed"
        s.projects.update(p)
        assert s.projects.get_by_name("alpha").description == "changed"
        assert [x.name for x in s.projects.list()] == ["alpha"]
        with pytest.raises(NotFoundError):
            s.projects.require(12345)


def test_multiple_projects_are_isolated(db):
    with db.session() as s:
        a = s.projects.create(Project(name="a"))
        b = s.projects.create(Project(name="b"))
        s.sources.add(Source(a.id, SourceType.PYTHON, "same.py", "h1"))
        s.sources.add(Source(b.id, SourceType.PYTHON, "same.py", "h2"))  # same name, other project
        with pytest.raises(DuplicateError):
            s.sources.add(Source(a.id, SourceType.PYTHON, "same.py", "h3"))
    with db.session() as s:
        assert len(s.sources.list_for_project(a.id)) == 1
        assert s.sources.get_by_name(b.id, "same.py").content_hash == "h2"


# ------------------------------------------------------------------ sources
def test_multiple_source_types_per_project(db, project_id):
    with db.session() as s:
        s.sources.add(Source(project_id, SourceType.PYTHON, "a.py", "1"))
        s.sources.add(Source(project_id, SourceType.MARKDOWN, "README.md", "2"))
        s.sources.add(Source(project_id, SourceType.PDF, "ref.pdf", "3", metadata={"page_count": 12}))
    with db.session() as s:
        assert len(s.sources.list_for_project(project_id)) == 3
        pdfs = s.sources.list_for_project(project_id, SourceType.PDF)
        assert pdfs[0].metadata == {"page_count": 12}


# ---------------------------------------------------- entities / chunks / claims
def test_entity_roundtrip_with_parameters_and_parent(db, project_id, py_source_id):
    with db.session() as s:
        cls = s.entities.add(CodeEntity(project_id, py_source_id, EntityType.CLASS, "Repo",
                                        "pkg.a.Repo", start_line=1, end_line=30))
        method = s.entities.add(CodeEntity(
            project_id, py_source_id, EntityType.METHOD, "save", "pkg.a.Repo.save",
            parent_id=cls.id, return_annotation="bool", docstring="Save it.",
            decorators=["staticmethod"], is_async=True, is_public=True, start_line=5,
            parameters=[Parameter("item", ParameterKind.POSITIONAL_OR_KEYWORD, "str"),
                        Parameter("force", ParameterKind.KEYWORD_ONLY, "bool", "False")]))
    with db.session() as s:
        got = s.entities.find_by_qualified_name(project_id, "pkg.a.Repo.save")[0]
        assert got.parent_id == cls.id and got.is_async and got.has_docstring
        assert got.parameters[1].default == "False" and got.parameters[1].kind == ParameterKind.KEYWORD_ONLY
        assert [e.name for e in s.entities.list_for_source(py_source_id)] == ["Repo", "save"]
        assert len(s.entities.list_for_project(project_id, EntityType.METHOD)) == 1


def test_chunks_with_location_metadata_and_get_many_order(db, project_id, py_source_id):
    with db.session() as s:
        ent = s.entities.add(CodeEntity(project_id, py_source_id, EntityType.FUNCTION, "f", "pkg.a.f"))
        c0 = s.chunks.add(Chunk(project_id, py_source_id, 0, ChunkType.CODE, "def f(): ...",
                                start_line=1, end_line=2, entity_id=ent.id))
        c1 = s.chunks.add(Chunk(project_id, py_source_id, 1, ChunkType.CODE, "x = 1", start_line=4))
        with pytest.raises(DuplicateError):
            s.chunks.add(Chunk(project_id, py_source_id, 1, ChunkType.CODE, "dup"))
    with db.session() as s:
        got = s.chunks.get_many([c1.id, 9999, c0.id])
        assert [c.id for c in got] == [c1.id, c0.id]
        assert got[1].entity_id == ent.id and got[1].end_line == 2
        assert s.chunks.count_for_project(project_id) == 2
        assert s.chunks.delete_for_source(py_source_id) == 2


def test_claim_roundtrip(db, project_id):
    with db.session() as s:
        md = s.sources.add(Source(project_id, SourceType.MARKDOWN, "README.md", "h"))
        s.claims.add(Claim(project_id, md.id, ClaimType.PARAMETER, "save(item, force)",
                           subject_name="save", extracted={"params": ["item", "force"]},
                           start_line=12))
        s.claims.add(Claim(project_id, md.id, ClaimType.BEHAVIORAL, "Retries 3 times",
                           origin=ClaimOrigin.GENERATED))
    with db.session() as s:
        assert len(s.claims.list_for_project(project_id)) == 2
        c = s.claims.list_for_project(project_id, ClaimType.PARAMETER)[0]
        assert c.extracted == {"params": ["item", "force"]} and c.origin == ClaimOrigin.EXISTING_DOC


# ------------------------------------------------------- runs & verification
def test_full_verification_run_roundtrip(db, project_id, py_source_id):
    with db.session() as s:
        md = s.sources.add(Source(project_id, SourceType.MARKDOWN, "README.md", "h"))
        ent = s.entities.add(CodeEntity(project_id, py_source_id, EntityType.FUNCTION, "f", "a.f"))
        chunk = s.chunks.add(Chunk(project_id, py_source_id, 0, ChunkType.CODE, "def f(): ..."))
        claim = s.claims.add(Claim(project_id, md.id, ClaimType.BEHAVIORAL, "f retries"))
        run = s.runs.create(VerificationRun(project_id, config_snapshot={"rrf_k": 60}))
        s.runs.mark_running(run.id)
        s.findings.add(Finding(run.id, project_id, VerificationTier.DETERMINISTIC,
                               IssueType.PARAMETER_NAME_MISMATCH, Severity.HIGH,
                               "Doc says 'x' but code has 'y'", entity_id=ent.id,
                               details={"documented": "x", "actual": "y"}))
        s.findings.add(Finding(run.id, project_id, VerificationTier.SEMANTIC,
                               IssueType.UNSUPPORTED_CLAIM, Severity.LOW, "No evidence found",
                               claim_id=claim.id))
        s.findings.add_claim_result(ClaimResult(
            run.id, claim.id, VerificationTier.SEMANTIC, ClaimVerdict.INSUFFICIENT_EVIDENCE, 0.4,
            "Nothing relevant retrieved",
            evidence=[EvidenceLink(chunk.id, EvidenceStance.RELATED, 0.12, 1)]))
        s.runs.save_trust_breakdown(run.id, TrustScoreBreakdown(
            [TrustScoreComponent("accuracy", 0.8, 0.5, 40.0, "8 of 10 claims"),
             TrustScoreComponent("severity_penalty", 0.2, 0.2, -4.0)], 36.0))
        s.runs.complete(run.id, summary={"claims": 10})

    with db.session() as s:
        r = s.runs.require(run.id)
        assert r.status == RunStatus.COMPLETED and r.trust_score == 36.0
        assert r.started_at and r.finished_at and r.summary == {"claims": 10}
        assert r.config_snapshot == {"rrf_k": 60}
        b = s.runs.get_trust_breakdown(run.id)
        assert [c.name for c in b.components] == ["accuracy", "severity_penalty"]
        assert b.components[1].contribution == -4.0
        assert len(s.findings.list_for_run(run.id)) == 2
        assert len(s.findings.list_for_run(run.id, issue_type=IssueType.UNSUPPORTED_CLAIM)) == 1
        assert s.findings.count_by_severity(run.id) == {Severity.HIGH: 1, Severity.LOW: 1}
        res = s.findings.list_claim_results(run.id)[0]
        assert res.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert res.evidence[0].chunk_id == chunk.id and res.evidence[0].relevance_score == 0.12
        assert [x.id for x in s.runs.list_for_project(project_id)] == [run.id]


def test_failed_run_and_missing_breakdown(db, project_id):
    with db.session() as s:
        run = s.runs.create(VerificationRun(project_id))
        s.runs.fail(run.id, "model unavailable")
        assert s.runs.get_trust_breakdown(run.id) is None
    with db.session() as s:
        r = s.runs.require(run.id)
        assert r.status == RunStatus.FAILED and r.error_message == "model unavailable"


# ----------------------------------------------------------------- cascades
def test_deleting_source_cascades_but_keeps_findings_history(db, project_id, py_source_id):
    with db.session() as s:
        ent = s.entities.add(CodeEntity(project_id, py_source_id, EntityType.FUNCTION, "f", "a.f"))
        s.chunks.add(Chunk(project_id, py_source_id, 0, ChunkType.CODE, "x", entity_id=ent.id))
        run = s.runs.create(VerificationRun(project_id))
        s.findings.add(Finding(run.id, project_id, VerificationTier.DETERMINISTIC,
                               IssueType.MISSING_DOCUMENTATION, Severity.LOW, "undocumented",
                               source_id=py_source_id, entity_id=ent.id))
    with db.session() as s:
        s.sources.delete(py_source_id)
    with db.session() as s:
        counts = s.conn.execute("SELECT (SELECT COUNT(*) FROM chunks), "
                                "(SELECT COUNT(*) FROM code_entities)").fetchone()
        assert tuple(counts) == (0, 0)
        f = s.findings.list_for_run(run.id)[0]          # finding survives, links nulled
        assert f.source_id is None and f.entity_id is None


def test_deleting_project_removes_everything(db, project_id, py_source_id):
    with db.session() as s:
        s.entities.add(CodeEntity(project_id, py_source_id, EntityType.FUNCTION, "f", "a.f"))
        s.chunks.add(Chunk(project_id, py_source_id, 0, ChunkType.CODE, "x"))
        s.runs.create(VerificationRun(project_id))
    with db.session() as s:
        assert s.projects.delete(project_id)
    assert sum(db.table_counts().values()) == 0
