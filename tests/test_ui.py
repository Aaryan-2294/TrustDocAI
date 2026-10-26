"""Unit and headless integration tests for the Streamlit UI (Part 7)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure non-UI environments cleanly skip this entire module with a clear explanation
pytest.importorskip(
    "streamlit",
    reason="Streamlit is required for UI tests (install with: pip install -e .[ui])",
)

import streamlit as st
from streamlit.testing.v1 import AppTest

from trustdoc.app import AppContext
from trustdoc.exceptions import CleanupError
from trustdoc.ingestion.storage import UploadStore
from trustdoc.models import (
    CodeEntity,
    EntityType,
    Finding,
    IssueType,
    Parameter,
    ParameterKind,
    Project,
    Severity,
    Source,
    SourceType,
    TrustScoreComponent,
    VerificationTier,
)
from trustdoc.ui.components import (
    HEURISTIC_DISCLAIMER,
    render_citations_list,
    render_claim_counts,
    render_evidence_drawer,
    render_findings_list,
    render_header,
    render_heuristic_disclaimer,
    render_status_badge,
    render_trust_score_card,
)
from trustdoc.ui.state import (
    create_project,
    delete_project,
    delete_source,
    get_project,
    get_project_stats,
    list_entities,
    list_projects,
    list_sources,
)
from trustdoc.verification.types import FindingReport, ScoreReport
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    AskStatus,
    GenerationStatus,
)


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


# ------------------------------------------------------------------- state tests
def test_ui_database_queries(db):
    """Test state query helpers for projects, sources, and entities."""
    # 1. Create project
    proj = create_project(db, "ui-test-project", "UI test description")
    assert proj.id is not None
    assert proj.name == "ui-test-project"

    # 2. List projects
    projs = list_projects(db)
    assert any(p.id == proj.id for p in projs)

    # 3. Get project
    fetched = get_project(db, proj.id)
    assert fetched is not None
    assert fetched.name == "ui-test-project"
    assert fetched.description == "UI test description"

    # 4. Add source and entity
    with db.session() as s:
        src = s.sources.add(
            Source(
                project_id=proj.id,
                source_type=SourceType.PYTHON,
                name="test_mod.py",
                content_hash="hash123",
                size_bytes=256,
            )
        )
        ent = s.entities.add(
            CodeEntity(
                project_id=proj.id,
                source_id=src.id,
                entity_type=EntityType.FUNCTION,
                name="calculate",
                qualified_name="test_mod.calculate",
                parameters=[Parameter(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
                start_line=1,
                end_line=5,
            )
        )

    # 5. List sources
    sources = list_sources(db, proj.id)
    assert len(sources) == 1
    assert sources[0].name == "test_mod.py"

    sources_md = list_sources(db, proj.id, source_type=SourceType.MARKDOWN)
    assert len(sources_md) == 0

    # 6. List entities
    entities = list_entities(db, proj.id)
    assert len(entities) == 1
    assert entities[0].qualified_name == "test_mod.calculate"

    # 7. Project stats
    stats = get_project_stats(db, proj.id)
    assert stats["sources"] == 1
    assert stats["entities"] == 1
    assert stats["chunks"] == 0

    # 8. Delete source
    assert delete_source(db, src.id) is True
    assert len(list_sources(db, proj.id)) == 0

    # 9. Delete project
    assert delete_project(db, proj.id) is True
    assert get_project(db, proj.id) is None


# ------------------------------------------------------------------- deletion cleanup tests
def test_delete_project_removes_records_uploads_and_indexes(tmp_path):
    """Deleting a project removes DB records, upload files, and index files."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.ingestion import IngestionService
    from trustdoc.retrieval.embedding import HashingEmbedder
    from trustdoc.retrieval.index_manager import IndexManager

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    store = UploadStore(settings.storage.uploads_dir)
    idx_mgr = IndexManager(ctx.db, settings, HashingEmbedder())
    ingest_svc = IngestionService(ctx.db, settings)

    # Create project and ingest a source
    proj = create_project(ctx.db, "clean-proj")
    res = ingest_svc.ingest_bytes(proj.id, "mod.py", b"def add(a: int, b: int) -> int: return a + b\n")

    # Build index
    idx_mgr.build(proj.id)

    # Verify upload file and index directory exist
    upload_file = store.resolve(res.source.stored_path)
    assert upload_file.is_file()
    proj_upload_dir = store.project_dir(proj.id)
    assert proj_upload_dir.is_dir()
    proj_index_dir = idx_mgr.index_dir(proj.id)
    assert proj_index_dir.is_dir()
    assert proj.id in idx_mgr._cache

    # Delete project
    success = delete_project(ctx, proj.id, store=store, index_manager=idx_mgr)
    assert success is True

    # Verify DB records deleted
    assert get_project(ctx.db, proj.id) is None
    with ctx.db.session() as s:
        assert len(s.sources.list_for_project(proj.id)) == 0
        assert len(s.chunks.list_for_project(proj.id)) == 0

    # Verify upload directory and index directory removed
    assert not proj_upload_dir.exists()
    assert not proj_index_dir.exists()
    assert proj.id not in idx_mgr._cache


def test_delete_source_removes_unshared_upload_and_invalidates_index(tmp_path):
    """Deleting an unshared source removes its upload file and invalidates index."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.ingestion import IngestionService
    from trustdoc.retrieval.embedding import HashingEmbedder
    from trustdoc.retrieval.index_manager import IndexManager

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    store = UploadStore(settings.storage.uploads_dir)
    idx_mgr = IndexManager(ctx.db, settings, HashingEmbedder())
    ingest_svc = IngestionService(ctx.db, settings)

    proj = create_project(ctx.db, "src-del-proj")
    res = ingest_svc.ingest_bytes(proj.id, "single.py", b"def single(): pass\n")
    idx_mgr.build(proj.id)

    upload_file = store.resolve(res.source.stored_path)
    assert upload_file.is_file()
    assert idx_mgr.index_dir(proj.id).is_dir()

    # Delete source
    assert delete_source(ctx, res.source.id, store=store, index_manager=idx_mgr) is True

    # Verify source DB record deleted
    with ctx.db.session() as s:
        assert s.sources.get(res.source.id) is None

    # Stored upload file must be removed because no other source references it
    assert not upload_file.exists()

    # Index must be invalidated (removed from disk and cache)
    assert not idx_mgr.index_dir(proj.id).exists()
    assert proj.id not in idx_mgr._cache
    assert idx_mgr.is_stale(proj.id)


def test_delete_source_shared_stored_path(tmp_path):
    """Deleting one source with a shared stored path preserves file until all references are gone."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.ingestion import IngestionService

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    store = UploadStore(settings.storage.uploads_dir)
    ingest_svc = IngestionService(ctx.db, settings)

    proj = create_project(ctx.db, "shared-path-proj")
    shared_bytes = b"# Identical content shared across two source names\ndef common(): return 42\n"

    # Ingest source 1 and source 2 with identical content
    res1 = ingest_svc.ingest_bytes(proj.id, "path_a.py", shared_bytes)
    res2 = ingest_svc.ingest_bytes(proj.id, "path_b.py", shared_bytes)

    assert res1.source.stored_path == res2.source.stored_path
    upload_file = store.resolve(res1.source.stored_path)
    assert upload_file.is_file()

    # 1. Delete source 1: source 2 still references stored_path, so file MUST NOT be deleted
    assert delete_source(ctx, res1.source.id, store=store) is True
    assert upload_file.is_file()  # Still exists!

    # 2. Delete source 2: now no sources reference stored_path, so file MUST be deleted
    assert delete_source(ctx, res2.source.id, store=store) is True
    assert not upload_file.exists()  # Cleaned up!


def test_delete_project_upload_failure_still_cleans_index(tmp_path):
    """If upload cleanup fails during project deletion, index cleanup is still attempted."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.exceptions import CleanupError
    from trustdoc.ingestion import IngestionService
    from trustdoc.retrieval.embedding import HashingEmbedder
    from trustdoc.retrieval.index_manager import IndexManager

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    store = UploadStore(settings.storage.uploads_dir)
    idx_mgr = IndexManager(ctx.db, settings, HashingEmbedder())
    ingest_svc = IngestionService(ctx.db, settings)

    proj = create_project(ctx.db, "fail-up-proj")
    ingest_svc.ingest_bytes(proj.id, "m.py", b"def f(): pass\n")
    idx_mgr.build(proj.id)

    assert idx_mgr.index_dir(proj.id).is_dir()
    assert proj.id in idx_mgr._cache

    mock_store = MagicMock()
    mock_store.project_dir.return_value = store.project_dir(proj.id)
    mock_store.delete_project.side_effect = CleanupError("Permission denied on uploads")

    with pytest.raises(CleanupError) as exc_info:
        delete_project(ctx, proj.id, store=mock_store, index_manager=idx_mgr)

    # 1. Error message identifies the failed upload task and remaining safe retry path
    assert "Permission denied on uploads" in str(exc_info.value)
    assert str(store.project_dir(proj.id)) in str(exc_info.value)
    assert "retried safely" in str(exc_info.value)

    # 2. Database project record was deleted
    assert get_project(ctx.db, proj.id) is None

    # 3. Index invalidation was STILL attempted and succeeded!
    assert not idx_mgr.index_dir(proj.id).exists()
    assert proj.id not in idx_mgr._cache


def test_delete_source_upload_failure_still_invalidates_index(tmp_path):
    """If stored upload cleanup fails during source deletion, index invalidation is still attempted."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.exceptions import CleanupError
    from trustdoc.ingestion import IngestionService
    from trustdoc.retrieval.embedding import HashingEmbedder
    from trustdoc.retrieval.index_manager import IndexManager

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    store = UploadStore(settings.storage.uploads_dir)
    idx_mgr = IndexManager(ctx.db, settings, HashingEmbedder())
    ingest_svc = IngestionService(ctx.db, settings)

    proj = create_project(ctx.db, "fail-src-up-proj")
    res = ingest_svc.ingest_bytes(proj.id, "single.py", b"def single(): pass\n")
    idx_mgr.build(proj.id)

    assert idx_mgr.index_dir(proj.id).is_dir()
    assert proj.id in idx_mgr._cache

    mock_store = MagicMock()
    mock_store.resolve.return_value = store.resolve(res.source.stored_path)
    mock_store.delete.side_effect = CleanupError("Permission denied on file")

    with pytest.raises(CleanupError) as exc_info:
        delete_source(ctx, res.source.id, store=mock_store, index_manager=idx_mgr)

    # 1. Error message identifies the failed stored file task and remaining safe retry path
    assert "Permission denied on file" in str(exc_info.value)
    assert str(store.resolve(res.source.stored_path)) in str(exc_info.value)
    assert "retried safely" in str(exc_info.value)

    # 2. Source record in DB was deleted
    with ctx.db.session() as s:
        assert s.sources.get(res.source.id) is None

    # 3. Index invalidation was STILL attempted and succeeded!
    assert not idx_mgr.index_dir(proj.id).exists()
    assert proj.id not in idx_mgr._cache


def test_delete_project_both_fail_aggregates_errors(tmp_path):
    """When both upload and index cleanups fail, all errors are aggregated."""
    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.exceptions import CleanupError

    settings = Settings.from_env({
        "TRUSTDOC_DATA_DIR": str(tmp_path / "data"),
        "TRUSTDOC_LOG_TO_FILE": "false",
    })
    ctx = bootstrap(settings)
    proj = create_project(ctx.db, "dual-fail-proj")

    mock_store = MagicMock()
    mock_store.project_dir.return_value = Path("uploads/project_1")
    mock_store.delete_project.side_effect = CleanupError("Store failure")

    mock_idx_mgr = MagicMock()
    mock_idx_mgr.index_dir.return_value = Path("indexes/project_1")
    mock_idx_mgr.delete.side_effect = CleanupError("Index failure")

    with pytest.raises(CleanupError) as exc_info:
        delete_project(ctx, proj.id, store=mock_store, index_manager=mock_idx_mgr)

    err = str(exc_info.value)
    assert "Store failure" in err
    assert "Index failure" in err
    assert str(Path("uploads/project_1")) in err
    assert str(Path("indexes/project_1")) in err


# ------------------------------------------------------------------- component tests
def test_heuristic_disclaimer_content():
    """Verify that the heuristic disclaimer explicitly mentions non-calibrated truth."""
    assert "automated heuristic metric" in HEURISTIC_DISCLAIMER
    assert "not a calibrated probability of truth" in HEURISTIC_DISCLAIMER


@patch("streamlit.caption")
def test_render_heuristic_disclaimer(mock_caption):
    render_heuristic_disclaimer()
    mock_caption.assert_called_once()
    assert "Notice:" in mock_caption.call_args[0][0]
    assert HEURISTIC_DISCLAIMER in mock_caption.call_args[0][0]


@patch("streamlit.markdown")
def test_render_header(mock_markdown):
    render_header()
    mock_markdown.assert_called_once()
    assert "TrustDoc AI" in mock_markdown.call_args[0][0]


@patch("streamlit.success")
@patch("streamlit.warning")
@patch("streamlit.error")
@patch("streamlit.info")
def test_render_status_badge(mock_info, mock_error, mock_warning, mock_success):
    # Completed + verified
    render_status_badge(GenerationStatus.COMPLETED, is_verified=True)
    mock_success.assert_called()

    # Completed + unverified
    render_status_badge(GenerationStatus.COMPLETED, is_verified=False)
    mock_info.assert_called()

    # Verification skipped
    render_status_badge(GenerationStatus.VERIFICATION_SKIPPED)
    mock_warning.assert_called()

    # Verification error
    render_status_badge(GenerationStatus.VERIFICATION_ERROR, error="Verification crashed")
    mock_error.assert_called()

    # Insufficient evidence
    render_status_badge(AskStatus.INSUFFICIENT_EVIDENCE)
    mock_info.assert_called()

    # Answered
    render_status_badge(AskStatus.ANSWERED)
    mock_success.assert_called()

    # Retrieval error
    render_status_badge(AskStatus.RETRIEVAL_ERROR, error="FAISS index missing")
    mock_error.assert_called()


@patch("streamlit.columns")
@patch("streamlit.markdown")
@patch("streamlit.info")
@patch("streamlit.caption")
def test_render_trust_score_card_none(mock_caption, mock_info, mock_md, mock_cols):
    render_trust_score_card(None)
    mock_info.assert_called()
    assert "Not Measurable" in mock_info.call_args[0][0]
    mock_caption.assert_called()


@patch("streamlit.columns")
@patch("streamlit.markdown")
@patch("streamlit.caption")
def test_render_trust_score_card_score_report(mock_caption, mock_md, mock_cols):
    col1 = MagicMock()
    col2 = MagicMock()
    mock_cols.return_value = [col1, col2]

    report = ScoreReport(
        final=88.5,
        base=95.0,
        components=[
            TrustScoreComponent(name="accuracy", raw_score=0.9, weight=0.5, contribution=45.0),
            TrustScoreComponent(name="coverage", raw_score=0.8, weight=0.25, contribution=20.0),
            TrustScoreComponent(name="grounding", raw_score=0.95, weight=0.25, contribution=23.75),
        ],
        penalties=[{"label": "High Contradiction", "penalty": 5.0, "detail": "1 high finding"}],
        inputs={},
        notes=[],
    )

    render_trust_score_card(report)
    mock_cols.assert_called_once_with([1, 2])
    col1.markdown.assert_called_once()
    assert "88.5" in col1.markdown.call_args[0][0]
    mock_caption.assert_called()


@patch("streamlit.columns")
def test_render_claim_counts(mock_cols):
    c1, c2, c3, c4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_cols.return_value = [c1, c2, c3, c4]

    counts = {"supported": 5, "contradicted": 2, "insufficient_evidence": 1, "not_checked": 0}
    render_claim_counts(counts)

    c1.metric.assert_called_once_with(label="✅ Supported", value=5)
    c2.metric.assert_called_once_with(label="❌ Contradicted", value=2)
    c3.metric.assert_called_once_with(label="❓ Insufficient Evidence", value=1)
    c4.metric.assert_called_once_with(label="⚪ Not Checked", value=0)


@patch("streamlit.success")
def test_render_findings_list_empty(mock_success):
    render_findings_list([])
    mock_success.assert_called_once()
    assert "No defect findings" in mock_success.call_args[0][0]


@patch("streamlit.expander")
@patch("streamlit.markdown")
@patch("streamlit.info")
def test_render_findings_list_with_findings(mock_info, mock_md, mock_expander):
    exp_ctx = MagicMock()
    mock_expander.return_value.__enter__.return_value = exp_ctx

    finding = FindingReport(
        uid="find1",
        claim_uid="claim1",
        category="parameter",
        status="contradicted",
        severity=Severity.HIGH,
        issue_type="unknown_parameter",
        message="Parameter 'foo' not found in function signature",
        suggestion="Remove parameter 'foo' or update function definition.",
        tier=VerificationTier.DETERMINISTIC,
        doc_location={"source": "README.md", "start_line": 12},
        code_location={"source": "api.py", "start_line": 40},
    )

    render_findings_list([finding])
    mock_expander.assert_called()


@patch("streamlit.markdown")
def test_render_citations_list(mock_md):
    cits = [
        AskCitation(
            id="E1",
            label="api.py:10-25",
            source_name="api.py",
            source_type="python",
            source_id=1,
            chunk_id=1,
            start_line=10,
            end_line=25,
            qualified_name="my_func",
        )
    ]
    render_citations_list(cits)
    mock_md.assert_called()


# ------------------------------------------------------------------- AppTest headless tests
def test_streamlit_app_loads_headlessly(tmp_path, monkeypatch):
    """Run trustdoc/ui/app.py headlessly using Streamlit's official AppTest framework."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TRUSTDOC_LOG_TO_FILE", "false")

    app_path = Path(__file__).resolve().parent.parent / "trustdoc" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    # Verify no unhandled exception occurred during render
    assert not at.exception
    # Check that header title is present in markdown widgets
    assert any("TrustDoc AI" in str(elem.value) for elem in at.markdown)


def test_streamlit_app_with_active_project(tmp_path, monkeypatch):
    """Verify AppTest loads an existing project with all 4 tabs rendered."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TRUSTDOC_LOG_TO_FILE", "false")

    from trustdoc.app import bootstrap
    from trustdoc.config import Settings
    from trustdoc.models import Project

    settings = Settings.from_env()
    ctx = bootstrap(settings)
    with ctx.db.session() as s:
        s.projects.create(Project(name="demo-apptest", description="Demo app test project"))

    app_path = Path(__file__).resolve().parent.parent / "trustdoc" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    assert not at.exception
    assert any("demo-apptest" in str(elem.value) for elem in at.markdown)


def test_cli_ui_parser():
    """Verify CLI parser accepts 'ui' subcommand with optional --port and --host."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    ui_p = sub.add_parser("ui")
    ui_p.add_argument("--port", type=int, default=None)
    ui_p.add_argument("--host", default=None)

    args = parser.parse_args(["ui", "--port", "8502", "--host", "127.0.0.1"])
    assert args.command == "ui"
    assert args.port == 8502
    assert args.host == "127.0.0.1"
