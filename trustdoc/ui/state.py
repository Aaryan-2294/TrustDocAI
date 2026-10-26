"""Session state and application wiring for the Streamlit UI (Part 7)."""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from trustdoc.app import AppContext, bootstrap
from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import CleanupError
from trustdoc.ingestion import IngestionService
from trustdoc.ingestion.storage import UploadStore
from trustdoc.models import (
    CodeEntity,
    EntityType,
    Project,
    Source,
    SourceType,
)
from trustdoc.retrieval.factory import create_retriever
from trustdoc.retrieval.index_manager import IndexManager
from trustdoc.retrieval.retriever import Retriever
from trustdoc.verification import VerificationEngine
from trustdoc.workflows import (
    AskService,
    GenerationService,
    create_ask_service,
    create_generation_service,
)


@dataclass
class AppServices:
    """Singleton collection of initialized TrustDoc AI services."""

    ctx: AppContext
    ingestion: IngestionService
    retriever: Retriever
    ask: AskService
    generation: GenerationService
    verification: VerificationEngine


@st.cache_resource(show_spinner=False)
def get_app_context(db_path: str | None = None) -> AppContext:
    """Bootstrap application settings and database (cached across Streamlit reruns)."""
    settings = Settings.from_env()
    if db_path:
        storage_cfg = dataclasses.replace(settings.storage, db_path=db_path)
        settings = dataclasses.replace(settings, storage=storage_cfg)
    return bootstrap(settings)


@st.cache_resource(show_spinner=False)
def get_services(_ctx: AppContext) -> AppServices:
    """Instantiate and cache core workflow services for the current AppContext."""
    retriever = create_retriever(_ctx.db, _ctx.settings)
    ingestion = IngestionService(_ctx.db, _ctx.settings)
    verification = VerificationEngine(_ctx.db, _ctx.settings, retriever=retriever)
    ask = create_ask_service(_ctx.db, _ctx.settings, retriever=retriever)
    generation = create_generation_service(
        _ctx.db, _ctx.settings, retriever=retriever, verification_engine=verification
    )
    return AppServices(
        ctx=_ctx,
        ingestion=ingestion,
        retriever=retriever,
        ask=ask,
        generation=generation,
        verification=verification,
    )


# ------------------------------------------------------------------- database queries
def list_projects(db: Database) -> list[Project]:
    """Retrieve all projects ordered by name."""
    with db.session() as s:
        return s.projects.list()


def create_project(db: Database, name: str, description: str = "") -> Project:
    """Create a new project."""
    with db.session() as s:
        p = Project(name=name.strip(), description=description.strip())
        return s.projects.create(p)


def get_project(db: Database, project_id: int) -> Project | None:
    """Get a project by ID."""
    with db.session() as s:
        return s.projects.get(project_id)


def _safe_resolve_upload_path(store: UploadStore, stored_path: str) -> str:
    """Best-effort resolution of stored upload path for error reporting.

    If path resolution fails (e.g. invalid or escaping path), returns the original
    stored-path value and the resolution error rather than raising.
    """
    try:
        return str(store.resolve(stored_path))
    except Exception as exc:  # noqa: BLE001
        return f"{stored_path} (resolution failed: {exc})"


def _safe_resolve_project_upload_dir(store: UploadStore, project_id: int) -> str:
    """Best-effort resolution of project upload directory for error reporting.

    If directory resolution fails, returns the project identifier and the resolution
    error rather than raising.
    """
    try:
        return str(store.project_dir(project_id))
    except Exception as exc:  # noqa: BLE001
        return f"project_{project_id} (resolution failed: {exc})"


def _safe_resolve_index_dir(
    index_manager: IndexManager | None, settings: Settings, project_id: int
) -> str:
    """Best-effort resolution of project index directory for error reporting."""
    try:
        if index_manager is not None:
            return str(index_manager.index_dir(project_id))
        return str((Path(settings.storage.indexes_dir) / f"project_{int(project_id)}").resolve())
    except Exception as exc:  # noqa: BLE001
        return f"project_{project_id} (resolution failed: {exc})"


def delete_project(
    target: AppContext | Database,
    project_id: int,
    settings: Settings | None = None,
    *,
    store: UploadStore | None = None,
    index_manager: IndexManager | None = None,
) -> bool:
    """Delete a project and clean up its database records, uploaded files, and indexes.

    Transaction ordering guarantees that files on disk are removed only after database
    records have been successfully deleted and committed. Cleanup operations for uploaded
    files and index artifacts are executed independently so failure in one does not block
    the other. Any cleanup errors are aggregated with actionable path details.
    """
    if isinstance(target, AppContext):
        db = target.db
        settings = settings or target.settings
    else:
        db = target
        settings = settings or Settings.from_env()

    # 1. Database deletion (cascades to sources, entities, chunks, claims, runs, findings)
    with db.session() as s:
        proj = s.projects.get(project_id)
        if proj is None:
            return False
        s.projects.delete(project_id)

    cleanup_errors: list[str] = []

    # 2. Filesystem cleanup of uploaded files
    upload_store = store
    try:
        if upload_store is None:
            upload_store = UploadStore(settings.storage.uploads_dir)
        upload_store.delete_project(project_id)
    except Exception as exc:  # noqa: BLE001
        if upload_store is not None:
            proj_upload_dir = _safe_resolve_project_upload_dir(upload_store, project_id)
        else:
            proj_upload_dir = f"project_{project_id}"
        cleanup_errors.append(f"upload directory ({proj_upload_dir}): {exc}")

    # 3. Cleanup of persisted and in-memory indexes (attempted even if upload cleanup failed)
    try:
        if index_manager is not None:
            index_manager.delete(project_id)
        else:
            idx_dir = (Path(settings.storage.indexes_dir) / f"project_{int(project_id)}").resolve()
            if idx_dir.is_dir() and idx_dir.is_relative_to(Path(settings.storage.indexes_dir).resolve()):
                shutil.rmtree(idx_dir)
    except Exception as exc:  # noqa: BLE001
        idx_path = _safe_resolve_index_dir(index_manager, settings, project_id)
        cleanup_errors.append(f"index directory ({idx_path}): {exc}")

    if cleanup_errors:
        err_msg = (
            f"Project {project_id} database records were deleted, but filesystem cleanup failed for: "
            + "; ".join(cleanup_errors)
            + ". These remaining paths can be cleaned up or retried safely."
        )
        raise CleanupError(err_msg)

    return True


def list_sources(
    db: Database, project_id: int, source_type: SourceType | None = None
) -> list[Source]:
    """List all sources for a project."""
    with db.session() as s:
        return s.sources.list_for_project(project_id, source_type=source_type)


def delete_source(
    target: AppContext | Database,
    source_id: int,
    settings: Settings | None = None,
    *,
    store: UploadStore | None = None,
    index_manager: IndexManager | None = None,
) -> bool:
    """Delete a single source, remove its stored upload if unreferenced, and invalidate index.

    Cleanup operations for stored upload files and project indexes are executed independently
    so failure in file cleanup does not prevent index invalidation. Any cleanup errors are
    aggregated with actionable path details.
    """
    if isinstance(target, AppContext):
        db = target.db
        settings = settings or target.settings
    else:
        db = target
        settings = settings or Settings.from_env()

    stored_path: str | None = None
    project_id: int | None = None
    still_used = False

    # 1. Database deletion within transaction
    with db.session() as s:
        src = s.sources.get(source_id)
        if src is None:
            return False
        project_id = src.project_id
        stored_path = src.stored_path
        s.sources.delete(source_id)

        if stored_path:
            remaining = s.conn.execute(
                "SELECT COUNT(*) FROM sources WHERE stored_path = ?", (stored_path,)
            ).fetchone()[0]
            still_used = remaining > 0

    cleanup_errors: list[str] = []

    # 2. Cleanup stored file if no remaining source references it
    if stored_path and not still_used:
        upload_store = store
        try:
            if upload_store is None:
                upload_store = UploadStore(settings.storage.uploads_dir)
            upload_store.delete(stored_path)
        except Exception as exc:  # noqa: BLE001
            if upload_store is not None:
                resolved_file = _safe_resolve_upload_path(upload_store, stored_path)
            else:
                resolved_file = str(stored_path)
            cleanup_errors.append(f"stored file ({resolved_file}): {exc}")

    # 3. Invalidate or remove the project's index so the next retrieval rebuilds it
    # (attempted even if stored-file cleanup failed)
    if project_id is not None:
        try:
            if index_manager is not None:
                index_manager.delete(project_id)
            else:
                idx_dir = (Path(settings.storage.indexes_dir) / f"project_{int(project_id)}").resolve()
                if idx_dir.is_dir() and idx_dir.is_relative_to(Path(settings.storage.indexes_dir).resolve()):
                    shutil.rmtree(idx_dir)
        except Exception as exc:  # noqa: BLE001
            idx_path = _safe_resolve_index_dir(index_manager, settings, project_id)
            cleanup_errors.append(f"index directory ({idx_path}): {exc}")

    if cleanup_errors:
        err_msg = (
            f"Source {source_id} database records were deleted, but cleanup failed for: "
            + "; ".join(cleanup_errors)
            + ". These remaining paths can be cleaned up or retried safely."
        )
        raise CleanupError(err_msg)

    return True


def list_entities(
    db: Database, project_id: int, entity_type: EntityType | None = None
) -> list[CodeEntity]:
    """List code entities (functions, classes, methods) for a project."""
    with db.session() as s:
        return s.entities.list_for_project(project_id, entity_type=entity_type)


def get_project_stats(db: Database, project_id: int) -> dict[str, int]:
    """Calculate aggregate counts for sources, code entities, and chunks."""
    with db.session() as s:
        sources = s.sources.list_for_project(project_id)
        entities = s.entities.list_for_project(project_id)
        chunk_count = s.chunks.count_for_project(project_id)
        return {
            "sources": len(sources),
            "entities": len(entities),
            "chunks": chunk_count,
        }
