"""Session state and application wiring for the Streamlit UI (Part 7)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import streamlit as st

from trustdoc.app import AppContext, bootstrap
from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.ingestion import IngestionService
from trustdoc.models import (
    CodeEntity,
    EntityType,
    Project,
    Source,
    SourceType,
)
from trustdoc.retrieval.factory import create_retriever
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


def delete_project(db: Database, project_id: int) -> bool:
    """Delete a project and all associated sources, chunks, and runs."""
    with db.session() as s:
        return s.projects.delete(project_id)


def list_sources(
    db: Database, project_id: int, source_type: SourceType | None = None
) -> list[Source]:
    """List all sources for a project."""
    with db.session() as s:
        return s.sources.list_for_project(project_id, source_type=source_type)


def delete_source(db: Database, source_id: int) -> bool:
    """Delete a single source from the database."""
    with db.session() as s:
        return s.sources.delete(source_id)


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
