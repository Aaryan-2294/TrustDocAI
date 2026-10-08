"""Ingestion service: the single entry point for adding sources to a project.

Flow for one file::

    sanitize name -> hash -> (unchanged? stop) -> extract (pure, may fail)
        -> store original file -> [transaction: replace old source,
           register source, persist entities, persist chunks] -> commit

Extraction happens before anything is written, and the database work happens in
one transaction, so a failing file leaves no partial state. Re-ingesting a name
with new content replaces the old source (its entities, chunks and claims
cascade away; the new source gets a new id).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from trustdoc.config import Settings
from trustdoc.db import Database, Session
from trustdoc.exceptions import (
    FileTooLargeError,
    IngestionError,
    TrustDocError,
)
from trustdoc.ingestion.extraction import Extraction, extract_source
from trustdoc.ingestion.storage import (
    SUPPORTED_EXTENSIONS,
    UploadStore,
    detect_source_type,
    sanitize_source_name,
)
from trustdoc.logging_setup import get_logger
from trustdoc.models import Source, SourceType

log = get_logger("ingestion")

DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
IGNORED_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "build", "dist", "site-packages", ".eggs",
})


class IngestionStatus(str, Enum):
    INGESTED = "ingested"    # new source
    UPDATED = "updated"      # same name, new content: old source replaced
    UNCHANGED = "unchanged"  # same name and content: nothing reprocessed


@dataclass
class IngestionResult:
    source: Source
    status: IngestionStatus
    entity_count: int = 0
    chunk_count: int = 0
    warnings: list[str] = field(default_factory=list)
    replaced_source_id: int | None = None


@dataclass
class IngestionFailure:
    name: str
    error: str


@dataclass
class BatchIngestionResult:
    results: list[IngestionResult] = field(default_factory=list)
    failures: list[IngestionFailure] = field(default_factory=list)
    skipped_unsupported: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures


class IngestionService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._db = db
        self._cfg = settings.chunking
        self._store = UploadStore(settings.storage.uploads_dir)
        self._max_bytes = max_file_bytes

    @property
    def store(self) -> UploadStore:
        return self._store

    # ------------------------------------------------------------ public API
    def ingest_bytes(
        self,
        project_id: int,
        name: str,
        data: bytes,
        *,
        source_type: SourceType | None = None,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest in-memory content under the project-relative ``name``."""
        name = sanitize_source_name(name)
        stype = source_type or detect_source_type(name)
        if len(data) > self._max_bytes:
            raise FileTooLargeError(
                f"{name}: {len(data)} bytes exceeds the limit of {self._max_bytes} bytes"
            )
        digest = hashlib.sha256(data).hexdigest()

        with self._db.session() as s:
            s.projects.require(project_id)
            existing = s.sources.get_by_name(project_id, name)
            if existing and existing.content_hash == digest and not force:
                log.info("Unchanged, skipping %s (project %d)", name, project_id)
                return IngestionResult(
                    existing, IngestionStatus.UNCHANGED,
                    entity_count=len(s.entities.list_for_source(existing.id)),
                    chunk_count=len(s.chunks.list_for_source(existing.id)),
                )

        extraction = extract_source(stype, name, data, self._cfg)  # may raise ParsingError
        stored_path, created_file = self._store.save(project_id, digest, stype, data)

        try:
            result, stale_path = self._persist(
                project_id, name, stype, digest, len(data), stored_path, extraction
            )
        except BaseException:
            if created_file:
                self._store.delete(stored_path)
            raise

        if stale_path:
            self._cleanup_if_unreferenced(project_id, stale_path)
        log.info(
            "%s %s (project %d): %d entities, %d chunks",
            result.status.value.capitalize(), name, project_id,
            result.entity_count, result.chunk_count,
        )
        return result

    def ingest_file(
        self,
        project_id: int,
        path: str | Path,
        *,
        name: str | None = None,
        source_type: SourceType | None = None,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest a file from disk. ``name`` defaults to the file's base name."""
        file_path = Path(path)
        if not file_path.is_file():
            raise IngestionError(f"Not a file: {file_path.name}")
        size = file_path.stat().st_size
        if size > self._max_bytes:
            raise FileTooLargeError(
                f"{file_path.name}: {size} bytes exceeds the limit of {self._max_bytes} bytes"
            )
        return self.ingest_bytes(
            project_id, name or file_path.name, file_path.read_bytes(),
            source_type=source_type, force=force,
        )

    def ingest_directory(
        self, project_id: int, root: str | Path, *, force: bool = False
    ) -> BatchIngestionResult:
        """Ingest every supported file under ``root`` (recursively, symlinks ignored).

        Names are paths relative to ``root``. One bad file never aborts the batch;
        it is reported in ``failures`` instead.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            raise IngestionError(f"Not a directory: {root_path.name}")
        batch = BatchIngestionResult()
        for current, dirs, files in os.walk(root_path, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs
                if d not in IGNORED_DIRECTORIES and not d.startswith(".")
                and not Path(current, d).is_symlink()
            )
            for filename in sorted(files):
                full = Path(current, filename)
                if full.is_symlink():
                    continue
                if full.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    batch.skipped_unsupported += 1
                    continue
                rel = full.relative_to(root_path).as_posix()
                try:
                    batch.results.append(self.ingest_file(project_id, full, name=rel, force=force))
                except (IngestionError, TrustDocError) as exc:
                    log.warning("Failed to ingest %s: %s", rel, exc)
                    batch.failures.append(IngestionFailure(rel, str(exc)))
        return batch

    def remove_source(self, source_id: int) -> bool:
        """Delete a source (entities, chunks, claims cascade) and its stored file."""
        with self._db.session() as s:
            source = s.sources.get(source_id)
            if source is None:
                return False
            s.sources.delete(source_id)
            project_id, stored = source.project_id, source.stored_path
        if stored:
            self._cleanup_if_unreferenced(project_id, stored)
        return True

    # ------------------------------------------------------------- internals
    def _persist(
        self,
        project_id: int,
        name: str,
        stype: SourceType,
        digest: str,
        size: int,
        stored_path: str,
        extraction: Extraction,
    ) -> tuple[IngestionResult, str | None]:
        warnings = list(extraction.warnings)
        with self._db.session() as s:
            existing = s.sources.get_by_name(project_id, name)
            status = IngestionStatus.INGESTED
            replaced_id = stale_path = None
            if existing:
                status = IngestionStatus.UPDATED
                replaced_id = existing.id
                if existing.stored_path != stored_path:
                    stale_path = existing.stored_path
                s.sources.delete(existing.id)

            twin = next(
                (x for x in s.sources.list_for_project(project_id)
                 if x.content_hash == digest and x.name != name),
                None,
            )
            if twin:
                warnings.append(f"{name}: identical content to existing source {twin.name!r}")

            metadata = {**extraction.metadata}
            if warnings:
                metadata["warnings"] = warnings
            if twin:
                metadata["duplicate_of"] = twin.name

            source = s.sources.add(Source(
                project_id=project_id, source_type=stype, name=name, content_hash=digest,
                size_bytes=size, stored_path=stored_path, metadata=metadata,
            ))
            self._persist_content(s, project_id, source, extraction)
            result = IngestionResult(
                source=source, status=status, entity_count=len(extraction.entities),
                chunk_count=len(extraction.chunks), warnings=warnings,
                replaced_source_id=replaced_id,
            )
        return result, stale_path

    @staticmethod
    def _persist_content(s: Session, project_id: int, source: Source, ex: Extraction) -> None:
        saved = []
        for item in ex.entities:  # parents always precede children
            entity = item.entity
            entity.project_id, entity.source_id = project_id, source.id
            entity.parent_id = saved[item.parent_index].id if item.parent_index is not None else None
            saved.append(s.entities.add(entity))
        chunks = []
        for draft in ex.chunks:
            chunk = draft.chunk
            chunk.project_id, chunk.source_id = project_id, source.id
            chunk.entity_id = saved[draft.entity_index].id if draft.entity_index is not None else None
            chunks.append(chunk)
        s.chunks.add_many(chunks)

    def _cleanup_if_unreferenced(self, project_id: int, stored_path: str) -> None:
        with self._db.session() as s:
            still_used = any(x.stored_path == stored_path for x in s.sources.list_for_project(project_id))
        if not still_used:
            self._store.delete(stored_path)
