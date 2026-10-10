"""Project-scoped index lifecycle: build, load, rebuild, staleness, metadata.

Layout (one directory per project, under ``<data_dir>/indexes``)::

    project_<id>/
        manifest.json     what was built, from what, with which model
        faiss.index       FAISS vectors            faiss_ids.npy  position -> chunk id
        bm25.json         tokenised corpus + chunk ids

Staleness is decided from the database, not from timestamps: the manifest stores
a hash of the project's (chunk id, source id, char count) rows and its sources'
content hashes, plus the embedding model name, tokenizer version and index
format version. Any difference marks the index stale. (Re-ingesting a changed
file creates new chunk ids, so changes are always visible in that hash.)

Builds are all-or-nothing: files are written to a temporary directory and swapped
in only when complete. Indexing is an explicit operation - queries never embed
the corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import (
    IndexCorruptError,
    IndexNotBuiltError,
    RetrievalError,
    StaleIndexError,
)
from trustdoc.logging_setup import get_logger
from trustdoc.retrieval.bm25_index import K1, B, Bm25Index
from trustdoc.retrieval.embedding import Embedder, validate_vectors
from trustdoc.retrieval.faiss_index import METRIC, FaissIndex
from trustdoc.retrieval.text import TOKENIZER_VERSION, chunk_index_text
from trustdoc.utils_time import utc_now_iso

log = get_logger("retrieval.index")

INDEX_FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"


class IndexState(str, Enum):
    MISSING = "missing"
    FRESH = "fresh"
    STALE = "stale"
    CORRUPT = "corrupt"


@dataclass
class IndexManifest:
    project_id: int
    build_id: str
    built_at: str
    format_version: int
    tokenizer_version: int
    embedding_model: str
    embedding_dimension: int
    knowledge_hash: str
    chunk_count: int
    source_count: int
    faiss_metric: str = METRIC
    bm25: dict[str, Any] = field(default_factory=lambda: {"library": "rank_bm25.BM25Okapi", "k1": K1, "b": B})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        return cls(**data)


@dataclass
class IndexStatus:
    project_id: int
    state: IndexState
    reasons: list[str] = field(default_factory=list)
    manifest: IndexManifest | None = None
    current_chunk_count: int = 0

    @property
    def is_fresh(self) -> bool:
        return self.state == IndexState.FRESH


@dataclass
class ProjectIndex:
    """Loaded, query-ready indexes for one project."""

    manifest: IndexManifest
    faiss: FaissIndex
    bm25: Bm25Index


class IndexManager:
    def __init__(self, db: Database, settings: Settings, embedder: Embedder) -> None:
        self._db = db
        self._root = Path(settings.storage.indexes_dir)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._cache: dict[int, ProjectIndex] = {}

    # ------------------------------------------------------------------ paths
    def index_dir(self, project_id: int) -> Path:
        return self._root / f"project_{int(project_id)}"

    # ------------------------------------------------------------- inspection
    def metadata(self, project_id: int) -> IndexManifest | None:
        """The stored manifest, or None when there is no (readable) index."""
        try:
            return self._read_manifest(project_id)
        except IndexCorruptError:
            return None

    def status(self, project_id: int) -> IndexStatus:
        """Compare the stored index against the project's current chunks/sources."""
        knowledge_hash, chunk_count, _ = self._current_knowledge(project_id)
        path = self.index_dir(project_id)
        if not (path / MANIFEST_FILE).is_file():
            return IndexStatus(project_id, IndexState.MISSING, ["no index has been built"],
                               current_chunk_count=chunk_count)
        try:
            manifest = self._read_manifest(project_id)
        except IndexCorruptError as exc:
            return IndexStatus(project_id, IndexState.CORRUPT, [str(exc)],
                               current_chunk_count=chunk_count)
        reasons: list[str] = []
        if manifest.format_version != INDEX_FORMAT_VERSION:
            reasons.append("index format version changed")
        if manifest.tokenizer_version != TOKENIZER_VERSION:
            reasons.append("tokenizer version changed")
        if manifest.embedding_model != self._embedder.model_name:
            reasons.append(
                f"embedding model changed ({manifest.embedding_model} -> {self._embedder.model_name})"
            )
        if manifest.knowledge_hash != knowledge_hash:
            reasons.append("project chunks/sources changed since the index was built")
        state = IndexState.STALE if reasons else IndexState.FRESH
        return IndexStatus(project_id, state, reasons, manifest, chunk_count)

    def is_stale(self, project_id: int) -> bool:
        """True unless a fresh index exists (missing/corrupt count as stale)."""
        return not self.status(project_id).is_fresh

    # ------------------------------------------------------------------ build
    def build(self, project_id: int, *, force: bool = False) -> IndexManifest:
        """Build (or refresh) the project's indexes. No-op when already fresh."""
        with self._lock:
            status = self.status(project_id)
            if status.is_fresh and not force:
                log.info("Index for project %d is fresh; nothing to do", project_id)
                return status.manifest  # type: ignore[return-value]

            with self._db.session() as s:
                s.projects.require(project_id)
                chunks = s.chunks.list_for_project(project_id)
            knowledge_hash, _, source_count = self._current_knowledge(project_id)

            chunk_ids = [c.id for c in chunks]
            texts = [chunk_index_text(c) for c in chunks]
            log.info("Building index for project %d: %d chunks", project_id, len(chunks))
            if chunks:
                vectors = self._embedder.embed_documents(texts)
                validate_vectors(vectors, len(chunks))
                faiss_index = FaissIndex.build(chunk_ids, vectors)
            else:
                faiss_index = FaissIndex.build([], None)  # type: ignore[arg-type]
            bm25_index = Bm25Index.build(chunk_ids, texts)

            manifest = IndexManifest(
                project_id=project_id, build_id=uuid.uuid4().hex, built_at=utc_now_iso(),
                format_version=INDEX_FORMAT_VERSION, tokenizer_version=TOKENIZER_VERSION,
                embedding_model=self._embedder.model_name,
                embedding_dimension=faiss_index.dimension, knowledge_hash=knowledge_hash,
                chunk_count=len(chunks), source_count=source_count,
            )
            self._write_atomically(project_id, manifest, faiss_index, bm25_index)
            self._cache[project_id] = ProjectIndex(manifest, faiss_index, bm25_index)
            return manifest

    def rebuild(self, project_id: int) -> IndexManifest:
        return self.build(project_id, force=True)

    # ------------------------------------------------------------------- load
    def load(self, project_id: int, *, allow_stale: bool = False) -> ProjectIndex:
        """Return the query-ready index, loading from disk once and then caching.

        Raises ``IndexNotBuiltError`` / ``StaleIndexError`` / ``IndexCorruptError``.
        """
        with self._lock:
            return self._materialize(project_id, self.status(project_id), allow_stale)

    def _materialize(self, project_id: int, status: IndexStatus, allow_stale: bool = False) -> ProjectIndex:
        with self._lock:
            if status.state == IndexState.MISSING:
                raise IndexNotBuiltError(f"No index for project {project_id}; build it first")
            if status.state == IndexState.CORRUPT:
                raise IndexCorruptError("; ".join(status.reasons))
            if status.state == IndexState.STALE and not allow_stale:
                raise StaleIndexError(
                    f"Index for project {project_id} is stale: {'; '.join(status.reasons)}"
                )
            manifest = status.manifest
            assert manifest is not None
            cached = self._cache.get(project_id)
            if cached and cached.manifest.build_id == manifest.build_id:
                return cached
            directory = self.index_dir(project_id)
            loaded = ProjectIndex(
                manifest,
                FaissIndex.load(directory, manifest.chunk_count),
                Bm25Index.load(directory, manifest.chunk_count),
            )
            self._cache[project_id] = loaded
            log.info("Loaded index for project %d (%d chunks)", project_id, manifest.chunk_count)
            return loaded

    def ensure(self, project_id: int) -> ProjectIndex:
        """Load a fresh index, building/rebuilding first if it is missing, stale or corrupt."""
        with self._lock:
            status = self.status(project_id)
            if status.state != IndexState.FRESH:
                self.build(project_id, force=True)
                status = self.status(project_id)
            return self._materialize(project_id, status)

    def delete(self, project_id: int) -> bool:
        with self._lock:
            self._cache.pop(project_id, None)
            path = self.index_dir(project_id)
            if path.exists():
                shutil.rmtree(path)
                return True
            return False

    # -------------------------------------------------------------- internals
    def _current_knowledge(self, project_id: int) -> tuple[str, int, int]:
        with self._db.session() as s:
            signatures = s.chunks.list_signatures(project_id)
            sources = [(x.id, x.content_hash) for x in s.sources.list_for_project(project_id)]
        payload = json.dumps({"chunks": signatures, "sources": sorted(sources)}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(signatures), len(sources)

    def _read_manifest(self, project_id: int) -> IndexManifest:
        path = self.index_dir(project_id) / MANIFEST_FILE
        try:
            return IndexManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise IndexCorruptError(f"unreadable index manifest for project {project_id}: {exc}") from exc

    def _write_atomically(
        self, project_id: int, manifest: IndexManifest, faiss_index: FaissIndex, bm25: Bm25Index
    ) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        final = self.index_dir(project_id)
        tmp = self._root / f".building-project_{project_id}-{manifest.build_id}"
        old = self._root / f".old-project_{project_id}-{manifest.build_id}"
        try:
            faiss_index.save(tmp)
            bm25.save(tmp)
            (tmp / MANIFEST_FILE).write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
            if final.exists():
                os.replace(final, old)
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            if old.exists() and not final.exists():
                os.replace(old, final)  # restore the previous index
            raise
        finally:
            shutil.rmtree(old, ignore_errors=True)
