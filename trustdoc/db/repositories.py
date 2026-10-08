"""Repositories: the only place SQL lives. Each is bound to one connection."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Sequence

from trustdoc.exceptions import DatabaseError, DuplicateError, NotFoundError
from trustdoc.models import (
    Chunk,
    ChunkType,
    Claim,
    ClaimOrigin,
    ClaimResult,
    ClaimType,
    ClaimVerdict,
    CodeEntity,
    EntityType,
    EvidenceLink,
    EvidenceStance,
    Finding,
    IssueType,
    Parameter,
    Project,
    RunStatus,
    Severity,
    Source,
    SourceType,
    TrustScoreBreakdown,
    TrustScoreComponent,
    VerificationRun,
    VerificationTier,
    utcnow,
)

_IN_BATCH = 500  # stay well below SQLite's variable limit


# ------------------------------------------------------------------- helpers
def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


class _Repo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _insert(self, sql: str, params: Sequence[Any]) -> int:
        try:
            cur = self._conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc):
                raise DuplicateError(str(exc)) from exc
            raise DatabaseError(str(exc)) from exc
        return int(cur.lastrowid)

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()


# ------------------------------------------------------------------ projects
class ProjectRepository(_Repo):
    def create(self, project: Project) -> Project:
        project.id = self._insert(
            "INSERT INTO projects (name, description, created_at, updated_at) VALUES (?,?,?,?)",
            (project.name, project.description, _ts(project.created_at), _ts(project.updated_at)),
        )
        return project

    def get(self, project_id: int) -> Project | None:
        row = self._one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return self._map(row) if row else None

    def require(self, project_id: int) -> Project:
        project = self.get(project_id)
        if project is None:
            raise NotFoundError(f"Project {project_id} not found")
        return project

    def get_by_name(self, name: str) -> Project | None:
        row = self._one("SELECT * FROM projects WHERE name = ?", (name,))
        return self._map(row) if row else None

    def list(self) -> list[Project]:
        return [self._map(r) for r in self._all("SELECT * FROM projects ORDER BY name")]

    def update(self, project: Project) -> Project:
        if project.id is None:
            raise ValueError("Cannot update an unsaved project")
        project.updated_at = utcnow()
        self._conn.execute(
            "UPDATE projects SET name=?, description=?, updated_at=? WHERE id=?",
            (project.name, project.description, _ts(project.updated_at), project.id),
        )
        return project

    def delete(self, project_id: int) -> bool:
        """Delete a project and (via cascade) everything that belongs to it."""
        return self._conn.execute("DELETE FROM projects WHERE id=?", (project_id,)).rowcount > 0

    @staticmethod
    def _map(r: sqlite3.Row) -> Project:
        return Project(
            id=r["id"], name=r["name"], description=r["description"],
            created_at=_dt(r["created_at"]), updated_at=_dt(r["updated_at"]),
        )


# ------------------------------------------------------------------- sources
class SourceRepository(_Repo):
    def add(self, source: Source) -> Source:
        source.id = self._insert(
            """INSERT INTO sources (project_id, source_type, name, content_hash, size_bytes,
                                    stored_path, metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (source.project_id, source.source_type.value, source.name, source.content_hash,
             source.size_bytes, source.stored_path, _dumps(source.metadata), _ts(source.created_at)),
        )
        return source

    def get(self, source_id: int) -> Source | None:
        row = self._one("SELECT * FROM sources WHERE id=?", (source_id,))
        return self._map(row) if row else None

    def require(self, source_id: int) -> Source:
        source = self.get(source_id)
        if source is None:
            raise NotFoundError(f"Source {source_id} not found")
        return source

    def get_by_name(self, project_id: int, name: str) -> Source | None:
        row = self._one("SELECT * FROM sources WHERE project_id=? AND name=?", (project_id, name))
        return self._map(row) if row else None

    def list_for_project(
        self, project_id: int, source_type: SourceType | None = None
    ) -> list[Source]:
        sql, params = "SELECT * FROM sources WHERE project_id=?", [project_id]
        if source_type is not None:
            sql += " AND source_type=?"
            params.append(source_type.value)
        return [self._map(r) for r in self._all(sql + " ORDER BY name", params)]

    def delete(self, source_id: int) -> bool:
        """Delete a source; its entities, chunks and claims cascade."""
        return self._conn.execute("DELETE FROM sources WHERE id=?", (source_id,)).rowcount > 0

    @staticmethod
    def _map(r: sqlite3.Row) -> Source:
        return Source(
            id=r["id"], project_id=r["project_id"], source_type=SourceType(r["source_type"]),
            name=r["name"], content_hash=r["content_hash"], size_bytes=r["size_bytes"],
            stored_path=r["stored_path"], metadata=json.loads(r["metadata_json"]),
            created_at=_dt(r["created_at"]),
        )


# ------------------------------------------------------------------ entities
class EntityRepository(_Repo):
    def add(self, entity: CodeEntity) -> CodeEntity:
        entity.id = self._insert(
            """INSERT INTO code_entities
               (project_id, source_id, parent_id, entity_type, name, qualified_name,
                parameters_json, return_annotation, docstring, decorators_json,
                is_async, is_public, start_line, end_line, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (entity.project_id, entity.source_id, entity.parent_id, entity.entity_type.value,
             entity.name, entity.qualified_name,
             _dumps([p.to_dict() for p in entity.parameters]), entity.return_annotation,
             entity.docstring, _dumps(entity.decorators), int(entity.is_async),
             int(entity.is_public), entity.start_line, entity.end_line, _ts(entity.created_at)),
        )
        return entity

    def add_many(self, entities: Iterable[CodeEntity]) -> list[CodeEntity]:
        """Insert in order; a parent must precede its children so ``parent_id`` is set."""
        return [self.add(e) for e in entities]

    def get(self, entity_id: int) -> CodeEntity | None:
        row = self._one("SELECT * FROM code_entities WHERE id=?", (entity_id,))
        return self._map(row) if row else None

    def list_for_project(
        self, project_id: int, entity_type: EntityType | None = None
    ) -> list[CodeEntity]:
        sql, params = "SELECT * FROM code_entities WHERE project_id=?", [project_id]
        if entity_type is not None:
            sql += " AND entity_type=?"
            params.append(entity_type.value)
        return [self._map(r) for r in self._all(sql + " ORDER BY source_id, start_line, id", params)]

    def list_for_source(self, source_id: int) -> list[CodeEntity]:
        rows = self._all(
            "SELECT * FROM code_entities WHERE source_id=? ORDER BY start_line, id", (source_id,)
        )
        return [self._map(r) for r in rows]

    def find_by_qualified_name(self, project_id: int, qualified_name: str) -> list[CodeEntity]:
        rows = self._all(
            "SELECT * FROM code_entities WHERE project_id=? AND qualified_name=?",
            (project_id, qualified_name),
        )
        return [self._map(r) for r in rows]

    @staticmethod
    def _map(r: sqlite3.Row) -> CodeEntity:
        return CodeEntity(
            id=r["id"], project_id=r["project_id"], source_id=r["source_id"],
            parent_id=r["parent_id"], entity_type=EntityType(r["entity_type"]),
            name=r["name"], qualified_name=r["qualified_name"],
            parameters=[Parameter.from_dict(p) for p in json.loads(r["parameters_json"])],
            return_annotation=r["return_annotation"], docstring=r["docstring"],
            decorators=json.loads(r["decorators_json"]), is_async=bool(r["is_async"]),
            is_public=bool(r["is_public"]), start_line=r["start_line"], end_line=r["end_line"],
            created_at=_dt(r["created_at"]),
        )


# -------------------------------------------------------------------- chunks
class ChunkRepository(_Repo):
    def add(self, chunk: Chunk) -> Chunk:
        chunk.id = self._insert(
            """INSERT INTO chunks
               (project_id, source_id, entity_id, chunk_index, chunk_type, text, char_count,
                start_line, end_line, page_number, heading, metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (chunk.project_id, chunk.source_id, chunk.entity_id, chunk.chunk_index,
             chunk.chunk_type.value, chunk.text, chunk.char_count, chunk.start_line,
             chunk.end_line, chunk.page_number, chunk.heading, _dumps(chunk.metadata),
             _ts(chunk.created_at)),
        )
        return chunk

    def add_many(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        return [self.add(c) for c in chunks]

    def get(self, chunk_id: int) -> Chunk | None:
        row = self._one("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        return self._map(row) if row else None

    def get_many(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Fetch chunks by id, returned in the order requested (missing ids skipped)."""
        found: dict[int, Chunk] = {}
        ids = list(chunk_ids)
        for i in range(0, len(ids), _IN_BATCH):
            batch = ids[i : i + _IN_BATCH]
            marks = ",".join("?" * len(batch))
            for r in self._all(f"SELECT * FROM chunks WHERE id IN ({marks})", batch):  # noqa: S608
                found[r["id"]] = self._map(r)
        return [found[i] for i in ids if i in found]

    def list_for_source(self, source_id: int) -> list[Chunk]:
        rows = self._all("SELECT * FROM chunks WHERE source_id=? ORDER BY chunk_index", (source_id,))
        return [self._map(r) for r in rows]

    def list_for_project(self, project_id: int) -> list[Chunk]:
        rows = self._all(
            "SELECT * FROM chunks WHERE project_id=? ORDER BY source_id, chunk_index", (project_id,)
        )
        return [self._map(r) for r in rows]

    def count_for_project(self, project_id: int) -> int:
        return int(self._one("SELECT COUNT(*) FROM chunks WHERE project_id=?", (project_id,))[0])

    def delete_for_source(self, source_id: int) -> int:
        """Remove a source's chunks (e.g. before re-chunking)."""
        return self._conn.execute("DELETE FROM chunks WHERE source_id=?", (source_id,)).rowcount

    @staticmethod
    def _map(r: sqlite3.Row) -> Chunk:
        return Chunk(
            id=r["id"], project_id=r["project_id"], source_id=r["source_id"],
            entity_id=r["entity_id"], chunk_index=r["chunk_index"],
            chunk_type=ChunkType(r["chunk_type"]), text=r["text"], start_line=r["start_line"],
            end_line=r["end_line"], page_number=r["page_number"], heading=r["heading"],
            metadata=json.loads(r["metadata_json"]), created_at=_dt(r["created_at"]),
        )


# -------------------------------------------------------------------- claims
class ClaimRepository(_Repo):
    def add(self, claim: Claim) -> Claim:
        claim.id = self._insert(
            """INSERT INTO claims
               (project_id, source_id, chunk_id, claim_type, origin, text, subject_name,
                start_line, page_number, extracted_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (claim.project_id, claim.source_id, claim.chunk_id, claim.claim_type.value,
             claim.origin.value, claim.text, claim.subject_name, claim.start_line,
             claim.page_number, _dumps(claim.extracted), _ts(claim.created_at)),
        )
        return claim

    def add_many(self, claims: Iterable[Claim]) -> list[Claim]:
        return [self.add(c) for c in claims]

    def get(self, claim_id: int) -> Claim | None:
        row = self._one("SELECT * FROM claims WHERE id=?", (claim_id,))
        return self._map(row) if row else None

    def list_for_project(
        self, project_id: int, claim_type: ClaimType | None = None
    ) -> list[Claim]:
        sql, params = "SELECT * FROM claims WHERE project_id=?", [project_id]
        if claim_type is not None:
            sql += " AND claim_type=?"
            params.append(claim_type.value)
        return [self._map(r) for r in self._all(sql + " ORDER BY source_id, id", params)]

    def list_for_source(self, source_id: int) -> list[Claim]:
        return [self._map(r) for r in self._all(
            "SELECT * FROM claims WHERE source_id=? ORDER BY id", (source_id,))]

    @staticmethod
    def _map(r: sqlite3.Row) -> Claim:
        return Claim(
            id=r["id"], project_id=r["project_id"], source_id=r["source_id"],
            chunk_id=r["chunk_id"], claim_type=ClaimType(r["claim_type"]),
            origin=ClaimOrigin(r["origin"]), text=r["text"], subject_name=r["subject_name"],
            start_line=r["start_line"], page_number=r["page_number"],
            extracted=json.loads(r["extracted_json"]), created_at=_dt(r["created_at"]),
        )


# ---------------------------------------------------------------------- runs
class RunRepository(_Repo):
    def create(self, run: VerificationRun) -> VerificationRun:
        run.id = self._insert(
            """INSERT INTO verification_runs
               (project_id, status, trust_score, config_json, summary_json, error_message,
                created_at, started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run.project_id, run.status.value, run.trust_score, _dumps(run.config_snapshot),
             _dumps(run.summary), run.error_message, _ts(run.created_at), _ts(run.started_at),
             _ts(run.finished_at)),
        )
        return run

    def get(self, run_id: int) -> VerificationRun | None:
        row = self._one("SELECT * FROM verification_runs WHERE id=?", (run_id,))
        return self._map(row) if row else None

    def require(self, run_id: int) -> VerificationRun:
        run = self.get(run_id)
        if run is None:
            raise NotFoundError(f"Verification run {run_id} not found")
        return run

    def list_for_project(self, project_id: int) -> list[VerificationRun]:
        rows = self._all(
            "SELECT * FROM verification_runs WHERE project_id=? ORDER BY id DESC", (project_id,)
        )
        return [self._map(r) for r in rows]

    def mark_running(self, run_id: int) -> None:
        self._conn.execute(
            "UPDATE verification_runs SET status=?, started_at=? WHERE id=?",
            (RunStatus.RUNNING.value, _ts(utcnow()), run_id),
        )

    def complete(
        self, run_id: int, trust_score: float | None = None, summary: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            """UPDATE verification_runs
               SET status=?, finished_at=?, trust_score=COALESCE(?, trust_score),
                   summary_json=COALESCE(?, summary_json)
               WHERE id=?""",
            (RunStatus.COMPLETED.value, _ts(utcnow()), trust_score,
             _dumps(summary) if summary is not None else None, run_id),
        )

    def fail(self, run_id: int, error_message: str) -> None:
        self._conn.execute(
            "UPDATE verification_runs SET status=?, finished_at=?, error_message=? WHERE id=?",
            (RunStatus.FAILED.value, _ts(utcnow()), error_message, run_id),
        )

    # ---- Trust Score breakdown
    def save_trust_breakdown(self, run_id: int, breakdown: TrustScoreBreakdown) -> None:
        """Replace the run's stored breakdown and record its final score on the run."""
        self._conn.execute("DELETE FROM trust_score_components WHERE run_id=?", (run_id,))
        for position, c in enumerate(breakdown.components):
            self._insert(
                """INSERT INTO trust_score_components
                   (run_id, position, name, raw_score, weight, contribution, detail)
                   VALUES (?,?,?,?,?,?,?)""",
                (run_id, position, c.name, c.raw_score, c.weight, c.contribution, c.detail),
            )
        self._conn.execute(
            "UPDATE verification_runs SET trust_score=? WHERE id=?", (breakdown.final_score, run_id)
        )

    def get_trust_breakdown(self, run_id: int) -> TrustScoreBreakdown | None:
        run = self.get(run_id)
        if run is None or run.trust_score is None:
            return None
        rows = self._all(
            "SELECT * FROM trust_score_components WHERE run_id=? ORDER BY position", (run_id,)
        )
        components = [
            TrustScoreComponent(
                name=r["name"], raw_score=r["raw_score"], weight=r["weight"],
                contribution=r["contribution"], detail=r["detail"],
            )
            for r in rows
        ]
        return TrustScoreBreakdown(components=components, final_score=run.trust_score)

    @staticmethod
    def _map(r: sqlite3.Row) -> VerificationRun:
        return VerificationRun(
            id=r["id"], project_id=r["project_id"], status=RunStatus(r["status"]),
            trust_score=r["trust_score"], config_snapshot=json.loads(r["config_json"]),
            summary=json.loads(r["summary_json"]), error_message=r["error_message"],
            created_at=_dt(r["created_at"]), started_at=_dt(r["started_at"]),
            finished_at=_dt(r["finished_at"]),
        )


# ------------------------------------------------------- findings & verdicts
class FindingRepository(_Repo):
    def add(self, f: Finding) -> Finding:
        f.id = self._insert(
            """INSERT INTO findings
               (run_id, project_id, tier, issue_type, severity, message, suggestion, source_id,
                entity_id, claim_id, start_line, details_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f.run_id, f.project_id, f.tier.value, f.issue_type.value, f.severity.value,
             f.message, f.suggestion, f.source_id, f.entity_id, f.claim_id, f.start_line,
             _dumps(f.details), _ts(f.created_at)),
        )
        return f

    def add_many(self, findings: Iterable[Finding]) -> list[Finding]:
        return [self.add(f) for f in findings]

    def list_for_run(
        self,
        run_id: int,
        severity: Severity | None = None,
        issue_type: IssueType | None = None,
    ) -> list[Finding]:
        sql, params = "SELECT * FROM findings WHERE run_id=?", [run_id]
        if severity is not None:
            sql += " AND severity=?"
            params.append(severity.value)
        if issue_type is not None:
            sql += " AND issue_type=?"
            params.append(issue_type.value)
        return [self._map(r) for r in self._all(sql + " ORDER BY id", params)]

    def count_by_severity(self, run_id: int) -> dict[Severity, int]:
        rows = self._all(
            "SELECT severity, COUNT(*) AS n FROM findings WHERE run_id=? GROUP BY severity", (run_id,)
        )
        return {Severity(r["severity"]): r["n"] for r in rows}

    def add_claim_result(self, result: ClaimResult) -> ClaimResult:
        result.id = self._insert(
            """INSERT INTO claim_results
               (run_id, claim_id, tier, verdict, confidence, rationale, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (result.run_id, result.claim_id, result.tier.value, result.verdict.value,
             result.confidence, result.rationale, _ts(result.created_at)),
        )
        for link in result.evidence:
            link.claim_result_id = result.id
            link.id = self._insert(
                """INSERT INTO evidence_links
                   (claim_result_id, chunk_id, stance, relevance_score, rank)
                   VALUES (?,?,?,?,?)""",
                (result.id, link.chunk_id, link.stance.value, link.relevance_score, link.rank),
            )
        return result

    def list_claim_results(self, run_id: int) -> list[ClaimResult]:
        rows = self._all("SELECT * FROM claim_results WHERE run_id=? ORDER BY id", (run_id,))
        results = [
            ClaimResult(
                id=r["id"], run_id=r["run_id"], claim_id=r["claim_id"],
                tier=VerificationTier(r["tier"]), verdict=ClaimVerdict(r["verdict"]),
                confidence=r["confidence"], rationale=r["rationale"],
                created_at=_dt(r["created_at"]),
            )
            for r in rows
        ]
        by_id = {r.id: r for r in results}
        ids = list(by_id)
        for i in range(0, len(ids), _IN_BATCH):
            batch = ids[i : i + _IN_BATCH]
            marks = ",".join("?" * len(batch))
            for e in self._all(
                f"SELECT * FROM evidence_links WHERE claim_result_id IN ({marks}) "  # noqa: S608
                "ORDER BY claim_result_id, rank, id",
                batch,
            ):
                by_id[e["claim_result_id"]].evidence.append(
                    EvidenceLink(
                        id=e["id"], claim_result_id=e["claim_result_id"], chunk_id=e["chunk_id"],
                        stance=EvidenceStance(e["stance"]), relevance_score=e["relevance_score"],
                        rank=e["rank"],
                    )
                )
        return results

    @staticmethod
    def _map(r: sqlite3.Row) -> Finding:
        return Finding(
            id=r["id"], run_id=r["run_id"], project_id=r["project_id"],
            tier=VerificationTier(r["tier"]), issue_type=IssueType(r["issue_type"]),
            severity=Severity(r["severity"]), message=r["message"], suggestion=r["suggestion"],
            source_id=r["source_id"], entity_id=r["entity_id"], claim_id=r["claim_id"],
            start_line=r["start_line"], details=json.loads(r["details_json"]),
            created_at=_dt(r["created_at"]),
        )
