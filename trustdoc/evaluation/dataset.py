"""Evaluation dataset: queries with relevance labels that are stable across ingests.

Chunk ids change every time a project is ingested, so labels refer to chunks by
what they are - a source name plus a qualified name (code), a heading path
(Markdown) or a page number (PDF) - and are resolved to chunk ids per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trustdoc.exceptions import TrustDocError
from trustdoc.models import Chunk

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_QUERIES = DATA_DIR / "queries.json"


class EvaluationError(TrustDocError):
    """The dataset does not match the ingested corpus."""


@dataclass(frozen=True)
class RelevantRef:
    source: str
    qualified_name: str | None = None
    heading: str | None = None
    page: int | None = None

    def describe(self) -> str:
        key = self.qualified_name or self.heading or (f"p.{self.page}" if self.page else "?")
        return f"{self.source} :: {key}"

    def matches(self, chunk: Chunk, source_name: str) -> bool:
        if source_name != self.source:
            return False
        if self.qualified_name is not None:
            return chunk.metadata.get("qualified_name") == self.qualified_name
        if self.heading is not None:
            return chunk.heading == self.heading
        if self.page is not None:
            return chunk.page_number == self.page
        return False


@dataclass(frozen=True)
class EvalQuery:
    id: str
    category: str
    text: str
    relevant: tuple[RelevantRef, ...]


@dataclass
class EvalDataset:
    name: str
    description: str
    illustrative: bool
    project_dir: Path
    queries: list[EvalQuery] = field(default_factory=list)

    def categories(self) -> list[str]:
        return sorted({q.category for q in self.queries})


def load_dataset(path: str | Path = DEFAULT_QUERIES) -> EvalDataset:
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for q in raw["queries"]:
        refs = tuple(RelevantRef(**r) for r in q["relevant"])
        if not refs:
            raise EvaluationError(f"query {q['id']!r} has no relevant labels")
        queries.append(EvalQuery(q["id"], q["category"], q["text"], refs))
    ids = [q.id for q in queries]
    if len(set(ids)) != len(ids):
        raise EvaluationError("query ids must be unique")
    return EvalDataset(
        name=raw["name"], description=raw.get("description", ""),
        illustrative=bool(raw.get("illustrative", True)),
        project_dir=(path.parent / raw["project_dir"]).resolve(), queries=queries,
    )


def resolve_relevance(
    dataset: EvalDataset, chunks: list[Chunk], source_names: dict[int, str]
) -> dict[str, list[set[int]]]:
    """Map each query to its relevant groups of chunk ids; fail loudly on unmatched labels."""
    resolved: dict[str, list[set[int]]] = {}
    missing: list[str] = []
    for q in dataset.queries:
        groups: list[set[int]] = []
        for ref in q.relevant:
            ids = {c.id for c in chunks if ref.matches(c, source_names[c.source_id])}
            if not ids:
                missing.append(f"{q.id}: {ref.describe()}")
            groups.append(ids)
        resolved[q.id] = groups
    if missing:
        raise EvaluationError("relevance labels matched no chunk:\n  " + "\n  ".join(missing))
    return resolved
