"""Turn parsed sources into Chunk drafts with full provenance.

All three builders use ``chunk_text`` for size limits, so max_chars /
overlap_chars from the chunking config apply uniformly. Drafts carry
placeholder ids (0) and an ``entity_index``; the ingestion service stamps real
ids when persisting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from trustdoc.chunking.text import TextSegment, chunk_text
from trustdoc.config import ChunkingConfig
from trustdoc.models import Chunk, ChunkType, EntityType
from trustdoc.parsing.markdown_parser import MarkdownDocument
from trustdoc.parsing.pdf_reader import PdfDocument
from trustdoc.parsing.python_analyzer import UNASSIGNED, PythonAnalysis


@dataclass
class ChunkDraft:
    chunk: Chunk
    entity_index: int | None = None   # index into PythonAnalysis.entities


@dataclass
class _Counter:
    value: int = 0

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def _split(text: str, cfg: ChunkingConfig, base_line: int = 1) -> list[TextSegment]:
    return chunk_text(text, cfg.max_chars, cfg.overlap_chars, base_line)


# --------------------------------------------------------------------- python
def build_python_chunks(
    analysis: PythonAnalysis, source_name: str, cfg: ChunkingConfig
) -> list[ChunkDraft]:
    """One logical chunk per module-level region, class header, function and method.

    * A function/method chunk is its whole definition (decorators included),
      with any nested functions/classes kept inside it.
    * A class chunk holds the lines the class owns itself (header, docstring,
      attributes); its methods and nested classes get their own chunks.
    * Module-level code (imports, constants, scripts) becomes module chunks.
    Over-long regions are split by the shared chunker (``part``/``parts``).
    """
    entities = analysis.entities
    lines = analysis.lines
    bearing = {
        i for i, e in enumerate(entities)
        if e.entity.entity_type == EntityType.MODULE or not e.in_function
    }
    children: dict[int, list[int]] = defaultdict(list)
    for i in bearing:
        parent = entities[i].parent_index
        if parent is not None:
            children[parent].append(i)

    pieces: list[tuple[int, int, int, int]] = []  # (start, end, entity_index, depth)
    for i in sorted(bearing):
        ent = entities[i].entity
        covered = set()
        for c in children[i]:
            child = entities[c].entity
            covered.update(range(child.start_line, child.end_line + 1))
        depth = ent.qualified_name.count(".")
        run_start: int | None = None
        for ln in range(ent.start_line, ent.end_line + 1):
            if ln in covered:
                if run_start is not None:
                    pieces.append((run_start, ln - 1, i, depth))
                    run_start = None
            elif run_start is None:
                run_start = ln
        if run_start is not None:
            pieces.append((run_start, ent.end_line, i, depth))

    counter = _Counter()
    drafts: list[ChunkDraft] = []
    for start, end, i, _depth in sorted(pieces, key=lambda p: (p[0], p[3])):
        while start <= end and not lines[start - 1].strip():
            start += 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        if start > end:
            continue
        ex = entities[i]
        ent = ex.entity
        region = ("module_level" if ent.entity_type == EntityType.MODULE
                  else "class_body" if ent.entity_type == EntityType.CLASS else "definition")
        parent_q = (entities[ex.parent_index].entity.qualified_name
                    if ex.parent_index is not None else None)
        base_meta: dict[str, Any] = {
            "language": "python",
            "module": analysis.module_name,
            "qualified_name": ent.qualified_name,
            "entity_type": ent.entity_type.value,
            "parent_qualified_name": parent_q,
            "signature": ex.signature,
            "region": region,
            "context_header": f"{source_name} :: {ent.qualified_name}",
        }
        text = "\n".join(lines[start - 1 : end])
        segments = _split(text, cfg, base_line=start)
        for part, seg in enumerate(segments, start=1):
            drafts.append(ChunkDraft(
                Chunk(
                    project_id=UNASSIGNED, source_id=UNASSIGNED, chunk_index=counter.next(),
                    chunk_type=ChunkType.CODE, text=seg.text,
                    start_line=seg.start_line, end_line=seg.end_line,
                    metadata={**base_meta, "part": part, "parts": len(segments)},
                ),
                entity_index=i,
            ))
    return drafts


# ------------------------------------------------------------------- markdown
def build_markdown_chunks(
    doc: MarkdownDocument, source_name: str, cfg: ChunkingConfig
) -> list[ChunkDraft]:
    """One chunk per heading section (split further when longer than max_chars).

    Heading-only sections (a heading with no body of its own) are skipped; the
    heading still appears in the ``heading`` path of its descendants.
    """
    counter = _Counter()
    drafts: list[ChunkDraft] = []
    for section in doc.sections:
        if not section.has_body:
            continue
        label = section.path_label
        segments = _split(section.text, cfg, base_line=section.start_line)
        for part, seg in enumerate(segments, start=1):
            drafts.append(ChunkDraft(Chunk(
                project_id=UNASSIGNED, source_id=UNASSIGNED, chunk_index=counter.next(),
                chunk_type=ChunkType.MARKDOWN, text=seg.text,
                start_line=seg.start_line, end_line=seg.end_line, heading=label,
                metadata={
                    "heading_level": section.level,
                    "heading_path": list(section.heading_path),
                    "section_start_line": section.start_line,
                    "section_end_line": section.end_line,
                    "context_header": f"{source_name} \u00a7 {label}" if label else source_name,
                    "part": part, "parts": len(segments),
                },
            )))
    return drafts


# ------------------------------------------------------------------------ pdf
def build_pdf_chunks(doc: PdfDocument, source_name: str, cfg: ChunkingConfig) -> list[ChunkDraft]:
    """Chunks never span pages, so every chunk maps to exactly one page number."""
    counter = _Counter()
    drafts: list[ChunkDraft] = []
    page_count = len(doc.pages)
    for page in doc.pages:
        if not page.has_text:
            continue
        segments = _split(page.text, cfg)
        for part, seg in enumerate(segments, start=1):
            drafts.append(ChunkDraft(Chunk(
                project_id=UNASSIGNED, source_id=UNASSIGNED, chunk_index=counter.next(),
                chunk_type=ChunkType.PDF_TEXT, text=seg.text, page_number=page.number,
                metadata={
                    "page_count": page_count,
                    "char_start": seg.start_char, "char_end": seg.end_char,
                    "context_header": f"{source_name}, p.{page.number}",
                    "part": part, "parts": len(segments),
                },
            )))
    return drafts
