"""Pure extraction: bytes -> entities, chunk drafts and source metadata.

No database, no filesystem. Failures raise ``ParsingError`` before anything is
persisted, which is what keeps ingestion atomic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from trustdoc.chunking import (
    ChunkDraft,
    build_markdown_chunks,
    build_pdf_chunks,
    build_python_chunks,
)
from trustdoc.config import ChunkingConfig
from trustdoc.models import SourceType
from trustdoc.parsing.markdown_parser import parse_markdown
from trustdoc.parsing.pdf_reader import read_pdf
from trustdoc.parsing.python_analyzer import ExtractedEntity, analyze_python


@dataclass
class Extraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    chunks: list[ChunkDraft] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def extract_source(
    source_type: SourceType, name: str, data: bytes, cfg: ChunkingConfig
) -> Extraction:
    if source_type == SourceType.PYTHON:
        analysis = analyze_python(data, name)
        counts = Counter(e.entity.entity_type.value for e in analysis.entities)
        return Extraction(
            entities=analysis.entities,
            chunks=build_python_chunks(analysis, name, cfg),
            metadata={
                "language": "python",
                "module_name": analysis.module_name,
                "encoding": analysis.encoding,
                "line_count": len(analysis.lines),
                "entity_counts": dict(counts),
            },
        )

    if source_type == SourceType.MARKDOWN:
        doc = parse_markdown(data, name)
        meta = {k: v for k, v in doc.metadata.items()}
        return Extraction(chunks=build_markdown_chunks(doc, name, cfg), metadata=meta)

    if source_type == SourceType.PDF:
        pdf = read_pdf(data, name)
        warnings: list[str] = []
        meta = dict(pdf.metadata)
        if pdf.pages and not meta["pages_with_text"]:
            meta["no_extractable_text"] = True
            warnings.append(
                f"{name}: no extractable text (scanned/image-only PDF?). OCR is not supported."
            )
        elif meta["empty_pages"]:
            warnings.append(f"{name}: pages without extractable text: {meta['empty_pages']}")
        return Extraction(chunks=build_pdf_chunks(pdf, name, cfg), metadata=meta, warnings=warnings)

    raise ValueError(f"Unhandled source type: {source_type}")
