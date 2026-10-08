"""Text-based PDF extraction with PyMuPDF (optional dependency). No OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trustdoc.exceptions import MissingDependencyError, ParsingError
from trustdoc.parsing.textutil import normalize_newlines

_PDF_META_KEYS = ("title", "author", "subject", "keywords", "creator", "producer",
                  "creationDate", "modDate", "format")


@dataclass(frozen=True)
class PdfPage:
    number: int           # 1-based
    text: str             # newlines normalised; may be empty
    image_count: int = 0

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass
class PdfDocument:
    pages: list[PdfPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24 exposes the `pymupdf` name
        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # legacy name
        if hasattr(fitz, "open"):
            return fitz
    except ImportError:
        pass
    raise MissingDependencyError(
        "PDF ingestion requires PyMuPDF. Install it with: pip install -e \".[pdf]\""
    )


def read_pdf(data: bytes, source_name: str = "<source>") -> PdfDocument:
    pymupdf = _import_pymupdf()
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for corrupt input
        raise ParsingError(f"{source_name}: cannot open PDF ({exc})") from exc
    try:
        if doc.needs_pass:
            raise ParsingError(f"{source_name}: PDF is password-protected")
        pages: list[PdfPage] = []
        try:
            for index in range(doc.page_count):
                page = doc.load_page(index)
                text = normalize_newlines(page.get_text("text") or "")
                pages.append(PdfPage(index + 1, text, len(page.get_images(full=True))))
        except Exception as exc:
            raise ParsingError(f"{source_name}: failed reading page {len(pages) + 1} ({exc})") from exc

        raw = doc.metadata or {}
        info = {k: raw[k] for k in _PDF_META_KEYS if raw.get(k)}
        text_pages = [p.number for p in pages if p.has_text]
        empty_pages = [p.number for p in pages if not p.has_text]
        metadata = {
            "page_count": len(pages),
            "pages_with_text": len(text_pages),
            "empty_pages": empty_pages,
            "image_only_pages": [p.number for p in pages if not p.has_text and p.image_count],
            "char_count": sum(len(p.text.strip()) for p in pages),
            "pdf_info": info,
        }
        return PdfDocument(pages=pages, metadata=metadata)
    finally:
        doc.close()
