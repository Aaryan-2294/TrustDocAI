"""The reusable windowing algorithm used by Python, Markdown and PDF chunking.

``chunk_text`` splits a region of text into segments of at most ``max_chars``
characters. It prefers to cut at paragraph breaks, then line breaks, then
sentence ends, then spaces, and only hard-cuts as a last resort. Consecutive
segments overlap by roughly ``overlap_chars`` (aligned to a line/word start).

Every segment records its character offsets and the 1-based line range it
covers (relative to ``base_line``), which is what lets callers preserve
provenance.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    start_char: int       # offset into the original region text
    end_char: int         # exclusive
    start_line: int       # absolute line numbers (base_line applied)
    end_line: int


def chunk_text(
    text: str, max_chars: int, overlap_chars: int = 0, base_line: int = 1
) -> list[TextSegment]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    newline_offsets = [i for i, ch in enumerate(text) if ch == "\n"]

    def line_of(offset: int) -> int:
        return base_line + bisect_right(newline_offsets, offset - 1)

    segments: list[TextSegment] = []

    def emit(start: int, end: int) -> None:
        trimmed = _trim(text, start, end)
        if trimmed is None:
            return
        s, e = trimmed
        segments.append(TextSegment(text[s:e], s, e, line_of(s), line_of(e - 1)))

    n = len(text)
    if n <= max_chars:
        emit(0, n)
        return segments

    pos = 0
    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            end = _find_cut(text, pos, end, max_chars)
        emit(pos, end)
        if end >= n:
            break
        overlap = min(overlap_chars, (end - pos) // 2)
        nxt = end - overlap
        if overlap:
            nxt = _align_forward(text, nxt, end)
        pos = nxt if nxt > pos else end
    return segments


def _find_cut(text: str, pos: int, end: int, max_chars: int) -> int:
    """Choose the best cut point in (pos + max_chars/2, end]."""
    floor = pos + max(1, max_chars // 2)
    for marker, keep in (("\n\n", 2), ("\n", 1)):
        i = text.rfind(marker, floor, end)
        if i != -1:
            return i + keep
    best = max(text.rfind(". ", floor, end), text.rfind("? ", floor, end),
               text.rfind("! ", floor, end))
    if best != -1:
        return best + 2
    i = text.rfind(" ", floor, end)
    if i != -1:
        return i + 1
    return end


def _align_forward(text: str, start: int, limit: int) -> int:
    """Move ``start`` to the next line start (else next word start) before ``limit``."""
    if start > 0 and text[start - 1] == "\n":
        return start
    nl = text.find("\n", start, limit)
    if nl != -1 and nl + 1 < limit:
        return nl + 1
    sp = text.find(" ", start, limit)
    if sp != -1 and sp + 1 < limit:
        return sp + 1
    return start


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Drop blank leading lines and trailing whitespace; keep first-line indentation."""
    i = start
    while i < end and text[i].isspace():
        i += 1
    if i >= end:
        return None
    nl = text.rfind("\n", start, i)
    if nl != -1:
        start = nl + 1
    j = end
    while j > start and text[j - 1].isspace():
        j -= 1
    return start, j
