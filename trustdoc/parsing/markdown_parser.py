"""Markdown structure extraction (headings -> sections) without rendering.

Sections start at every heading (any level) and run to the line before the next
heading. Text before the first heading is a "preamble" section. Headings inside
fenced code blocks and YAML front matter are ignored. Both ATX (``## Title``)
and Setext (``Title`` + ``===``/``---``) headings are recognised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trustdoc.exceptions import ParsingError
from trustdoc.parsing.textutil import normalize_newlines, split_lines

PATH_SEPARATOR = " > "

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class MarkdownSection:
    heading: str | None               # own heading text; None for the preamble
    level: int                        # 1-6; 0 for the preamble
    heading_path: tuple[str, ...]     # ancestors + own heading
    start_line: int                   # heading line (or first content line)
    end_line: int                     # last non-blank line of the section
    text: str                         # exact lines start_line..end_line
    has_body: bool                    # any content besides the heading line

    @property
    def path_label(self) -> str | None:
        return PATH_SEPARATOR.join(self.heading_path) if self.heading_path else None


@dataclass
class MarkdownDocument:
    lines: list[str]
    sections: list[MarkdownSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def decode_markdown(data: bytes, source_name: str = "<source>") -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParsingError(f"{source_name}: Markdown is not valid UTF-8 ({exc})") from exc
    return normalize_newlines(text)


def parse_markdown(data: bytes, source_name: str = "<source>") -> MarkdownDocument:
    text = decode_markdown(data, source_name)
    lines = split_lines(text) if text else []
    headings = _find_headings(lines)  # list of (line_no, level, text)

    sections: list[MarkdownSection] = []
    stack: list[tuple[int, str]] = []
    bounds = [h[0] for h in headings] + [len(lines) + 1]

    if headings and headings[0][0] > 1 or (not headings and lines):
        end = (headings[0][0] - 1) if headings else len(lines)
        section = _make_section(lines, None, 0, (), 1, end)
        if section:
            sections.append(section)

    for idx, (line_no, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(t for _, t in stack)
        # Setext headings occupy two lines (text + underline); sections always
        # start at the heading's first line.
        section = _make_section(lines, title, level, path, line_no, bounds[idx + 1] - 1)
        if section:
            sections.append(section)

    first_h1 = next((t for _, lv, t in headings if lv == 1), None)
    meta = {
        "line_count": len(lines),
        "heading_count": len(headings),
        "title": first_h1,
        "headings": [{"level": lv, "text": t, "line": ln} for ln, lv, t in headings],
        "has_front_matter": _front_matter_end(lines) > 0,
    }
    return MarkdownDocument(lines=lines, sections=sections, metadata=meta)


def _make_section(
    lines: list[str], heading: str | None, level: int, path: tuple[str, ...], start: int, end: int
) -> MarkdownSection | None:
    # trim trailing blank lines (and leading ones for the preamble)
    while end >= start and not lines[end - 1].strip():
        end -= 1
    while heading is None and start <= end and not lines[start - 1].strip():
        start += 1
    if end < start:
        return None
    heading_lines = 1
    if heading is not None and start < len(lines) and _SETEXT.match(lines[start]) and not _ATX.match(lines[start - 1]):
        heading_lines = 2
    body = [ln for ln in lines[start - 1 + heading_lines : end] if ln.strip()] if heading is not None else [1]
    return MarkdownSection(
        heading=heading, level=level, heading_path=path, start_line=start, end_line=end,
        text="\n".join(lines[start - 1 : end]), has_body=bool(body),
    )


def _front_matter_end(lines: list[str]) -> int:
    """1-based line of the closing delimiter of YAML front matter, else 0."""
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                return i + 1
    return 0


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    fence: tuple[str, int] | None = None
    skip_until = _front_matter_end(lines)
    prev_text = False  # previous line is a plain paragraph line (Setext candidate)
    for i, line in enumerate(lines, start=1):
        if i <= skip_until:
            prev_text = False
            continue
        fm = _FENCE.match(line)
        if fence is not None:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not line.strip()[len(fm.group(1)):].strip():
                fence = None
            prev_text = False
            continue
        if fm:
            fence = (fm.group(1)[0], len(fm.group(1)))
            prev_text = False
            continue
        atx = _ATX.match(line)
        if atx:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", (atx.group(2) or "")).strip()
            title = "" if set(title) == {"#"} else title
            found.append((i, len(atx.group(1)), title or "(untitled)"))
            prev_text = False
            continue
        setext = _SETEXT.match(line)
        if setext and prev_text:
            level = 1 if setext.group(1)[0] == "=" else 2
            found.append((i - 1, level, lines[i - 2].strip()))
            prev_text = False
            continue
        stripped = line.strip()
        prev_text = bool(stripped) and not stripped.startswith(("-", "*", "+", ">", "|")) \
            and not re.match(r"^\d+[.)]\s", stripped)
    return found
