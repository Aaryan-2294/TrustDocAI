"""Text helpers shared by the parsers.

All downstream line numbers are 1-based and computed on text whose newlines
have been normalised to ``\\n`` (so ``\\r\\n`` and ``\\r`` each count as one line).
"""

from __future__ import annotations


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_lines(text: str) -> list[str]:
    """Split normalised text into lines; a trailing newline adds no empty line."""
    lines = text.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines
