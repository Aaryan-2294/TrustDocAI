"""Deterministic citation-ID validation for generated answers.

Only checks that cited IDs (``[E1]``, ``[E1, E3]``) correspond to evidence that
was actually supplied to the model. It does NOT check that a cited chunk
supports the claim next to it; claim-level verification is Part 5.

Invalid IDs are never "fixed" by guessing a replacement: they are replaced with
an explicit ``[unknown source Ex]`` marker in the sanitised text and reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection

_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"E(\d+)", re.IGNORECASE)


@dataclass
class CitationAnalysis:
    sanitized_text: str
    valid_ids: list[str] = field(default_factory=list)       # unique, first-appearance order
    invalid_ids: list[str] = field(default_factory=list)     # unique, first-appearance order
    mentions: dict[str, int] = field(default_factory=dict)   # valid id -> number of citations

    @property
    def has_valid(self) -> bool:
        return bool(self.valid_ids)


def analyze_citations(text: str, supplied_ids: Collection[str]) -> CitationAnalysis:
    supplied = {i.upper() for i in supplied_ids}
    analysis = CitationAnalysis(sanitized_text=text)

    def replace(match: re.Match) -> str:
        ids = [f"E{int(n)}" for n in _ID.findall(match.group(1))]
        valid = [i for i in dict.fromkeys(ids) if i in supplied]
        invalid = [i for i in dict.fromkeys(ids) if i not in supplied]
        for i in ids:
            if i in supplied:
                analysis.mentions[i] = analysis.mentions.get(i, 0) + 1
                if i not in analysis.valid_ids:
                    analysis.valid_ids.append(i)
            elif i not in analysis.invalid_ids:
                analysis.invalid_ids.append(i)
        parts = []
        if valid:
            parts.append("[" + ", ".join(valid) + "]")
        if invalid:
            noun = "source" if len(invalid) == 1 else "sources"
            parts.append(f"[unknown {noun} {', '.join(invalid)}]")
        return " ".join(parts)

    analysis.sanitized_text = _GROUP.sub(replace, text)
    return analysis
