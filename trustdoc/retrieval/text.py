"""Text preparation shared by the BM25 and embedding indexes.

Both indexes see the *same* text per chunk (``chunk_index_text``), and BM25 uses
one tokenizer for documents and queries, so indexing and querying cannot drift.
Bump ``TOKENIZER_VERSION`` when tokenization changes; stored indexes are then
reported as stale and rebuilt.
"""

from __future__ import annotations

import re

from trustdoc.models import Chunk

TOKENIZER_VERSION = 1

_WORD = re.compile(r"\w+", re.UNICODE)
_CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep identifiers *and* their parts.

    ``load_config`` -> ``load_config, load, config``; ``WorkerPool`` ->
    ``workerpool, worker, pool``. Exact identifier queries therefore match the
    whole identifier (highest evidence) while natural-language words still match
    the parts. Dotted names split at dots (``a.b.C`` -> ``a, b, c``).
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower().strip("_")
        if not lower:
            continue
        tokens.append(lower)
        spaced = _CAMEL_2.sub(r"\1 \2", _CAMEL_1.sub(r"\1 \2", word))
        parts = [p.lower() for chunk in spaced.split() for p in chunk.split("_") if p]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p != lower)
    return tokens


def chunk_index_text(chunk: Chunk) -> str:
    """Text that is indexed/embedded/reranked: provenance header + chunk text.

    The header (e.g. ``pkg/a.py :: pkg.a.Store.put``) lets class and module
    names match for methods whose source only mentions the method name. It
    holds logical names only, never a filesystem path.
    """
    header = chunk.metadata.get("context_header")
    return f"{header}\n{chunk.text}" if header else chunk.text
