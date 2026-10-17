"""Helpers for the verification tests: build code entities and claims without a database."""

from __future__ import annotations

import itertools

from trustdoc.models import ClaimType, CodeEntity, EntityType, Parameter, ParameterKind
from trustdoc.verification.extraction import extract_markdown_claims
from trustdoc.verification.resolver import AliasIndex, EntityResolver
from trustdoc.verification.static_verifier import StaticVerifier

_ids = itertools.count(1)
K = ParameterKind


def P(name, default=None, annotation=None, kind=K.POSITIONAL_OR_KEYWORD):
    return Parameter(name, kind, annotation, default)


def entity(qname, *, type=EntityType.FUNCTION, params=(), ret=None, doc=None, public=True, decorators=(),
           source_id=1, start=10, end=20, id=None):
    return CodeEntity(
        project_id=1, source_id=source_id, entity_type=type, name=qname.rsplit(".", 1)[-1], qualified_name=qname,
        parameters=list(params), return_annotation=ret, docstring=doc, decorators=list(decorators), is_public=public,
        start_line=start, end_line=end, id=id if id is not None else next(_ids),
    )


def module(qname="pkg.mod", **kw):
    return entity(qname, type=EntityType.MODULE, start=1, end=100, **kw)


def md_claims(text, name="README.md"):
    result = extract_markdown_claims(text, name)
    return result.claims


def of_type(claims, category):
    return [c for c in claims if c.claim_type == category]


def only(claims, category):
    found = of_type(claims, category)
    assert len(found) == 1, f"expected exactly one {category}, got {[(c.claim_type.value, c.text) for c in claims]}"
    return found[0]


def verifier(entities, *, aliases=(), chunks=None, source_names=None):
    resolver = EntityResolver(entities, AliasIndex(set(aliases)))
    return StaticVerifier(resolver, source_names or {1: "pkg/mod.py"}, chunks or {})


def check(md_text, entities, category, **kw):
    """Extract one claim of a category from Markdown and verify it statically."""
    v = verifier(entities, **kw)
    return v.verify(only(md_claims(md_text), category))


ENUM = ClaimType


# ---- shared corpus used by the engine, CLI and integration tests
PY = b'''"""Module."""


class TaskQueue:
    """A queue."""

    def __init__(self, backend=None, max_size=10):
        pass

    def enqueue(self, payload, priority=0):
        """Add a task."""

    def purge(self):
        pass


def compute_backoff(attempt, base_seconds=2.0, jitter=True) -> float:
    """Delay.

    Args:
        attempt (int): n
        ghost: nope
    """


def retry_failed(max_attempts=3) -> int:
    """Retry."""


def undocumented(a, b):
    pass


def _private():
    pass
'''
README = b"""# Demo

The `compute_backoff` function accepts `attempt` and `base_delay` and returns a float.

Call `retry_all()` to requeue.

The `retry_failed` function returns an integer.

The `TaskQueue` class persists jobs to disk. Failed tasks are retried up to three times by default.
"""
