"""Resolve documented names to code entities WITHOUT inventing contradictions.

A name that cannot be found is only reported as nonexistent when that is trustworthy. The resolver
therefore distinguishes:

* FOUND      one entity (overload/setter/conditional redefinitions are collapsed)
* AMBIGUOUS  several distinct entities share the name
* ALIAS      not defined here, but bound by an import, an assignment or ``__all__`` (re-export, constant, ...)
* EXTERNAL   builtin / standard-library / third-party-looking name; outside this project
* NOT_FOUND  nothing matches (``strong`` when the documented name is rooted in a project package)
"""

from __future__ import annotations

import ast
import builtins
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from trustdoc.models import CodeEntity, EntityType, Parameter, ParameterKind

_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_BOUND_FIRST = frozenset({"self", "cls", "mcs", "metacls"})
_SECONDARY_DECORATORS = ("overload", ".setter", ".getter", ".deleter")
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(sys.builtin_module_names)
_BUILTINS = frozenset(dir(builtins))


class ResolutionKind(str, Enum):
    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    ALIAS = "alias"
    EXTERNAL = "external"
    NOT_FOUND = "not_found"
    INVALID = "invalid"


@dataclass(frozen=True)
class Resolution:
    kind: ResolutionKind
    name: str
    entity: CodeEntity | None = None
    candidates: tuple[CodeEntity, ...] = ()
    strong: bool = False            # NOT_FOUND only: the name is rooted in a project package
    reason: str = ""


@dataclass
class AliasIndex:
    """Names that exist in the project without being a def/class: imports, assignments, ``__all__``."""

    names: set[str] = field(default_factory=set)

    @classmethod
    def from_sources(cls, texts: Iterable[str]) -> "AliasIndex":
        names: set[str] = set()
        for text in texts:
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError, RecursionError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update((a.asname or a.name.split(".")[0]) for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.update((a.asname or a.name) for a in node.names if a.name != "*")
            for node in tree.body:                      # module-level assignments only
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
                for t in targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                        if t.id == "__all__" and isinstance(getattr(node, "value", None), (ast.List, ast.Tuple)):
                            names.update(e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
        return cls(names)


def clean_name(raw: str) -> str:
    name = (raw or "").strip().strip("`").strip().rstrip(".")
    name = re.sub(r"\(.*\)\s*$", "", name).strip()
    for prefix in ("self.", "cls."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip(".")


def is_valid_name(name: str) -> bool:
    return bool(_DOTTED.match(name))


class EntityResolver:
    def __init__(self, entities: Sequence[CodeEntity], aliases: AliasIndex | None = None) -> None:
        self.entities = list(entities)
        self.aliases = aliases or AliasIndex()
        self._by_suffix: dict[str, list[CodeEntity]] = {}
        self._by_qname: dict[str, list[CodeEntity]] = {}
        self.project_roots: set[str] = set()
        for e in self.entities:
            parts = e.qualified_name.split(".")
            if e.entity_type == EntityType.MODULE:
                self.project_roots.add(parts[0])
            self._by_qname.setdefault(e.qualified_name, []).append(e)
            for i in range(len(parts)):
                self._by_suffix.setdefault(".".join(parts[i:]), []).append(e)

    # ------------------------------------------------------------------ resolve
    def resolve(self, raw: str, *, kind_word: str | None = None) -> Resolution:
        name = clean_name(raw)
        if not is_valid_name(name):
            return Resolution(ResolutionKind.INVALID, name, reason="not a Python identifier path")
        found = list(self._by_suffix.get(name, ()))
        if found:
            found = self._filter_kind(found, kind_word)
            groups: dict[str, list[CodeEntity]] = {}
            for e in found:
                groups.setdefault(e.qualified_name, []).append(e)
            primaries = [self._collapse(g) for g in groups.values()]
            primaries = [p for p in primaries if p is not None]
            if len(primaries) == 1:
                return Resolution(ResolutionKind.FOUND, name, primaries[0], tuple(primaries))
            if primaries:
                return Resolution(ResolutionKind.AMBIGUOUS, name, None, tuple(primaries),
                                  reason=f"{len(primaries)} entities match: " + ", ".join(p.qualified_name for p in primaries[:5]))
            return Resolution(ResolutionKind.AMBIGUOUS, name, None, tuple(found),
                              reason="overloaded or conditionally redefined with differing signatures")

        parts = name.split(".")
        root, simple = parts[0], parts[-1]
        if len(parts) == 1 and (simple in self.aliases.names):
            return Resolution(ResolutionKind.ALIAS, name, reason=f"{simple!r} is bound by an import, assignment or __all__ rather than defined as a function/class")
        if len(parts) > 1 and root in self.project_roots:
            return Resolution(ResolutionKind.NOT_FOUND, name, strong=True,
                              reason=f"no entity matches {name!r} although {root!r} is a project package")
        if root in _BUILTINS or root in _STDLIB:
            return Resolution(ResolutionKind.EXTERNAL, name, reason="builtin or standard-library name")
        if len(parts) > 1 and root[:1].islower() and root not in self.project_roots:
            return Resolution(ResolutionKind.EXTERNAL, name, reason=f"{root!r} is not a project package (external library?)")
        if simple in self.aliases.names:
            return Resolution(ResolutionKind.ALIAS, name, reason=f"{simple!r} is bound by an import or assignment")
        return Resolution(ResolutionKind.NOT_FOUND, name, strong=False, reason="no matching entity in the indexed Python sources")

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _filter_kind(cands: list[CodeEntity], kind_word: str | None) -> list[CodeEntity]:
        wanted = {
            "class": {EntityType.CLASS}, "function": {EntityType.FUNCTION, EntityType.METHOD},
            "method": {EntityType.METHOD, EntityType.FUNCTION}, "constructor": {EntityType.CLASS},
            "module": {EntityType.MODULE},
        }.get((kind_word or "").lower())
        if wanted:
            narrowed = [c for c in cands if c.entity_type in wanted]
            if narrowed:
                return narrowed
        non_module = [c for c in cands if c.entity_type != EntityType.MODULE]
        return non_module or cands

    @staticmethod
    def _collapse(group: list[CodeEntity]) -> CodeEntity | None:
        primary = [e for e in group if not any(d == "overload" or d.endswith(".overload") or d.endswith(_SECONDARY_DECORATORS[1:])
                                               for d in e.decorators)]
        group = primary or group
        first = group[0]
        sig = lambda e: [(p.name, p.kind, p.default, p.annotation) for p in e.parameters]  # noqa: E731
        return first if all(sig(e) == sig(first) and e.return_annotation == first.return_annotation for e in group) else None

    def init_of(self, cls: CodeEntity) -> CodeEntity | None:
        found = self._by_qname.get(f"{cls.qualified_name}.__init__", [])
        return found[0] if found else None

    def effective_parameters(self, entity: CodeEntity) -> list[Parameter] | None:
        """Parameters a caller supplies (bound ``self``/``cls`` removed). None if not determinable."""
        target = entity
        if entity.entity_type == EntityType.CLASS:
            target = self.init_of(entity)
            if target is None:
                return None
        params = list(target.parameters)
        is_static = any(d == "staticmethod" or d.endswith(".staticmethod") for d in target.decorators)
        if target.entity_type == EntityType.METHOD and not is_static and params and params[0].name in _BOUND_FIRST:
            params = params[1:]
        return params

    def known_names(self) -> list[str]:
        return sorted({e.name for e in self.entities if e.entity_type != EntityType.MODULE})
