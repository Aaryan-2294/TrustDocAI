"""Conservative comparison of documented types against source annotations.

Pure functions, no I/O. The comparison is deliberately cautious: it answers MISMATCH only when both
sides are clearly different well-known types, and UNKNOWN whenever an alias, a user-defined name, or a
difference in generic specificity makes a verdict unsafe.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from enum import Enum

# typing aliases -> canonical builtin spelling
_TYPING_ALIASES = {
    "List": "list", "Dict": "dict", "Set": "set", "FrozenSet": "frozenset", "Tuple": "tuple",
    "Type": "type", "Text": "str", "NoneType": "None",
}
_BUILTIN_TYPES = frozenset({
    "int", "float", "str", "bool", "list", "dict", "set", "frozenset", "tuple", "bytes", "bytearray",
    "complex", "None", "object", "type",
})
_PLAIN_WORDS = {  # English words used in prose -> canonical type
    "float": "float", "floating-point number": "float", "floating point number": "float",
    "int": "int", "integer": "int", "integers": "int", "string": "str", "strings": "str", "str": "str",
    "boolean": "bool", "booleans": "bool", "bool": "bool", "list": "list", "lists": "list",
    "dictionary": "dict", "dictionaries": "dict", "dict": "dict", "dicts": "dict", "tuple": "tuple",
    "tuples": "tuple", "set": "set", "sets": "set", "bytes": "bytes", "nothing": "None", "none": "None",
}


class TypeMatch(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizedType:
    canonical: str            # e.g. "list[int]", "int | None"
    base: str | None          # for a non-union type: its (last-component) base name, else None
    parametrised: bool
    members: tuple[str, ...]  # canonical union members (a single member for non-unions)


def word_to_type(phrase: str) -> str | None:
    """Map an English phrase from prose ("a float", "nothing") to a canonical type name."""
    return _PLAIN_WORDS.get(re.sub(r"\s+", " ", phrase.strip().lower()))


def normalize_type(text: str | None) -> NormalizedType | None:
    """Parse a type expression into a canonical form, or None if it is not a recognisable type."""
    if text is None:
        return None
    src = text.strip().strip("`").strip()
    if not src or len(src) > 200:
        return None
    if (src[0] == src[-1]) and src[0] in "\"'" and len(src) > 2:
        src = src[1:-1].strip()
    try:
        tree = ast.parse(src, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    try:
        members = _members(tree)
    except _NotAType:
        return None
    members = tuple(sorted(dict.fromkeys(members)))
    canonical = " | ".join(members)
    if len(members) == 1:
        return NormalizedType(canonical, _base(members[0]), "[" in members[0], members)
    return NormalizedType(canonical, None, True, members)


class _NotAType(Exception):
    pass


def _members(node: ast.AST) -> list[str]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _members(node.left) + _members(node.right)
    if isinstance(node, ast.Subscript):
        head = _dotted(node.value)
        name = _last(head)
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name == "Optional" and len(args) == 1:
            return _members(args[0]) + ["None"]
        if name == "Union":
            out: list[str] = []
            for a in args:
                out += _members(a)
            return out
        inner = ", ".join(" | ".join(sorted(dict.fromkeys(_members(a)))) if not isinstance(a, ast.Constant) or a.value is not Ellipsis else "..." for a in args)
        return [f"{_alias(name)}[{inner}]"]
    if isinstance(node, ast.Constant):
        if node.value is None:
            return ["None"]
        if node.value is Ellipsis:
            return ["..."]
        if isinstance(node.value, str):          # forward reference such as "Task"
            inner = normalize_type(node.value)
            if inner is None:
                raise _NotAType
            return list(inner.members)
        raise _NotAType
    if isinstance(node, (ast.Name, ast.Attribute)):
        return [_alias(_last(_dotted(node)))]
    raise _NotAType


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    raise _NotAType


def _last(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _alias(name: str) -> str:
    return _TYPING_ALIASES.get(name, name)


def _base(member: str) -> str:
    return member.split("[", 1)[0]


def compare_types(documented: str | None, actual: str | None) -> TypeMatch:
    """Compare a documented type with a source annotation."""
    doc, act = normalize_type(documented), normalize_type(actual)
    if doc is None or act is None:
        return TypeMatch.UNKNOWN
    if doc.canonical == act.canonical:
        return TypeMatch.MATCH
    if len(doc.members) == 1 and len(act.members) == 1:
        d, a = doc.members[0], act.members[0]
        if _base(d) == _base(a):
            # same base: a bare documented base type is compatible with a parametrised annotation;
            # the reverse (docs more specific than the annotation) cannot be confirmed.
            return TypeMatch.MATCH if not doc.parametrised else (
                TypeMatch.UNKNOWN if not act.parametrised else TypeMatch.MISMATCH
            )
        if _base(d) in _BUILTIN_TYPES and _base(a) in _BUILTIN_TYPES:
            if {_base(d), _base(a)} == {"int", "bool"}:      # bool is an int: do not call it wrong
                return TypeMatch.UNKNOWN
            return TypeMatch.MISMATCH
        return TypeMatch.UNKNOWN                              # user-defined names / aliases: undecidable here
    # unions: a documented member of the annotated union, or vice versa, cannot be decided
    if set(doc.members) & set(act.members):
        return TypeMatch.UNKNOWN
    if all(_base(m) in _BUILTIN_TYPES for m in doc.members + act.members):
        return TypeMatch.MISMATCH
    return TypeMatch.UNKNOWN


def is_user_type_name(name: str) -> bool:
    return name not in _BUILTIN_TYPES and not hasattr(builtins, name)
