"""Static Python analysis with the standard-library ``ast`` module.

The analyzer only *parses* source text; it never imports, compiles to code
objects, or executes anything from the analysed file.

Conventions
-----------
* ``qualified_name``: ``<module>.<Class>.<method>``; entities nested inside a
  function use ``<locals>`` like ``__qualname__`` (``mod.outer.<locals>.inner``).
* Module name comes from the project-relative source name
  (``pkg/a.py`` -> ``pkg.a``; ``pkg/__init__.py`` -> ``pkg``).
* ``start_line`` of a decorated entity is its first decorator line.
* ``is_public`` is False when the name starts with an underscore (this includes
  dunder methods such as ``__init__``), when the entity sits inside a function,
  or when any enclosing class/module is private.
* Default values and annotations are the exact source text.
* Entities defined inside ``if``/``try``/``with``/``for``/``while``/``match``
  blocks are found and attributed to the enclosing module/class/function.
"""

from __future__ import annotations

import ast
import io
import tokenize
import warnings
from dataclasses import dataclass, field

from trustdoc.exceptions import ParsingError
from trustdoc.models import CodeEntity, EntityType, Parameter, ParameterKind
from trustdoc.parsing.textutil import normalize_newlines, split_lines

UNASSIGNED = 0  # placeholder id; the ingestion service stamps real ids on persist

_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEF_NODES = _FUNC_NODES + (ast.ClassDef,)


@dataclass
class ExtractedEntity:
    """A CodeEntity plus the structure that cannot be expressed before persisting."""

    entity: CodeEntity
    parent_index: int | None          # index into PythonAnalysis.entities
    signature: str | None = None      # rendered header, e.g. "def f(a, *, b=1) -> int"
    in_function: bool = False         # defined inside a function body


@dataclass
class PythonAnalysis:
    module_name: str
    encoding: str
    lines: list[str]                  # source lines, 1-based line n is lines[n-1]
    entities: list[ExtractedEntity] = field(default_factory=list)  # [0] is the module


def module_name_from_source_name(source_name: str) -> str:
    stem = source_name[:-3] if source_name.lower().endswith(".py") else source_name
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] == "__init__" and len(parts) > 1:
        parts.pop()
    return ".".join(parts) or "module"


def decode_python_source(data: bytes, source_name: str = "<source>") -> tuple[str, str]:
    """Decode honouring PEP 263 / BOM. Returns (normalised_text, encoding)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        text = data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise ParsingError(f"{source_name}: cannot decode Python source ({exc})") from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    return normalize_newlines(text), encoding


def analyze_python(data: bytes, source_name: str) -> PythonAnalysis:
    """Parse ``data`` and extract every module/class/function/method.

    Raises ``ParsingError`` for undecodable or syntactically invalid source.
    """
    text, encoding = decode_python_source(data, source_name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. invalid escape sequences
            tree = ast.parse(text, filename=source_name)
    except SyntaxError as exc:
        where = f" line {exc.lineno}" if exc.lineno else ""
        raise ParsingError(f"{source_name}:{where.strip()} invalid Python syntax: {exc.msg}") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        raise ParsingError(f"{source_name}: cannot parse Python source ({exc})") from exc

    lines = split_lines(text)
    return _Extractor(text, lines, encoding, module_name_from_source_name(source_name)).run(tree)


class _Extractor:
    def __init__(self, text: str, lines: list[str], encoding: str, module_name: str) -> None:
        self._lines = lines
        self._blines = [ln.encode("utf-8") for ln in text.split("\n")]
        self._encoding = encoding
        self._module = module_name
        self._out: list[ExtractedEntity] = []

    # ------------------------------------------------------------------ run
    def run(self, tree: ast.Module) -> PythonAnalysis:
        parts = self._module.split(".")
        module = CodeEntity(
            project_id=UNASSIGNED, source_id=UNASSIGNED, entity_type=EntityType.MODULE,
            name=parts[-1], qualified_name=self._module, docstring=ast.get_docstring(tree),
            is_public=not any(p.startswith("_") for p in parts),
            start_line=1, end_line=max(1, len(self._lines)),
        )
        self._out.append(ExtractedEntity(module, None))
        self._visit(tree.body, 0, "module", False)
        return PythonAnalysis(self._module, self._encoding, self._lines, self._out)

    def _visit(self, body: list[ast.stmt], parent_index: int, scope: str, in_function: bool) -> None:
        parent = self._out[parent_index].entity
        for node in _scope_defs(body):
            is_class = isinstance(node, ast.ClassDef)
            if is_class:
                etype = EntityType.CLASS
            else:
                etype = EntityType.METHOD if scope == "class" else EntityType.FUNCTION
            sep = ".<locals>." if scope == "function" else "."
            public = (
                parent.is_public and not in_function and not node.name.startswith("_")
            )
            decorators = [self._seg(d) for d in node.decorator_list]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            entity = CodeEntity(
                project_id=UNASSIGNED, source_id=UNASSIGNED, entity_type=etype, name=node.name,
                qualified_name=f"{parent.qualified_name}{sep}{node.name}",
                parameters=[] if is_class else self._parameters(node.args),
                return_annotation=None if is_class or node.returns is None else self._seg(node.returns),
                docstring=ast.get_docstring(node), decorators=decorators,
                is_async=isinstance(node, ast.AsyncFunctionDef), is_public=public,
                start_line=start, end_line=node.end_lineno,
            )
            self._out.append(ExtractedEntity(
                entity, parent_index, self._signature(node), in_function=in_function,
            ))
            index = len(self._out) - 1
            self._visit(node.body, index, "class" if is_class else "function",
                        in_function or not is_class)

    # -------------------------------------------------------------- details
    def _parameters(self, args: ast.arguments) -> list[Parameter]:
        params: list[Parameter] = []
        positional = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for i, (arg, default) in enumerate(zip(positional, defaults)):
            kind = (ParameterKind.POSITIONAL_ONLY if i < len(args.posonlyargs)
                    else ParameterKind.POSITIONAL_OR_KEYWORD)
            params.append(self._param(arg, kind, default))
        if args.vararg:
            params.append(self._param(args.vararg, ParameterKind.VAR_POSITIONAL, None))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            params.append(self._param(arg, ParameterKind.KEYWORD_ONLY, default))
        if args.kwarg:
            params.append(self._param(args.kwarg, ParameterKind.VAR_KEYWORD, None))
        return params

    def _param(self, arg: ast.arg, kind: ParameterKind, default: ast.expr | None) -> Parameter:
        return Parameter(
            name=arg.arg, kind=kind,
            annotation=self._seg(arg.annotation) if arg.annotation is not None else None,
            default=self._seg(default) if default is not None else None,
        )

    def _signature(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            bases = [self._seg(b) for b in node.bases] + [
                f"{k.arg}={self._seg(k.value)}" if k.arg else f"**{self._seg(k.value)}"
                for k in node.keywords
            ]
            return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = f" -> {self._seg(node.returns)}" if node.returns is not None else ""
        return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"

    def _seg(self, node: ast.AST) -> str:
        """Exact source text of ``node`` (AST columns are UTF-8 byte offsets)."""
        try:
            l0, l1 = node.lineno - 1, node.end_lineno - 1
            c0, c1 = node.col_offset, node.end_col_offset
            if l0 == l1:
                return self._blines[l0][c0:c1].decode("utf-8")
            parts = [self._blines[l0][c0:], *self._blines[l0 + 1 : l1], self._blines[l1][:c1]]
            return b"\n".join(parts).decode("utf-8")
        except (AttributeError, IndexError, TypeError, UnicodeDecodeError):
            return ast.unparse(node)


def _scope_defs(stmts: list[ast.stmt]):
    """Yield class/function definitions belonging to this scope, in source order.

    Looks through control-flow statements (if/try/with/for/while/match) but does
    not descend into the bodies of the definitions themselves.
    """
    for stmt in stmts:
        if isinstance(stmt, _DEF_NODES):
            yield stmt
            continue
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                yield from _scope_defs([child])
            elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                yield from _scope_defs(child.body)
