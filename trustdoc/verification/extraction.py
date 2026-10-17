"""Claim extraction from Markdown documentation and Python docstrings.

Pure functions (no database, no LLM). Deterministic patterns are used wherever a statement has a
checkable structure (signatures parsed with ``ast``, "`f` accepts `a` and `b`", "`f` returns a float",
docstring Args/Returns sections). Natural-language behaviour claims use a CONSERVATIVE rule-based
filter: only declarative sentences with a subject and a behaviour verb are kept; imperatives,
questions, conditionals, hedged statements ("may", "should"...) and second-person instructions are
ignored. When extraction is unsure it produces no claim rather than a fake one.

Each claim records its extraction method and a confidence (1.0 for structure parsed from code-like
syntax, lower for call-like forms and prose heuristics) in ``claim.extracted``.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import re
from typing import Any, Iterable, Mapping, Sequence

from trustdoc.models import Claim, ClaimOrigin, ClaimType, CodeEntity, EntityType
from trustdoc.parsing.markdown_parser import parse_markdown
from trustdoc.parsing.textutil import normalize_newlines, split_lines
from trustdoc.verification.typenorm import normalize_type, word_to_type
from trustdoc.verification.types import CONFIDENCE, METHOD, UID, UNASSIGNED, ExtractionResult, make_claim_uid

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DOTTED = rf"{_IDENT}(?:\.{_IDENT})*"
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_ENTITY_SPAN = re.compile(rf"^(?P<name>{_DOTTED})(?P<parens>\(\s*\))?$")
_SIG_SPAN = re.compile(
    rf"^(?:(?P<kw>async\s+def|def|class)\s+)?(?P<name>{_DOTTED})\s*\((?P<args>.*)\)\s*(?:->\s*(?P<ret>[^:]+?))?\s*:?$"
)
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_KIND = r"(?:function|method|class|constructor|callable|coroutine|decorator|module)"
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+.-]*)")
_LIST_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
_PARAM_HEADING = re.compile(r"^(?:parameters?|arguments?|args|params?|options)$", re.I)
_RETURN_HEADING = re.compile(r"^returns?$", re.I)
_LABEL_HEADING = re.compile(r"^(?:parameters?|arguments?|args|params?|options|returns?|raises?|examples?|notes?|usage)$", re.I)
_BUILTIN_NAMES = frozenset(dir(builtins)) | frozenset(keyword.kwlist)

_PARAM_SENT = re.compile(
    rf"^\s*(?:(?:the|a|an|this)\s+)?(?:(?P<k1>{_KIND})\s+)?`(?P<ent>[^`\n]+)`(?:\s+(?P<k2>{_KIND}))?\s+"
    rf"(?P<verb>accepts?|takes?|expects?|requires?)\s+(?:(?P<only>only|exactly)\s+)?(?:(?:the|these|those)\s+)?"
    rf"(?:(?:following|keyword|positional|optional|required|named)\s+)*(?:(?P<pw>parameters?|arguments?|args?|kwargs|params?)\s*:?\s+)?"
    rf"(?P<names>`[^`\n]+`(?:\s*(?:,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+|\s*&\s*)`[^`\n]+`)*)(?:\s+(?P<pw2>parameters?|arguments?|args?|params?)\b)?",
    re.I,
)
_PARAM_CTX = re.compile(
    r"^\s*(?P<verb>accepts?|takes?|expects?)\s+(?:(?P<only>only|exactly)\s+)?(?:(?:the|following)\s+)*"
    r"(?:(?:parameters?|arguments?|args?)\s*:?\s+)?(?P<names>`[^`\n]+`(?:\s*(?:,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+|\s*&\s*)`[^`\n]+`)*)",
    re.I,
)
_SUBJ = re.compile(rf"^\s*(?:(?:the|a|an|this)\s+)?(?:(?P<k1>{_KIND})\s+)?`(?P<ent>[^`\n]+)`(?:\s+(?P<k2>{_KIND}))?\b", re.I)
_TYPE_WORDS = (r"floating[- ]point\s+number|float|integers?|int|strings?|str|booleans?|bool|lists?|"
               r"dictionar(?:y|ies)|dicts?|tuples?|sets?|bytes|nothing|none")
_RET = re.compile(rf"\b(?:returns?|yields?|evaluates\s+to)\s+(?:(?:a|an|the)\s+)?(?:(?P<code>`[^`\n]+`)|(?P<word>{_TYPE_WORDS})\b)", re.I)
_RET_START = re.compile(rf"^\s*(?:returns?|yields?)\s+(?:(?:a|an|the)\s+)?(?:(?P<code>`[^`\n]+`)|(?P<word>{_TYPE_WORDS})\b)", re.I)
_RET_SECTION = re.compile(rf"^\s*(?:(?:a|an|the)\s+)?(?:(?P<code>`[^`\n]+`)|(?P<word>{_TYPE_WORDS})\b)", re.I)
_PARAM_ITEM = re.compile(rf"^[`*_]*(?P<star>\*{{0,2}})(?P<name>{_IDENT})[`*_]*\s*(?:\((?P<type>[^)]*)\))?\s*(?:[:\-\u2013\u2014|]|$)")

_IMPERATIVE_STARTERS = frozenset("""install run use call create add set make open click navigate see note check ensure import pass
provide start stop enable disable configure define write read build clone download copy edit update remove delete replace choose
select try avoid remember consider refer follow visit go type enter save load restart deploy contribute submit report test verify
then first next finally if when for to in on with without before after once unless while because although however also please
let take put keep wrap register inspect compare apply tune customize override extend subclass implement""".split())
_HEDGES = re.compile(r"\b(?:may|might|could|should|would|can|cannot|optionally|typically|usually|generally|often|sometimes|ideally|recommend\w*|possibly|perhaps)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(?:you|your|you're|we|our)\b", re.I)
_VERBS = (
    "persist|store|save|write|read|load|retry|retries|raise|throw|validate|check|cache|log|send|receive|delete|remove|create|generate|"
    "start|stop|close|open|connect|use|support|require|ignore|skip|block|wait|expire|encrypt|compress|serialize|schedule|run|execute|"
    "process|handle|notify|emit|reject|default|override|retain|rotate|poll|dispatch|prioriti[sz]e|hide|requeue|drop|evict|reclaim|compact|"
    "serve|order|sort|count|limit|cap|record|track|report|fail|succeed|restart|resume|pause|buffer|flush|replace|update|insert|select|fall"
)
_ACTIVE = re.compile(rf"\b(?:{_VERBS})(?:s|es|ed|ies|ing)?\b", re.I)
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|gets?|be|been)\s+(?:also\s+|automatically\s+|then\s+|first\s+)?"
    r"(?:persisted|stored|saved|written|read|loaded|retried|cached|logged|deleted|removed|validated|rejected|sent|capped|limited|"
    r"serialized|compressed|encrypted|scheduled|executed|processed|handled|requeued|dropped|evicted|ignored|skipped|created|generated|"
    r"opened|closed|hidden|ordered|sorted|recorded|tracked|expired|reclaimed|compacted|removed|restarted|returned|raised|set|used|kept|retained)\b",
    re.I,
)
_CONFIG_PHRASE = re.compile(r"\b(?:environment variables?|env(?:ironment)? vars?|config(?:uration)?\s+(?:file|key|option|setting|value))s?\b", re.I)
_CONFIG_VERB = re.compile(r"\b(?:sets?|controls?|overrides?|override|configures?|selects?|reads?|enables?|disables?|determines?|specif(?:y|ies)|defines?|uses?|loads?)\b", re.I)
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|cf|resp)\.", re.I)


# ----------------------------------------------------------------------------- small helpers
def looks_like_code_name(token: str) -> bool:
    """True for tokens that are clearly code (snake_case, CamelCase, dotted, calls), not plain words."""
    t = token.strip().strip("`")
    return bool(t) and ("_" in t or "." in t or t.endswith("()") or bool(re.search(r"[a-z][A-Z]", t)))


def entity_from_span(span: str) -> tuple[str, bool] | None:
    """(name, has_empty_parens) if a code span is a plausible entity reference, else None."""
    m = _ENTITY_SPAN.match(span.strip())
    if not m:
        return None
    name, parens = m.group("name"), bool(m.group("parens"))
    if "." not in name and name in _BUILTIN_NAMES:
        return None
    if _ALLCAPS.match(name) and not parens:
        return None
    return name, parens


def parse_def_args(args_src: str) -> list[dict[str, Any]] | None:
    """Parse the text between parentheses as a def parameter list; None if it is not one."""
    try:
        tree = ast.parse(f"def _f({args_src}):\n    pass")
    except (SyntaxError, ValueError):
        return None
    a = tree.body[0].args
    out: list[dict[str, Any]] = []
    positional = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        out.append(_param(arg, "positional_only" if i < len(a.posonlyargs) else "positional_or_keyword", default))
    if a.vararg:
        out.append(_param(a.vararg, "var_positional", None))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        out.append(_param(arg, "keyword_only", default))
    if a.kwarg:
        out.append(_param(a.kwarg, "var_keyword", None))
    return out


def _param(arg: ast.arg, kind: str, default: ast.expr | None) -> dict[str, Any]:
    return {"name": arg.arg, "kind": kind,
            "annotation": ast.unparse(arg.annotation) if arg.annotation is not None else None,
            "default": ast.unparse(default) if default is not None else None}


def _parse_param_tokens(spans: Sequence[str]) -> list[dict[str, Any]] | None:
    """Documented names like `a`, `*args`, `b=2`, `c: int` -> param dicts; None if any is not a parameter."""
    params = []
    for span in spans:
        m = re.match(rf"^(?P<star>\*{{0,2}})(?P<name>{_IDENT})\s*(?::\s*(?P<ann>[^=]+?))?\s*(?:=\s*(?P<default>.+))?$", span.strip())
        if not m or m.group("name") in _BUILTIN_NAMES - {"id", "type", "input", "format", "filter", "max", "min", "sum", "map", "object", "hash", "dir"}:
            return None
        kind = {"*": "var_positional", "**": "var_keyword"}.get(m.group("star"), "unknown")
        params.append({"name": m.group("name"), "kind": kind, "annotation": (m.group("ann") or "").strip() or None,
                       "default": (m.group("default") or "").strip() or None})
    return params


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Split prose into (offset, sentence) without splitting inside `code` or after abbreviations."""
    masked = _CODE_SPAN.sub(lambda m: "\x00" * len(m.group(0)), text)
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x01"), masked)
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z`\[(\"'])", masked):
        out.append((start, text[start:m.start()]))
        start = m.end()
    out.append((start, text[start:]))
    return [(o, s.strip()) for o, s in out if s.strip()]


def _heading_entity(text: str) -> tuple[str, str | None] | None:
    """(entity name, signature-span or None) when a heading names a code entity."""
    spans = _CODE_SPAN.findall(text)
    raw = spans[0] if spans else text.strip()
    sig = _SIG_SPAN.match(raw.strip())
    if sig and parse_def_args(sig.group("args")) is not None and (spans or looks_like_code_name(sig.group("name"))):
        return sig.group("name"), raw.strip()
    e = entity_from_span(raw)
    if e and (spans or looks_like_code_name(e[0]) or e[1]):
        return e[0], None
    return None


def _kind_word_near(sentence: str, start: int, end: int) -> str | None:
    before = re.search(rf"\b(?P<k>{_KIND})\s+$", sentence[:start], re.I)
    after = re.match(rf"^\s+(?P<k>{_KIND})\b", sentence[end:], re.I)
    word = (before or after)
    return word.group("k").lower() if word else None


# -------------------------------------------------------------------------- claim construction
class _Builder:
    def __init__(self, source_name: str, origin: ClaimOrigin) -> None:
        self.source_name, self.origin = source_name, origin
        self.result = ExtractionResult()
        self._entity_seen: set[str] = set()
        self._occurrence: dict[tuple[str, str], int] = {}

    def claim(self, category: ClaimType, text: str, line: int | None, *, key: str = "", subject: str | None = None,
              method: str, confidence: float, heading: str | None = None, label: str | None = None, **extra: Any) -> Claim:
        uid_text = f"{text}\x1e{key}"
        occ = self._occurrence.get((category.value, uid_text), 0)
        self._occurrence[(category.value, uid_text)] = occ + 1
        extracted = {
            UID: make_claim_uid(self.source_name, category, uid_text, occ), METHOD: method, CONFIDENCE: confidence,
            "doc_source_name": self.source_name, "doc_label": label or (f"{self.source_name}:{line}" if line else self.source_name),
            "heading": heading, **extra,
        }
        c = Claim(project_id=UNASSIGNED, source_id=UNASSIGNED, claim_type=category, text=text.strip(), origin=self.origin,
                  subject_name=subject, start_line=line, extracted=extracted)
        self.result.claims.append(c)
        return c

    def entity_claim(self, name: str, text: str, line: int, *, kind: str | None, heading: str | None, via: str) -> None:
        if name.lower() in self._entity_seen:
            return
        self._entity_seen.add(name.lower())
        self.claim(ClaimType.ENTITY_REFERENCE, text, line, key=name, subject=name, method=f"rule:{via}", confidence=0.9,
                   heading=heading, entity_name=name, kind_word=kind, callable_hint=True, context="prose")


# ------------------------------------------------------------------------------- Markdown
def extract_markdown_claims(text: str, source_name: str, *, origin: ClaimOrigin = ClaimOrigin.EXISTING_DOC) -> ExtractionResult:
    """Extract verifiable claims from one Markdown document. Never raises on odd input."""
    b = _Builder(source_name, origin)
    norm = normalize_newlines(text or "")
    lines = split_lines(norm) if norm.strip() else []
    if not lines:
        b.result.warnings.append(f"{source_name}: document is empty; no claims extracted")
        return b.result
    try:
        headings = {h["line"]: h for h in parse_markdown(norm.encode("utf-8"), source_name).metadata["headings"]}
    except Exception:  # noqa: BLE001 - extraction must degrade, never crash verification
        headings = {}
    _MarkdownWalker(b, lines, headings).run()
    if not b.result.claims:
        b.result.warnings.append(f"{source_name}: no verifiable claims were extracted")
    _assert_unique_uids(b.result.claims)
    return b.result


class _MarkdownWalker:
    def __init__(self, b: _Builder, lines: list[str], headings: dict[int, dict]) -> None:
        self.b, self.lines, self.headings = b, lines, headings
        self.stack: list[tuple[int, str, str | None]] = []          # (level, text, entity)
        self.param_section = self.return_section = False
        self.pending_params: list[dict[str, Any]] = []
        self.pending_line: int | None = None

    # -- context --------------------------------------------------------------------------
    @property
    def heading_text(self) -> str | None:
        return self.stack[-1][1] if self.stack else None

    @property
    def ctx_entity(self) -> str | None:
        for level, text, ent in reversed(self.stack):
            if ent:
                return ent
            if not _LABEL_HEADING.match(text.strip("` *")):
                return None
        return None

    def run(self) -> None:
        lines, n, i = self.lines, len(self.lines), 0
        fm_end = 0
        if lines[0].strip() == "---":
            for k in range(1, n):
                if lines[k].strip() in ("---", "..."):
                    fm_end = k + 1
                    break
        i = fm_end
        para: list[tuple[int, str]] = []
        while i < n:
            line, no = lines[i], i + 1
            fence = _FENCE.match(line)
            if fence:
                self._flush(para); para = []
                close = next((j for j in range(i + 1, n) if re.match(rf"^\s{{0,3}}{re.escape(fence.group(1)[0])}{{{len(fence.group(1))},}}\s*$", lines[j])), None)
                end = close if close is not None else n
                self._fenced(fence.group(2).lower(), lines[i + 1:end], i + 2)
                i = end + 1
                continue
            if no in self.headings:
                self._flush(para); para = []
                self._heading(self.headings[no], no)
                if not line.lstrip().startswith("#"):          # Setext: skip the underline
                    i += 1
                i += 1
                continue
            if line.lstrip().startswith("|"):
                self._flush(para); para = []
                j = i
                while j < n and lines[j].lstrip().startswith("|"):
                    j += 1
                self._table(lines[i:j], no)
                i = j
                continue
            item = _LIST_ITEM.match(line)
            if item:
                self._flush(para); para = []
                text, j = item.group(2), i + 1
                while j < n and lines[j].strip() and not _LIST_ITEM.match(lines[j]) and not _FENCE.match(lines[j]) \
                        and (j + 1) not in self.headings and lines[j][:1] in (" ", "\t"):
                    text += " " + lines[j].strip()
                    j += 1
                self._list_item(text, no)
                i = j
                continue
            if not line.strip():
                self._flush(para); para = []
            else:
                para.append((no, line.strip().lstrip(">").strip()))
            i += 1
        self._flush(para)
        self._flush_params()

    # -- blocks -----------------------------------------------------------------------------
    def _heading(self, h: dict, line: int) -> None:
        self._flush_params()
        level, text = h["level"], h["text"]
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        found = _heading_entity(text)
        self.stack.append((level, text, found[0] if found else None))
        plain = text.strip("` *")
        self.param_section = bool(_PARAM_HEADING.match(plain))
        self.return_section = bool(_RETURN_HEADING.match(plain))
        if found:
            self.b.result.mentions.add(found[0])
            if found[1]:
                self._signature(found[1], text, line, in_heading=True)
        # a heading that names an entity documents that entity: record its existence only with a signature/call form

    def _flush(self, para: list[tuple[int, str]]) -> None:
        if not para:
            return
        text, line_of, pos = "", [], 0
        for no, part in para:
            if text:
                text += " "
            line_of.append((len(text), no))
            text += part
        for off, sent in split_sentences(text):
            line = max((no for start, no in line_of if start <= off), default=para[0][0])
            self._sentence(sent, line)

    def _list_item(self, text: str, line: int) -> None:
        if self.param_section and self.ctx_entity:
            m = _PARAM_ITEM.match(text.strip())
            if m and m.group("name").lower() not in ("none", "returns", "return", "raises", "note", "example"):
                self._add_param(m, line)
                return
        for off, sent in split_sentences(text):
            self._sentence(sent, line)

    def _table(self, rows: list[str], line: int) -> None:
        if not (self.param_section and self.ctx_entity):
            for r in rows:                                    # tables are not prose: only collect mentions
                for span in _CODE_SPAN.findall(r):
                    e = entity_from_span(span)
                    if e and looks_like_code_name(e[0]):
                        self.b.result.mentions.add(e[0])
            return
        for k, row in enumerate(rows):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if k == 0 or not cells or re.match(r"^[:\- ]+$", cells[0]):
                continue
            m = _PARAM_ITEM.match(cells[0])
            if m and m.group("name").lower() not in ("parameter", "parameters", "name", "argument"):
                self._add_param(m, line + k)

    def _add_param(self, m: re.Match, line: int) -> None:
        if self.pending_line is None:
            self.pending_line = line
        kind = {"*": "var_positional", "**": "var_keyword"}.get(m.group("star"), "unknown")
        ann = (m.group("type") or "").split(",")[0].strip() or None
        self.pending_params.append({"name": m.group("name"), "kind": kind, "annotation": ann if ann and normalize_type(ann) else None, "default": None})

    def _flush_params(self) -> None:
        if self.pending_params and self.ctx_entity:
            names = ", ".join(p["name"] for p in self.pending_params)
            self.b.claim(ClaimType.PARAMETER, f"`{self.ctx_entity}` parameters section lists: {names}", self.pending_line,
                         key=names, subject=self.ctx_entity, method="rule:param_section", confidence=0.9, heading=self.heading_text,
                         entity_name=self.ctx_entity, documented_params=self.pending_params, complete=False, args_section=True,
                         callable_hint=True, context="param_section", form="section")
        self.pending_params, self.pending_line = [], None

    def _fenced(self, lang: str, block: list[str], start_line: int) -> None:
        if lang not in ("", "python", "py", "python3", "pycon"):
            return
        src = "\n".join(block)
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            return
        for node in tree.body:                                 # top-level definitions only (examples often subclass/override)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = parse_def_args(ast.unparse(node.args))
                if args is None:
                    continue
                ret = ast.unparse(node.returns) if node.returns is not None else None
                self.b.claim(ClaimType.SIGNATURE, f"def {node.name}({ast.unparse(node.args)})" + (f" -> {ret}" if ret else ""),
                             start_line + node.lineno - 1, key=node.name, subject=node.name, method="ast:fenced_def", confidence=0.8,
                             heading=self.heading_text, entity_name=node.name, documented_params=args, documented_return=ret,
                             complete=True, form="def", callable_hint=False, context="fenced_code")
                self.b.result.mentions.add(node.name)

    # -- sentences ----------------------------------------------------------------------------
    def _sentence(self, sent: str, line: int) -> None:
        b, heading = self.b, self.heading_text
        spans = [(m.group(1), m.start(), m.end()) for m in _CODE_SPAN.finditer(sent)]
        for span, start, end in spans:
            e = entity_from_span(span)
            if e and (looks_like_code_name(e[0]) or e[1]):
                b.result.mentions.add(e[0])
        covered: set[str] = set()                              # entities already covered by a structured claim
        for span, start, end in spans:
            if name := self._signature(span, sent, line, start=start, end=end):
                covered.add(name.lower())

        subject_made = False
        pm = _PARAM_SENT.match(sent)
        if pm and not (pm.group("verb").lower().startswith("require") and not (pm.group("pw") or pm.group("pw2"))):
            if name := self._param_claim(sent, line, pm.group("ent"), pm, pm.group("k1") or pm.group("k2")):
                covered.add(name.lower()); subject_made = True
        elif self.ctx_entity and (cm := _PARAM_CTX.match(sent)):
            if name := self._param_claim(sent, line, self.ctx_entity, cm, None, via_heading=True):
                covered.add(name.lower()); subject_made = True

        sm = _SUBJ.match(sent)
        if sm and (rm := _RET.search(sent, sm.end())):
            name = self._return_claim(sent, line, sm.group("ent"), rm, sm.group("k1") or sm.group("k2"))
        elif self.ctx_entity and (rm := _RET_START.match(sent)):
            name = self._return_claim(sent, line, self.ctx_entity, rm, None, via_heading=True)
        elif self.return_section and self.ctx_entity and (rm := _RET_SECTION.match(sent)):
            name = self._return_claim(sent, line, self.ctx_entity, rm, None, via_heading=True)
        else:
            name = None
        if name:
            covered.add(name.lower()); subject_made = True

        for span, start, end in spans:                         # `name()` or "the `X` class" (only if not already covered)
            e = entity_from_span(span)
            if not e or e[0].lower() in covered:
                continue
            kind = _kind_word_near(sent, start, end)
            if (e[1] or kind) and not (kind == "module" and not e[1]):
                b.entity_claim(e[0], sent, line, kind=kind, heading=heading, via="entity_mention")

        if not subject_made:                                   # a structured claim consumed the sentence; otherwise behaviour
            self._behavior(sent, line, spans)

    def _signature(self, span: str, sentence: str, line: int, *, start: int | None = None, end: int | None = None, in_heading: bool = False) -> str | None:
        m = _SIG_SPAN.match(span.strip())
        if not m:
            return None
        name, args_src = m.group("name"), m.group("args")
        if not args_src.strip() and not m.group("kw"):
            return None                                         # `f()` is an entity mention, handled separately
        if "." not in name and name in _BUILTIN_NAMES:
            return None
        args = parse_def_args(args_src)
        if args is None:
            return None
        is_def = bool(m.group("kw")) or bool(m.group("ret"))
        if not is_def:
            for p in args:                                      # call-like: entries without "=" are placeholders
                p["placeholder"] = p["default"] is None and p["kind"] in ("positional_or_keyword", "positional_only")
        ret = (m.group("ret") or "").strip() or None
        self.b.claim(ClaimType.SIGNATURE, span.strip() if in_heading else sentence, line, key=span.strip(), subject=name,
                     method="ast:signature_span" if is_def else "ast:call_like_span", confidence=1.0 if is_def else 0.7,
                     heading=self.heading_text, entity_name=name, documented_params=args, documented_return=ret,
                     complete=is_def, form="def" if is_def else "call_like", callable_hint=True, context="inline_signature",
                     kind_word=(m.group("kw") or "").split()[-1] if m.group("kw") else None)
        self.b.result.mentions.add(name)
        return name

    def _param_claim(self, sent: str, line: int, ent_span: str, m: re.Match, kind: str | None, *, via_heading: bool = False) -> str | None:
        ent = ent_span if via_heading else (entity_from_span(ent_span) or (None,))[0]
        if not ent:
            return None
        params = _parse_param_tokens(_CODE_SPAN.findall(m.group("names")))
        if not params:
            return None
        complete = bool(m.group("only"))
        self.b.claim(ClaimType.PARAMETER, sent, line, key=",".join(p["name"] for p in params), subject=ent,
                     method="rule:param_sentence", confidence=0.9 if not via_heading else 0.7, heading=self.heading_text,
                     entity_name=ent, documented_params=params, complete=complete, args_section=False, kind_word=(kind or "").lower() or None,
                     callable_hint=bool(kind) or _has_parens(ent_span) or "." in ent, context="prose", form="prose",
                     subject_source="heading" if via_heading else "sentence")
        self.b.result.mentions.add(ent)
        return ent

    def _return_claim(self, sent: str, line: int, ent_span: str, m: re.Match, kind: str | None, *, via_heading: bool = False) -> str | None:
        ent = ent_span if via_heading else (entity_from_span(ent_span) or (None,))[0]
        if not ent:
            return None
        raw = (m.group("code") or "").strip("`") if m.group("code") else (word_to_type(m.group("word")) or "")
        if not raw or normalize_type(raw) is None or raw in ("True", "False"):
            return None
        self.b.claim(ClaimType.RETURN, sent, line, key=raw, subject=ent, method="rule:return_sentence", confidence=0.9 if not via_heading else 0.7,
                     heading=self.heading_text, entity_name=ent, documented_return=raw, kind_word=(kind or "").lower() or None,
                     callable_hint=bool(kind) or _has_parens(ent_span) or "." in ent, context="prose", form="prose",
                     subject_source="heading" if via_heading else "sentence")
        self.b.result.mentions.add(ent)
        return ent

    def _behavior(self, sent: str, line: int, spans: list[tuple[str, int, int]]) -> None:
        verdict = behavior_candidate(sent)
        if verdict is None:
            return
        category, verb_pos = verdict
        subject = None
        for span, start, end in spans:
            e = entity_from_span(span)
            if e and start < verb_pos and (looks_like_code_name(e[0]) or e[1]):
                subject = e[0]
                break
        source = "sentence"
        if subject is None and self.ctx_entity and re.match(r"^\s*(?:it|this|they)\b", sent, re.I):
            subject, source = self.ctx_entity, "heading"
        self.b.claim(category, sent, line, key="", subject=subject, method="rule:behavior_sentence", confidence=0.6,
                     heading=self.heading_text, entity_name=subject, context="prose", form="prose", subject_source=source,
                     callable_hint=False)


def _has_parens(span: str) -> bool:
    return span.strip().endswith(")")


def behavior_candidate(sentence: str) -> tuple[ClaimType, int] | None:
    """Conservative test for a checkable declarative behaviour sentence -> (category, verb position)."""
    s = sentence.strip()
    plain = _CODE_SPAN.sub("X", s)
    words = plain.split()
    if not 4 <= len(words) <= 45 or s.endswith("?") or s.endswith(":") or "http" in s.lower():
        return None
    if words[0].strip("*_`\"'(").lower() in _IMPERATIVE_STARTERS:
        return None
    if _HEDGES.search(plain) or _SECOND_PERSON.search(plain):
        return None
    config_context = any(_ALLCAPS.match(sp) for sp in _CODE_SPAN.findall(s)) or bool(_CONFIG_PHRASE.search(s))
    config_verb = _CONFIG_VERB.search(plain) if config_context else None
    m = config_verb or _PASSIVE.search(plain) or _ACTIVE.search(plain)
    if not m or not plain[:m.start()].strip():
        return None                                              # needs a subject before the verb
    category = ClaimType.CONFIG_DEPENDENCY if config_verb else ClaimType.BEHAVIORAL
    # map the verb position back to the original sentence (code spans were collapsed to "X")
    pos = _orig_pos(s, m.start())
    return category, pos


def _orig_pos(original: str, plain_pos: int) -> int:
    pos_plain = pos_orig = 0
    for m in _CODE_SPAN.finditer(original):
        before = m.start() - pos_orig
        if pos_plain + before >= plain_pos:
            break
        pos_plain += before + 1
        pos_orig = m.end()
    return pos_orig + (plain_pos - pos_plain)


# ------------------------------------------------------------------------------ docstrings
_GOOGLE_PARAMS = re.compile(r"^(?:args|arguments|parameters|params|keyword args|keyword arguments|other parameters)\s*:\s*$", re.I)
_GOOGLE_RETURNS = re.compile(r"^(?:returns?|yields?)\s*:\s*$", re.I)
_GOOGLE_ENTRY = re.compile(rf"^(?P<star>\*{{0,2}})(?P<name>{_IDENT})\s*(?:\((?P<type>[^)]*)\))?\s*:")
_NUMPY_ENTRY = re.compile(rf"^(?P<star>\*{{0,2}})(?P<name>{_IDENT})(?:\s*:\s*(?P<type>.+))?$")
_NUMPY_UNDERLINE = re.compile(r"^-{3,}\s*$")
_REST_PARAM = re.compile(rf"^:param\s+(?:(?P<type>[^:]+?)\s+)?(?P<star>\*{{0,2}})(?P<name>{_IDENT})\s*:")
_REST_TYPE = re.compile(rf"^:type\s+(?P<name>{_IDENT})\s*:\s*(?P<type>.+)$")
_REST_RTYPE = re.compile(r"^:rtype\s*:\s*(?P<type>.+)$")


def parse_docstring(doc: str) -> dict[str, Any]:
    """Extract documented parameters and return type from Google, NumPy or reST docstrings."""
    lines = doc.expandtabs().splitlines()
    params: list[dict[str, Any]] = []
    rtype: str | None = None
    has_section = False
    i, n = 0, len(lines)
    indent = lambda s: len(s) - len(s.lstrip())  # noqa: E731
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        if _GOOGLE_PARAMS.match(stripped):                     # --- Google style parameters
            has_section = True
            base, j, entry_indent = indent(raw), i + 1, None
            while j < n and (not lines[j].strip() or indent(lines[j]) > base):
                if lines[j].strip():
                    if entry_indent is None:
                        entry_indent = indent(lines[j])
                    if indent(lines[j]) == entry_indent and (m := _GOOGLE_ENTRY.match(lines[j].strip())):
                        params.append(_doc_param(m))
                j += 1
            i = j
            continue
        if _GOOGLE_RETURNS.match(stripped):                    # --- Google style returns
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and indent(lines[j]) > indent(raw):
                m = re.match(r"^([A-Za-z_][\w.\[\], |]*?)\s*:\s+\S", lines[j].strip())
                if m and normalize_type(m.group(1)):
                    rtype = m.group(1).strip()
            i = j
            continue
        if re.match(r"^(?:parameters?|arguments?)$", stripped, re.I) and _NUMPY_UNDERLINE.match(nxt):   # --- NumPy
            has_section = True
            j = i + 2
            while j < n:
                line = lines[j]
                if line.strip() and indent(line) == indent(raw) and j + 1 < n and _NUMPY_UNDERLINE.match(lines[j + 1].strip()):
                    break
                if line.strip() and indent(line) == indent(raw) and (m := _NUMPY_ENTRY.match(line.strip())):
                    params.append(_doc_param(m))
                j += 1
            i = j
            continue
        if re.match(r"^returns?$", stripped, re.I) and _NUMPY_UNDERLINE.match(nxt):
            j = i + 2
            if j < n and lines[j].strip():
                first = lines[j].strip()
                cand = first.split(":", 1)[-1].strip() if ":" in first else first
                if normalize_type(cand):
                    rtype = cand
            i = j
            continue
        if (m := _REST_PARAM.match(stripped)):                 # --- reST
            has_section = True
            params.append(_doc_param(m))
        elif (m := _REST_TYPE.match(stripped)):
            for p in params:
                if p["name"] == m.group("name") and not p["annotation"] and normalize_type(m.group("type")):
                    p["annotation"] = m.group("type").strip()
        elif (m := _REST_RTYPE.match(stripped)) and normalize_type(m.group("type")):
            rtype = m.group("type").strip()
        i += 1
    seen, unique = set(), []
    for p in params:
        if p["name"] not in seen and p["name"] not in ("self", "cls"):
            seen.add(p["name"])
            unique.append(p)
    return {"params": unique, "return_type": rtype, "has_params_section": has_section}


def _doc_param(m: re.Match) -> dict[str, Any]:
    t = (m.groupdict().get("type") or "").strip()
    kind = {"*": "var_positional", "**": "var_keyword"}.get(m.group("star"), "unknown")
    return {"name": m.group("name"), "kind": kind, "annotation": t if t and normalize_type(t) else None, "default": None}


def extract_docstring_claims(
    entities: Iterable[CodeEntity], source_names: Mapping[int, str], *, origin: ClaimOrigin = ClaimOrigin.EXISTING_DOC
) -> ExtractionResult:
    """Parameter and return-type claims from the docstrings of already-extracted code entities."""
    result = ExtractionResult()
    builders: dict[str, _Builder] = {}
    for e in entities:
        if e.entity_type == EntityType.MODULE or not e.has_docstring:
            continue
        src = source_names.get(e.source_id)
        if src is None:
            continue
        parsed = parse_docstring(e.docstring or "")
        b = builders.setdefault(src, _Builder(src, origin))
        label = f"{src}:{e.start_line} (docstring of {e.qualified_name})"
        common = dict(subject=e.qualified_name, heading=None, label=label, entity_name=e.qualified_name, callable_hint=True,
                      context="docstring", form="docstring", exact_entity=True)
        if parsed["params"]:
            names = ", ".join(p["name"] for p in parsed["params"])
            b.claim(ClaimType.PARAMETER, f"Docstring of `{e.qualified_name}` documents parameters: {names}", e.start_line, key=e.qualified_name,
                    method="docstring:params", confidence=1.0, documented_params=parsed["params"], complete=False,
                    args_section=parsed["has_params_section"], **common)
        if parsed["return_type"]:
            b.claim(ClaimType.RETURN, f"Docstring of `{e.qualified_name}` documents return type `{parsed['return_type']}`", e.start_line,
                    key=e.qualified_name, method="docstring:returns", confidence=1.0, documented_return=parsed["return_type"], **common)
    for b in builders.values():
        result.extend(b.result)
    return result


def _assert_unique_uids(claims: Sequence[Claim]) -> None:
    seen: set[str] = set()
    for c in claims:
        uid = c.extracted[UID]
        if uid in seen:                                           # cannot happen by construction; keep identity safe
            c.extracted[UID] = make_claim_uid(c.extracted["doc_source_name"], c.claim_type, c.text + "\x1e" + uid, len(seen))
        seen.add(c.extracted[UID])
