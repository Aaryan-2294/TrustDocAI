import pytest

from trustdoc.exceptions import ParsingError
from trustdoc.models import EntityType, ParameterKind
from trustdoc.parsing.python_analyzer import (
    analyze_python,
    module_name_from_source_name,
)


def analyze(src: str, name: str = "pkg/mod.py"):
    return analyze_python(src.encode("utf-8"), name)


def by_qname(analysis):
    return {e.entity.qualified_name: e for e in analysis.entities}


def test_module_entity_and_name_derivation():
    a = analyze('"""Module doc."""\nX = 1\n')
    m = a.entities[0].entity
    assert (m.entity_type, m.qualified_name, m.name) == (EntityType.MODULE, "pkg.mod", "mod")
    assert m.docstring == "Module doc." and (m.start_line, m.end_line) == (1, 2)
    assert module_name_from_source_name("pkg/__init__.py") == "pkg"
    assert module_name_from_source_name("a.py") == "a"
    assert module_name_from_source_name("__init__.py") == "__init__"


def test_simple_function_details():
    src = 'def add(a: int, b: int = 2) -> int:\n    """Add."""\n    return a + b\n'
    f = by_qname(analyze(src))["pkg.mod.add"].entity
    assert f.entity_type == EntityType.FUNCTION and f.docstring == "Add."
    assert f.return_annotation == "int"
    assert [(p.name, p.annotation, p.default) for p in f.parameters] == [
        ("a", "int", None), ("b", "int", "2")]
    assert (f.start_line, f.end_line) == (1, 3) and not f.is_async


def test_parameter_kinds_and_defaults_as_source_text():
    src = "def f(a, b=None, /, c=(1, 2), *args, d: 'X' = {'k': 1}, e, **kw): pass\n"
    params = by_qname(analyze(src))["pkg.mod.f"].entity.parameters
    kinds = {p.name: p.kind for p in params}
    assert kinds == {
        "a": ParameterKind.POSITIONAL_ONLY, "b": ParameterKind.POSITIONAL_ONLY,
        "c": ParameterKind.POSITIONAL_OR_KEYWORD, "args": ParameterKind.VAR_POSITIONAL,
        "d": ParameterKind.KEYWORD_ONLY, "e": ParameterKind.KEYWORD_ONLY,
        "kw": ParameterKind.VAR_KEYWORD}
    d = {p.name: p for p in params}
    assert d["b"].default == "None" and d["b"].has_default and not d["a"].has_default
    assert d["c"].default == "(1, 2)" and d["d"].default == "{'k': 1}"
    assert d["d"].annotation == "'X'" and d["e"].default is None
    assert [p.name for p in params] == ["a", "b", "c", "args", "d", "e", "kw"]


def test_non_ascii_source_text_uses_correct_columns():
    f = by_qname(analyze('def f(x="\u00e9\u00e8", y=2): pass\n'))["pkg.mod.f"].entity
    assert [p.default for p in f.parameters] == ['"\u00e9\u00e8"', "2"]


def test_multiline_default_is_exact_source():
    src = "def f(a,\n      b=(1,\n         2)):\n    pass\n"
    f = by_qname(analyze(src))["pkg.mod.f"].entity
    assert f.parameters[1].default == "(1,\n         2)"
    assert (f.start_line, f.end_line) == (1, 4)


def test_classes_methods_and_relationships():
    src = (
        "class Repo(Base):\n"
        '    """A repo."""\n'
        "    def save(self, item): ...\n"
        "    class Meta:\n"
        "        def info(self): ...\n"
        "def helper(): ...\n"
    )
    a = analyze(src)
    q = by_qname(a)
    assert q["pkg.mod.Repo"].entity.entity_type == EntityType.CLASS
    assert q["pkg.mod.Repo.save"].entity.entity_type == EntityType.METHOD
    assert q["pkg.mod.Repo.Meta"].entity.entity_type == EntityType.CLASS
    assert q["pkg.mod.Repo.Meta.info"].entity.entity_type == EntityType.METHOD
    assert q["pkg.mod.helper"].entity.entity_type == EntityType.FUNCTION
    idx = {e.entity.qualified_name: i for i, e in enumerate(a.entities)}
    assert q["pkg.mod.Repo"].parent_index == 0
    assert q["pkg.mod.Repo.save"].parent_index == idx["pkg.mod.Repo"]
    assert q["pkg.mod.Repo.Meta.info"].parent_index == idx["pkg.mod.Repo.Meta"]
    assert q["pkg.mod.helper"].parent_index == 0
    assert all(e.parent_index is None or e.parent_index < i for i, e in enumerate(a.entities))
    assert q["pkg.mod.Repo"].signature == "class Repo(Base)"
    assert (q["pkg.mod.Repo"].entity.start_line, q["pkg.mod.Repo"].entity.end_line) == (1, 5)


def test_nested_functions_and_classes_in_functions():
    src = "def outer():\n    def inner(): ...\n    class Local: ...\n"
    q = by_qname(analyze(src))
    inner = q["pkg.mod.outer.<locals>.inner"]
    assert inner.entity.entity_type == EntityType.FUNCTION and inner.in_function
    assert q["pkg.mod.outer.<locals>.Local"].entity.entity_type == EntityType.CLASS
    assert not inner.entity.is_public and not q["pkg.mod.outer"].in_function


def test_decorators_are_source_text_and_extend_start_line():
    src = "import x\n\n@app.route('/a', methods=['GET'])\n@cached\ndef view(): ...\n"
    f = by_qname(analyze(src))["pkg.mod.view"].entity
    assert f.decorators == ["app.route('/a', methods=['GET'])", "cached"]
    assert (f.start_line, f.end_line) == (3, 5)


def test_async_functions_and_methods():
    q = by_qname(analyze("async def f(): ...\nclass C:\n    async def m(self): ...\n"))
    assert q["pkg.mod.f"].entity.is_async and q["pkg.mod.f"].signature.startswith("async def f")
    m = q["pkg.mod.C.m"].entity
    assert m.is_async and m.entity_type == EntityType.METHOD


def test_public_private_detection():
    src = (
        "def pub(): ...\n"
        "def _priv(): ...\n"
        "class _Hidden:\n    def visible(self): ...\n"
        "class Pub:\n    def __init__(self): ...\n    def _p(self): ...\n    def ok(self): ...\n"
    )
    q = by_qname(analyze(src))
    pub = {k.split(".", 2)[2]: v.entity.is_public for k, v in q.items() if k != "pkg.mod"}
    assert pub == {"pub": True, "_priv": False, "_Hidden": False, "_Hidden.visible": False,
                   "Pub": True, "Pub.__init__": False, "Pub._p": False, "Pub.ok": True}
    assert by_qname(analyze("x=1", "pkg/_internal.py"))["pkg._internal"].entity.is_public is False


def test_definitions_inside_control_flow_are_found():
    src = (
        "try:\n    import fast\nexcept ImportError:\n    def impl(): ...\n"
        "if True:\n    class Cond:\n        def m(self): ...\n"
        "with open('x') as f:\n    def w(): ...\n"
    )
    assert {"pkg.mod.impl", "pkg.mod.Cond", "pkg.mod.Cond.m", "pkg.mod.w"} <= set(by_qname(analyze(src)))


def test_docstring_is_cleaned_and_missing_is_none():
    src = 'def f():\n    """Line one.\n\n    More.\n    """\ndef g(): ...\n'
    q = by_qname(analyze(src))
    assert q["pkg.mod.f"].entity.docstring == "Line one.\n\nMore."
    assert q["pkg.mod.g"].entity.docstring is None and not q["pkg.mod.g"].entity.has_docstring


def test_line_numbers_are_identical_for_crlf_and_lf():
    lf = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    for text in (lf, lf.replace("\n", "\r\n"), lf.replace("\n", "\r")):
        q = by_qname(analyze(text))
        assert (q["pkg.mod.b"].entity.start_line, q["pkg.mod.b"].entity.end_line) == (4, 5)
        assert q["pkg.mod"].entity.end_line == 5


def test_empty_file_has_only_a_module_entity():
    a = analyze("")
    assert len(a.entities) == 1 and a.entities[0].entity.end_line == 1


@pytest.mark.parametrize("bad", ["def broken(:\n    pass\n", "x = (1,\n", "class\n"])
def test_malformed_python_raises_parsing_error(bad):
    with pytest.raises(ParsingError, match="mod.py"):
        analyze(bad)


def test_syntax_error_message_includes_line_number():
    with pytest.raises(ParsingError, match="line 3"):
        analyze("x = 1\ny = 2\ndef (:\n")


def test_undecodable_and_null_byte_sources_raise_parsing_error():
    with pytest.raises(ParsingError):
        analyze_python(b"x = '\xff\xfe'\n", "bad.py")
    with pytest.raises(ParsingError):
        analyze_python(b"x = 1\x00\n", "nul.py")


def test_pep263_encoding_cookie_is_honoured():
    src = "# -*- coding: latin-1 -*-\ndef f(x='caf\u00e9'): ...\n".encode("latin-1")
    a = analyze_python(src, "l.py")
    assert a.encoding == "iso-8859-1"
    assert by_qname(a)["l.f"].entity.parameters[0].default == "'caf\u00e9'"


def test_analysis_never_executes_code(tmp_path):
    marker = tmp_path / "executed.txt"
    src = f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(1)\ndef f(): ...\n"
    analyze(src)
    assert not marker.exists()
