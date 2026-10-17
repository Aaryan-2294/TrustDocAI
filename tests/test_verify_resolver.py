import pytest
from verify_support import P, entity, module

from trustdoc.models import EntityType
from trustdoc.parsing.python_analyzer import analyze_python
from trustdoc.verification.resolver import AliasIndex, EntityResolver, ResolutionKind as R, clean_name

K = __import__("trustdoc.models", fromlist=["ParameterKind"]).ParameterKind


@pytest.fixture
def resolver():
    ents = [module("pkg.mod"), entity("pkg.mod.compute_backoff", params=[P("attempt")]),
            entity("pkg.mod.Queue", type=EntityType.CLASS),
            entity("pkg.mod.Queue.enqueue", type=EntityType.METHOD, params=[P("self"), P("payload")]),
            entity("pkg.other.Store.enqueue", type=EntityType.METHOD, params=[P("self"), P("item")])]
    return EntityResolver(ents, AliasIndex({"exported"}))


@pytest.mark.parametrize("name", ["pkg.mod.compute_backoff", "compute_backoff", "mod.compute_backoff", "Queue", "pkg.mod.Queue.enqueue",
                                  "Queue.enqueue", "compute_backoff()", "`compute_backoff`", "self.compute_backoff"])
def test_found_by_exact_and_component_suffix(resolver, name):
    res = resolver.resolve(name)
    assert res.kind == R.FOUND and res.entity.name in ("compute_backoff", "Queue", "enqueue")


def test_simple_names_shared_by_several_entities_are_ambiguous(resolver):
    res = resolver.resolve("enqueue")
    assert res.kind == R.AMBIGUOUS and len(res.candidates) == 2 and "pkg.other.Store.enqueue" in res.reason


def test_kind_word_disambiguates():
    r = EntityResolver([module("pkg.a"), entity("pkg.a.Task", type=EntityType.CLASS), entity("pkg.b.Task", type=EntityType.FUNCTION)])
    assert r.resolve("Task").kind == R.AMBIGUOUS
    assert r.resolve("Task", kind_word="class").entity.entity_type == EntityType.CLASS
    assert r.resolve("Task", kind_word="function").entity.entity_type == EntityType.FUNCTION


def test_overloads_property_setters_and_identical_redefinitions_collapse():
    ents = [entity("pkg.C.x", type=EntityType.METHOD, decorators=["property"], params=[P("self")]),
            entity("pkg.C.x", type=EntityType.METHOD, decorators=["x.setter"], params=[P("self"), P("value")]),
            entity("pkg.f", params=[P("a")]), entity("pkg.f", params=[P("a")])]                       # try/except fallback, same signature
    r = EntityResolver(ents)
    assert r.resolve("pkg.C.x").kind == R.FOUND and r.resolve("pkg.C.x").entity.decorators == ["property"]
    assert r.resolve("pkg.f").kind == R.FOUND


def test_conditional_redefinitions_with_different_signatures_are_ambiguous_not_guessed():
    r = EntityResolver([entity("pkg.f", params=[P("a")]), entity("pkg.f", params=[P("a"), P("b")])])
    assert r.resolve("pkg.f").kind == R.AMBIGUOUS


def test_bound_but_not_defined_names_are_aliases_not_missing(resolver):
    res = resolver.resolve("exported")
    assert res.kind == R.ALIAS and "import" in res.reason


@pytest.mark.parametrize("name", ["json.loads", "os.path.join", "requests.get", "print", "len", "numpy.array"])
def test_builtin_stdlib_and_external_names_are_external(resolver, name):
    assert resolver.resolve(name).kind == R.EXTERNAL


def test_not_found_is_strong_only_when_rooted_in_a_project_package(resolver):
    strong = resolver.resolve("pkg.mod.nope")
    assert strong.kind == R.NOT_FOUND and strong.strong
    weak = resolver.resolve("retry_all")
    assert weak.kind == R.NOT_FOUND and not weak.strong
    assert resolver.resolve("Ghost.method").kind == R.NOT_FOUND and not resolver.resolve("Ghost.method").strong


@pytest.mark.parametrize("name", ["", "   ", "not valid!", "a-b", "1abc", "f(x, y)garbage ("])
def test_invalid_names(resolver, name):
    assert resolver.resolve(name).kind in (R.INVALID,)


def test_clean_name_strips_decoration():
    assert clean_name(" `self.run()` ") == "run" and clean_name("cls.make") == "make" and clean_name("pkg.f().") == "pkg.f"


def test_effective_parameters_remove_bound_arguments_only():
    ents = [module("pkg"),
            entity("pkg.C", type=EntityType.CLASS),
            entity("pkg.C.__init__", type=EntityType.METHOD, params=[P("self"), P("x"), P("y", "1")], public=False),
            entity("pkg.C.m", type=EntityType.METHOD, params=[P("self"), P("a")]),
            entity("pkg.C.cm", type=EntityType.METHOD, params=[P("cls"), P("a")], decorators=["classmethod"]),
            entity("pkg.C.sm", type=EntityType.METHOD, params=[P("a"), P("b")], decorators=["staticmethod"]),
            entity("pkg.C.odd", type=EntityType.METHOD, params=[P("this"), P("a")]),
            entity("pkg.fn", params=[P("self"), P("a")]),
            entity("pkg.NoInit", type=EntityType.CLASS)]
    r = EntityResolver(ents)
    names = lambda q: [p.name for p in r.effective_parameters(r.resolve(q).entity)]  # noqa: E731
    assert names("C.m") == ["a"] and names("C.cm") == ["a"] and names("C.sm") == ["a", "b"]
    assert names("C.odd") == ["this", "a"]                       # only conventional bound names are dropped
    assert names("pkg.fn") == ["self", "a"]                      # a plain function's `self` is a real parameter
    assert names("pkg.C") == ["x", "y"]                          # constructor: __init__ without self
    assert r.effective_parameters(r.resolve("NoInit").entity) is None


def test_alias_index_collects_imports_assignments_and_all():
    src = ("from .impl import helper as h, other\nimport os.path, json as j\nLIMIT = 3\nCOUNT: int = 4\n__all__ = ['exported', 'other']\n"
           "def real(): ...\n")
    idx = AliasIndex.from_sources([src, "def broken(:"])           # a broken file is skipped, not fatal
    assert {"h", "other", "os", "j", "LIMIT", "COUNT", "exported", "__all__"} <= idx.names and "real" not in idx.names


def test_resolver_works_on_real_analyzer_output():
    a = analyze_python(b"class A:\n    def run(self, x): ...\n\ndef run(y): ...\n", "pkg/m.py")
    r = EntityResolver([e.entity for e in a.entities])
    assert r.resolve("A.run").kind == R.FOUND and r.resolve("run").kind == R.AMBIGUOUS and r.resolve("pkg.m").entity.entity_type == EntityType.MODULE
