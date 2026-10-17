import pytest

from trustdoc.verification.typenorm import TypeMatch, compare_types, normalize_type, word_to_type

M, X, U = TypeMatch.MATCH, TypeMatch.MISMATCH, TypeMatch.UNKNOWN


@pytest.mark.parametrize("documented, actual, expected", [
    ("float", "float", M), ("int", "float", X), ("float", "int", X), ("str", "'str'", M), ("None", "None", M),
    ("int", "None", X), ("list", "List[str]", M), ("List[str]", "list[str]", M), ("list", "list[int]", M),
    ("list[str]", "list", U),                                  # docs more specific than the annotation: cannot confirm
    ("list[str]", "list[int]", X), ("dict", "Dict[str, int]", M), ("Optional[int]", "int | None", M),
    ("Union[int, None]", "Optional[int]", M), ("int | str", "str | int", M), ("int | str", "int", U),
    ("Path", "pathlib.Path", M), ("dict", "TaskResult", U), ("Task", "Job", U),   # user types / aliases: undecidable
    ("bool", "int", U), ("int", "bool", U),                    # bool is an int: never call it wrong
    ("typing.List[int]", "list[int]", M), ("tuple[int, ...]", "Tuple[int, ...]", M),
    ("float", "builtins_unknown_thing(", U), ("", "float", U), (None, "float", U), ("float", None, U),
])
def test_compare_types(documented, actual, expected):
    assert compare_types(documented, actual) == expected


def test_normalisation_is_canonical_and_order_insensitive():
    assert normalize_type("Optional[List[Dict[str, int]]]").canonical == "None | list[dict[str, int]]"
    assert normalize_type("str | int").canonical == normalize_type("int | str").canonical == "int | str"
    assert normalize_type("'Task'").canonical == "Task" and normalize_type("`float`").canonical == "float"


@pytest.mark.parametrize("junk", ["", "   ", "not a type!", "x" * 300, "1 + ", "def"])
def test_unparseable_text_is_not_a_type(junk):
    assert normalize_type(junk) is None


def test_prose_words_map_to_canonical_types():
    assert word_to_type("floating-point number") == "float" and word_to_type("Integers") == "int"
    assert word_to_type("nothing") == "None" and word_to_type("dictionary") == "dict" and word_to_type("banana") is None
