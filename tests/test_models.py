from trustdoc.models import (Chunk, ChunkType, CodeEntity, EntityType, IssueType, Parameter,
                             ParameterKind, Severity)


def test_parameter_roundtrip_and_default_semantics():
    p = Parameter("x", ParameterKind.KEYWORD_ONLY, "int", "None")
    assert Parameter.from_dict(p.to_dict()) == p
    assert p.has_default                      # default of None is still a default
    assert not Parameter("y").has_default


def test_severity_rank_orders():
    assert Severity.LOW.rank < Severity.MEDIUM.rank < Severity.HIGH.rank < Severity.CRITICAL.rank


def test_tier2_issue_types_are_distinct():
    assert IssueType.CONTRADICTED_CLAIM != IssueType.UNSUPPORTED_CLAIM


def test_location_labels():
    code = Chunk(1, 1, 0, ChunkType.CODE, "x", start_line=10, end_line=24)
    assert code.location_label("pkg/a.py") == "pkg/a.py:10-24"
    assert Chunk(1, 1, 0, ChunkType.CODE, "x", start_line=5, end_line=5).location_label("a.py") == "a.py:5"
    pdf = Chunk(1, 2, 0, ChunkType.PDF_TEXT, "x", page_number=3)
    assert pdf.location_label("ref.pdf") == "ref.pdf, p.3"
    md = Chunk(1, 3, 0, ChunkType.MARKDOWN, "x", start_line=1, end_line=9, heading="Install")
    assert md.location_label("README.md") == "README.md:1-9 \u00a7 Install"
    assert code.char_count == 1


def test_has_docstring():
    e = CodeEntity(1, 1, EntityType.FUNCTION, "f", "m.f", docstring="   ")
    assert not e.has_docstring
    e.docstring = "Does things."
    assert e.has_docstring
