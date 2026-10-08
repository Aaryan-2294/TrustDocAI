import pytest

from trustdoc.chunking import build_markdown_chunks, build_python_chunks
from trustdoc.config import ChunkingConfig
from trustdoc.models import ChunkType
from trustdoc.parsing.markdown_parser import parse_markdown
from trustdoc.parsing.python_analyzer import analyze_python

BIG = ChunkingConfig(max_chars=10_000, overlap_chars=0)

PY_SRC = '''"""Sample module."""
import os

LIMIT = 10


@decorator
def top(a, b=1):
    """Top-level."""
    def inner():
        return a
    return inner


class Repo:
    """A repo."""
    kind = "x"

    def save(self, item):
        return item

    class Meta:
        def info(self):
            return 1

    tail = 2


if __name__ == "__main__":
    top(1)
'''


def py_chunks(src=PY_SRC, cfg=BIG, name="pkg/mod.py"):
    analysis = analyze_python(src.encode(), name)
    return analysis, build_python_chunks(analysis, name, cfg)


def slice_lines(analysis, chunk):
    return "\n".join(analysis.lines[chunk.start_line - 1 : chunk.end_line])


def test_one_chunk_per_logical_unit_with_exact_source_text():
    analysis, drafts = py_chunks()
    by_q = {}
    for d in drafts:
        by_q.setdefault(d.chunk.metadata["qualified_name"], []).append(d.chunk)
    assert set(by_q) == {"pkg.mod", "pkg.mod.top", "pkg.mod.Repo", "pkg.mod.Repo.save",
                         "pkg.mod.Repo.Meta", "pkg.mod.Repo.Meta.info"}
    top = by_q["pkg.mod.top"][0]
    assert top.text.startswith("@decorator\ndef top") and top.start_line == 7  # decorator line
    assert "def inner" in top.text                                        # nested kept inside
    for d in drafts:  # chunk text is literally the source lines it claims to cover
        assert d.chunk.text == slice_lines(analysis, d.chunk)
        assert d.chunk.chunk_type == ChunkType.CODE and d.chunk.char_count == len(d.chunk.text)


def test_class_chunk_excludes_methods_and_module_chunk_excludes_definitions():
    _, drafts = py_chunks()
    repo = [d.chunk for d in drafts if d.chunk.metadata["qualified_name"] == "pkg.mod.Repo"]
    assert 'kind = "x"' in repo[0].text and "def save" not in repo[0].text
    assert any(c.text.strip() == "tail = 2" for c in repo)  # attribute after the methods
    module = [d.chunk for d in drafts if d.chunk.metadata["qualified_name"] == "pkg.mod"]
    joined = "\n".join(c.text for c in module)
    assert "import os" in joined and "LIMIT = 10" in joined and "__main__" in joined
    assert "def top" not in joined and "class Repo" not in joined


def test_entity_links_and_sequential_indexes():
    analysis, drafts = py_chunks()
    assert [d.chunk.chunk_index for d in drafts] == list(range(len(drafts)))
    for d in drafts:
        ent = analysis.entities[d.entity_index].entity
        assert ent.qualified_name == d.chunk.metadata["qualified_name"]
        assert ent.start_line <= d.chunk.start_line <= d.chunk.end_line <= ent.end_line
    starts = [d.chunk.start_line for d in drafts]
    assert starts == sorted(starts)  # file order
    meta = next(d.chunk.metadata for d in drafts if d.chunk.metadata["qualified_name"] == "pkg.mod.Repo.save")
    assert meta["signature"] == "def save(self, item)" and meta["parent_qualified_name"] == "pkg.mod.Repo"
    assert meta["context_header"] == "pkg/mod.py :: pkg.mod.Repo.save"


def test_oversized_function_is_split_with_accurate_lines():
    body = "\n".join(f"    value_{i} = compute({i})  # padding padding" for i in range(60))
    src = f"def long_one():\n{body}\n    return 0\n"
    analysis, drafts = py_chunks(src, ChunkingConfig(max_chars=400, overlap_chars=80))
    chunks = [d.chunk for d in drafts]
    assert len(chunks) > 3
    assert all(c.metadata["parts"] == len(chunks) for c in chunks)
    assert [c.metadata["part"] for c in chunks] == list(range(1, len(chunks) + 1))
    for c in chunks:
        assert len(c.text) <= 400 and c.text == slice_lines(analysis, c)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 62
    assert all(d.entity_index == drafts[0].entity_index for d in drafts)


def test_blank_only_module_yields_no_chunks():
    assert py_chunks("\n\n")[1] == []


# --------------------------------------------------------------------- markdown
MD = """Intro paragraph before any heading.

# Project
Overview text.

## Installation
Run the installer.

```bash
# comment, not a heading
pip install trustdoc
```

### Troubleshooting
If it fails, retry.

## Usage
Call the CLI.

Setext Title
============
Setext body.

## Empty parent
### Child
Child body.
"""


def md_doc(text=MD):
    return parse_markdown(text.encode(), "README.md")


def test_headings_levels_paths_and_lines():
    doc = md_doc()
    got = [(s.level, s.path_label, s.start_line, s.end_line) for s in doc.sections]
    assert got == [
        (0, None, 1, 1),
        (1, "Project", 3, 4),
        (2, "Project > Installation", 6, 12),
        (3, "Project > Installation > Troubleshooting", 14, 15),
        (2, "Project > Usage", 17, 18),
        (1, "Setext Title", 20, 22),
        (2, "Setext Title > Empty parent", 24, 24),
        (3, "Setext Title > Empty parent > Child", 25, 26),
    ]
    assert doc.metadata["heading_count"] == 7 and doc.metadata["title"] == "Project"


def test_fenced_code_hash_is_not_a_heading():
    assert "comment, not a heading" not in [h["text"] for h in md_doc().metadata["headings"]]


def test_markdown_chunks_have_heading_lines_and_exact_text():
    doc = md_doc()
    chunks = [d.chunk for d in build_markdown_chunks(doc, "README.md", BIG)]
    assert all(c.chunk_type == ChunkType.MARKDOWN for c in chunks)
    labels = [c.heading for c in chunks]
    assert "Project > Installation" in labels and "Setext Title > Empty parent" not in labels  # heading-only skipped
    assert None in labels  # preamble
    inst = next(c for c in chunks if c.heading == "Project > Installation")
    assert inst.start_line == 6 and inst.end_line == 12
    assert inst.text == "\n".join(doc.lines[5:12])
    assert inst.location_label("README.md") == "README.md:6-12 \u00a7 Project > Installation"
    assert inst.metadata["heading_path"] == ["Project", "Installation"]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_long_section_splits_but_keeps_heading_and_lines():
    text = "# Big\n" + "\n".join(f"sentence number {i} is here." for i in range(80))
    doc = parse_markdown(text.encode(), "big.md")
    chunks = [d.chunk for d in build_markdown_chunks(doc, "big.md", ChunkingConfig(300, 40))]
    assert len(chunks) > 3 and all(c.heading == "Big" for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 81
    assert all(len(c.text) <= 300 for c in chunks)


def test_front_matter_is_not_mistaken_for_a_setext_heading():
    doc = parse_markdown(b"---\ntitle: Demo\n---\n# Real\nText\n", "x.md")
    assert [s.path_label for s in doc.sections] == [None, "Real"]
    assert doc.metadata["has_front_matter"] is True


def test_markdown_without_headings_and_empty_markdown():
    doc = parse_markdown(b"just text\nmore text\n", "a.md")
    assert [(s.heading, s.start_line, s.end_line) for s in doc.sections] == [(None, 1, 2)]
    assert parse_markdown(b"", "e.md").sections == []


def test_atx_closing_hashes_and_crlf():
    doc = parse_markdown(b"## Title ##\r\nbody\r\n", "c.md")
    assert doc.sections[0].heading == "Title" and doc.sections[0].end_line == 2


def test_invalid_utf8_markdown_raises():
    from trustdoc.exceptions import ParsingError
    with pytest.raises(ParsingError):
        parse_markdown(b"\xff\xfe\x00bad", "bad.md")
