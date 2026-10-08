import pytest

from trustdoc.chunking import chunk_text


def test_short_text_is_one_segment():
    segs = chunk_text("hello\nworld", 100, 10, base_line=5)
    assert len(segs) == 1
    assert (segs[0].start_line, segs[0].end_line) == (5, 6)


def test_blank_text_yields_nothing():
    assert chunk_text("  \n\n  ", 100, 0) == []


def test_segments_respect_max_and_reproduce_source():
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(60))
    segs = chunk_text(text, 300, 60)
    assert len(segs) > 5
    for s in segs:
        assert len(s.text) <= 300
        assert s.text == text[s.start_char : s.end_char]


def test_prefers_paragraph_then_line_boundaries():
    para = "A" * 150
    text = f"{para}\n\n{para}\n\n{para}"
    segs = chunk_text(text, 320, 0)
    assert [s.text for s in segs] == [f"{para}\n\n{para}", para]
    lines = "\n".join(["w" * 90] * 6)
    assert all(s.text.startswith("w") and s.text.endswith("w") for s in chunk_text(lines, 200, 0))


def test_line_numbers_are_exact_with_base_line():
    text = "\n".join(f"L{i}" + "y" * 50 for i in range(1, 21))
    segs = chunk_text(text, 200, 0, base_line=100)
    assert segs[0].start_line == 100
    assert segs[-1].end_line == 119
    for s in segs:
        first, last = s.text.split("\n")[0], s.text.split("\n")[-1]
        assert first.startswith(f"L{s.start_line - 99}")
        assert last.startswith(f"L{s.end_line - 99}")


def test_overlap_repeats_trailing_lines():
    text = "\n".join(f"row{i:02d}" + "z" * 30 for i in range(30))
    segs = chunk_text(text, 250, 100)
    assert segs[1].start_line <= segs[0].end_line  # windows overlap
    assert segs[1].start_line > segs[0].start_line  # but always progress


def test_unbroken_text_hard_cuts_and_terminates():
    segs = chunk_text("a" * 5000, 1200, 150)
    assert all(len(s.text) <= 1200 for s in segs)
    assert segs[0].start_char == 0 and segs[-1].end_char == 5000


@pytest.mark.parametrize("max_chars, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("x", max_chars, overlap)
