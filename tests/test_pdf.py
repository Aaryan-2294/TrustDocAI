import pytest

pytest.importorskip("pymupdf")

from trustdoc.chunking import build_pdf_chunks  # noqa: E402
from trustdoc.config import ChunkingConfig  # noqa: E402
from trustdoc.exceptions import ParsingError  # noqa: E402
from trustdoc.models import ChunkType  # noqa: E402
from trustdoc.parsing.pdf_reader import read_pdf  # noqa: E402

BIG = ChunkingConfig(max_chars=10_000, overlap_chars=0)


def test_pages_text_and_metadata(make_pdf):
    doc = read_pdf(make_pdf(["First page text.", "Second page text."]), "ref.pdf")
    assert [p.number for p in doc.pages] == [1, 2]
    assert "First page text." in doc.pages[0].text and "Second" in doc.pages[1].text
    assert doc.metadata["page_count"] == 2 and doc.metadata["pages_with_text"] == 2
    assert doc.metadata["empty_pages"] == [] and doc.metadata["char_count"] > 20


def test_pdf_info_metadata_is_recorded(make_pdf):
    import pymupdf
    raw = make_pdf(["x"])
    d = pymupdf.open(stream=raw, filetype="pdf")
    d.set_metadata({"title": "Spec", "author": "Ann"})
    info = read_pdf(d.tobytes(), "m.pdf").metadata["pdf_info"]
    assert info["title"] == "Spec" and info["author"] == "Ann"


def test_empty_pages_are_skipped_gracefully(make_pdf):
    doc = read_pdf(make_pdf(["Alpha content", None, "Gamma content"]), "g.pdf")
    assert doc.metadata["empty_pages"] == [2] and doc.metadata["pages_with_text"] == 2
    chunks = [d.chunk for d in build_pdf_chunks(doc, "g.pdf", BIG)]
    assert [c.page_number for c in chunks] == [1, 3]
    assert all(c.chunk_type == ChunkType.PDF_TEXT and c.start_line is None for c in chunks)


def test_image_only_page_is_flagged_not_ocrd(make_pdf):
    import pymupdf
    d = pymupdf.open()
    page = d.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pix.clear_with(180)
    page.insert_image(page.rect, pixmap=pix)
    doc = read_pdf(d.tobytes(), "scan.pdf")
    assert doc.metadata["image_only_pages"] == [1] and doc.metadata["pages_with_text"] == 0
    assert build_pdf_chunks(doc, "scan.pdf", BIG) == []


def test_chunks_carry_page_numbers_and_never_span_pages(make_pdf):
    pages = [" ".join(f"word{p}_{i}" for i in range(120)) for p in (1, 2, 3)]
    doc = read_pdf(make_pdf(pages), "long.pdf")
    chunks = [d.chunk for d in build_pdf_chunks(doc, "long.pdf", ChunkingConfig(300, 50))]
    assert {c.page_number for c in chunks} == {1, 2, 3}
    for c in chunks:
        assert len(c.text) <= 300
        other = {str(p) for p in (1, 2, 3)} - {str(c.page_number)}
        assert not any(f"word{o}_" in c.text for o in other)
        assert c.metadata["page_count"] == 3
        assert c.location_label("long.pdf") == f"long.pdf, p.{c.page_number}"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    page_one = doc.pages[0].text
    for c in (x for x in chunks if x.page_number == 1):  # offsets index the page text
        assert page_one[c.metadata["char_start"] : c.metadata["char_end"]] == c.text


def test_corrupt_and_encrypted_pdfs_raise_parsing_error(make_pdf):
    import pymupdf
    with pytest.raises(ParsingError):
        read_pdf(b"this is not a pdf", "bad.pdf")
    enc = make_pdf(["secret"], encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(ParsingError, match="password"):
        read_pdf(enc, "locked.pdf")
