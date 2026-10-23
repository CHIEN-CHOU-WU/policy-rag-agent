from pathlib import Path

import pymupdf

from policy_rag.ingest.parse import parse_pdf


def test_printed_page_numbers_are_mapped_with_their_offset(handbook_pdf: Path) -> None:
    pages = parse_pdf(handbook_pdf).pages
    assert [page.pdf_page for page in pages] == [1, 2, 3, 4, 5]
    assert [page.page_number for page in pages[1:]] == [1, 2, 3, 4]
    assert all(page.page_numbering == "printed" for page in pages[1:])


def test_unnumbered_front_matter_falls_back_to_physical(handbook_pdf: Path) -> None:
    cover = parse_pdf(handbook_pdf).pages[0]
    assert (cover.page_number, cover.page_numbering) == (1, "physical")


def test_printed_page_numbers_are_removed_from_text(handbook_pdf: Path) -> None:
    for page in parse_pdf(handbook_pdf).pages:
        assert not any(line.text.isdigit() for line in page.lines)


def test_pdf_page_labels_take_precedence(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page().insert_text((72, 100), "Body text.", fontsize=11)
    doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 10}])
    path = tmp_path / "labelled.pdf"
    doc.save(path)

    pages = parse_pdf(path).pages
    assert [(p.page_number, p.page_numbering) for p in pages] == [
        (10, "pdf_label"),
        (11, "pdf_label"),
        (12, "pdf_label"),
    ]


def test_without_any_numbering_physical_pages_are_used(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 100), "Body text.", fontsize=11)
    path = tmp_path / "plain.pdf"
    doc.save(path)

    pages = parse_pdf(path).pages
    assert [(p.page_number, p.page_numbering) for p in pages] == [(1, "physical"), (2, "physical")]
