from pathlib import Path

from conftest import AFTER_TABLE_TEXT, FOOTER_TEXT, PART_B_FIRST, PART_B_SECOND
from fakes import make_page
from policy_rag.ingest.chunk import render_pages
from policy_rag.ingest.parse import parse_pdf
from policy_rag.store.page_store import JsonPageStore


def test_render_pages_keeps_page_numbers_and_reading_order(handbook_pdf: Path) -> None:
    pages = render_pages(parse_pdf(handbook_pdf))

    assert [(page.pdf_page, page.page_number) for page in pages] == [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4)]
    page_2 = pages[2].text
    assert page_2.index(PART_B_SECOND) < page_2.index("Costs at a glance")
    assert page_2.index("| Service | Part A | Part B |") < page_2.index(AFTER_TABLE_TEXT)
    assert PART_B_FIRST in pages[1].text


def test_render_pages_drops_running_footers(handbook_pdf: Path) -> None:
    assert not any(FOOTER_TEXT in page.text for page in render_pages(parse_pdf(handbook_pdf)))


def test_json_page_store_round_trip(tmp_path: Path) -> None:
    store = JsonPageStore(tmp_path / "nested" / "pages.json")
    store.save([make_page(9, "Page nine."), make_page(10, "Page ten.")])

    assert store.get(10) == [make_page(10, "Page ten.")]
    assert store.get(11) == []


def test_json_page_store_without_a_file_is_empty(tmp_path: Path) -> None:
    assert JsonPageStore(tmp_path / "missing.json").get(1) == []
