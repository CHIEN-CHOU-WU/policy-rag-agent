from pathlib import Path

import pytest

from conftest import (
    AFTER_TABLE_TEXT,
    FOOTER_TEXT,
    HELP_HEADING_LINES,
    HELP_TEXT,
    PART_A_TEXT,
    PART_B_FIRST,
    PART_B_SECOND,
    TABLE_ROWS,
    appeals_paragraphs,
)
from policy_rag.ingest.chunk import ChunkerConfig, chunk_document
from policy_rag.ingest.parse import parse_pdf
from policy_rag.models import Chunk

CONFIG = ChunkerConfig(max_chars=1500, overlap_chars=200)


@pytest.fixture(scope="module")
def chunks(handbook_pdf: Path) -> list[Chunk]:
    return chunk_document(parse_pdf(handbook_pdf), CONFIG)


def by_section(chunks: list[Chunk], title: str) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.section_title == title]


def test_headings_become_section_titles(chunks: list[Chunk]) -> None:
    titles = {chunk.section_title for chunk in chunks}
    assert {"What Part A covers", "What Part B covers", "Costs at a glance", "Appeals process"} <= titles


def test_section_path_tracks_heading_hierarchy(chunks: list[Chunk]) -> None:
    [part_a] = by_section(chunks, "What Part A covers")
    assert part_a.section_path.endswith("Section 1: Coverage basics > What Part A covers")

    [help_chunk] = by_section(chunks, " ".join(HELP_HEADING_LINES))
    # A same-level H1 closes the previous H1, so Section 1 is not an ancestor.
    assert "Section 1" not in help_chunk.section_path
    assert help_chunk.section_path.endswith("Section 2: Getting help > Where to find help in your community")


def test_slightly_different_heading_sizes_are_siblings(chunks: list[Chunk]) -> None:
    [appeals, *_] = by_section(chunks, "Appeals process")
    assert appeals.section_path.endswith("Section 1: Coverage basics > Appeals process")


def test_wrapped_heading_is_merged_into_one_title(chunks: list[Chunk]) -> None:
    [help_chunk] = by_section(chunks, " ".join(HELP_HEADING_LINES))
    assert HELP_TEXT in help_chunk.text


def test_short_section_is_one_chunk_with_its_title(chunks: list[Chunk]) -> None:
    [part_a] = by_section(chunks, "What Part A covers")
    assert part_a.text.startswith(part_a.section_path)
    assert PART_A_TEXT in part_a.text
    assert part_a.content_type == "text"


def test_section_spanning_pages_records_page_range(chunks: list[Chunk]) -> None:
    [part_b] = by_section(chunks, "What Part B covers")
    assert PART_B_FIRST in part_b.text and PART_B_SECOND in part_b.text
    assert (part_b.page_start, part_b.page_end) == (1, 2)
    assert (part_b.pdf_page_start, part_b.pdf_page_end) == (2, 3)
    assert part_b.page_numbering == "printed"


def test_table_is_a_single_intact_chunk(chunks: list[Chunk]) -> None:
    tables = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(tables) == 1
    [table] = tables
    assert table.section_title == "Costs at a glance"
    for row in TABLE_ROWS:
        assert "| " + " | ".join(row) + " |" in table.text
    assert AFTER_TABLE_TEXT not in table.text


def test_table_follows_reading_order_not_content_order(chunks: list[Chunk]) -> None:
    # The footer is drawn first on every page; the table must still land
    # under the heading printed above it, after the Part B text.
    sections_in_order = [chunk.section_title for chunk in chunks]
    table_index = [chunk.content_type for chunk in chunks].index("table")
    assert sections_in_order[table_index - 1] == "What Part B covers"


def test_running_footer_is_not_in_any_chunk(chunks: list[Chunk]) -> None:
    assert not any(FOOTER_TEXT in chunk.text for chunk in chunks)


def test_text_around_a_table_is_kept_separately(chunks: list[Chunk]) -> None:
    text_chunks = [c for c in by_section(chunks, "Costs at a glance") if c.content_type == "text"]
    assert [AFTER_TABLE_TEXT in c.text for c in text_chunks] == [True]


def test_table_is_never_split_even_when_longer_than_max(handbook_pdf: Path) -> None:
    tiny = ChunkerConfig(max_chars=60, overlap_chars=10)
    tables = [c for c in chunk_document(parse_pdf(handbook_pdf), tiny) if c.content_type == "table"]
    assert len(tables) == 1
    assert all(" | ".join(row) in tables[0].text for row in TABLE_ROWS)


def test_long_section_falls_back_to_length_split(chunks: list[Chunk]) -> None:
    appeals = by_section(chunks, "Appeals process")
    assert len(appeals) > 1
    assert all(len(chunk.text) <= CONFIG.max_chars for chunk in appeals)

    body = " ".join(chunk.text for chunk in appeals)
    for paragraph in appeals_paragraphs():
        for sentence in paragraph.split(". "):
            assert sentence.rstrip(".") in body


def test_split_pieces_overlap_on_whole_sentences(chunks: list[Chunk]) -> None:
    appeals = by_section(chunks, "Appeals process")
    for previous, current in zip(appeals, appeals[1:]):
        body = current.body
        first_sentence = body.split(". ")[0] + "."
        assert first_sentence in previous.text, "next piece should start with the previous tail"
        assert len(first_sentence) <= CONFIG.overlap_chars


def test_chunk_ids_are_unique_and_deterministic(handbook_pdf: Path, chunks: list[Chunk]) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "sample-handbook-0000"
    assert ids == [c.chunk_id for c in chunk_document(parse_pdf(handbook_pdf), CONFIG)]


def test_no_chunk_is_empty(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert chunk.body.strip()
