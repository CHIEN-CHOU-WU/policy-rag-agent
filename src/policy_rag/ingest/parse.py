"""Parse a PDF into pages of text lines and tables, keeping page numbers.

The output keeps the layout signals (font size, boldness, position) that the
chunker needs to find section headings. Tables are extracted separately and
their text is removed from the line stream so it is never chunked twice.
Running headers and footers (text in the page margins that repeats across
pages, such as the chapter name) are dropped so they don't pollute chunks.

Page numbers
------------
The page number printed on a page often differs from its position in the PDF
file (covers and front matter are usually unnumbered). Each page records both:

- ``pdf_page``: 1-based position in the file (stable, used to fetch a page).
- ``page_number``: the number a reader sees, used for citations.

``page_numbering`` records where ``page_number`` came from, in order of
preference: the PDF's page labels, the number printed in the page's
header/footer band, or the physical position as a last resort.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from policy_rag.models import PageNumbering

# Fraction of the page height treated as the header or footer band.
_MARGIN_BAND = 0.08
# Printed numbers must agree on one offset for at least this share of pages.
_MIN_OFFSET_AGREEMENT = 0.5
# Margin text seen on at least this many pages is a running header or footer.
_MIN_RUNNING_TEXT_PAGES = 3

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextLine:
    text: str
    size: float
    bold: bool
    bbox: BBox
    block: int


@dataclass(frozen=True)
class Table:
    markdown: str
    bbox: BBox


@dataclass(frozen=True)
class Page:
    pdf_page: int
    page_number: int
    page_numbering: PageNumbering
    lines: list[TextLine]
    tables: list[Table]


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    pages: list[Page]


def parse_pdf(path: Path) -> ParsedDocument:
    with pymupdf.open(path) as doc:
        raw_pages = [_read_page(page) for page in doc]
        numbering = _resolve_page_numbers(doc, raw_pages)
    running_texts = _running_texts(raw_pages)

    pages = [
        Page(
            pdf_page=index + 1,
            page_number=number,
            page_numbering=source,
            lines=[
                line
                for line in raw.lines
                if not (_in_margin_band(line.bbox, raw.height) and _shape(line.text) in running_texts)
            ],
            tables=raw.tables,
        )
        for index, (raw, (number, source)) in enumerate(zip(raw_pages, numbering))
    ]
    return ParsedDocument(source=path.name, pages=pages)


@dataclass(frozen=True)
class _RawPage:
    height: float
    lines: list[TextLine]
    tables: list[Table]
    printed_numbers: list[int]


def _read_page(page: pymupdf.Page) -> _RawPage:
    tables = _extract_tables(page)
    height = page.rect.height
    lines: list[TextLine] = []
    printed_numbers: list[int] = []

    for block_no, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(span["text"] for span in spans).split())
            bbox: BBox = tuple(line["bbox"])

            if _in_margin_band(bbox, height) and text.isdigit():
                # Page furniture: keep the number for page mapping, drop the text.
                printed_numbers.append(int(text))
                continue
            if any(_contains(table.bbox, bbox) for table in tables):
                continue
            if lines and (lines[-1].text, lines[-1].bbox) == (text, bbox):
                continue  # the same line drawn twice (overprinting)

            lines.append(
                TextLine(
                    text=text,
                    size=_dominant_size(spans),
                    bold=all(_is_bold(span) for span in spans),
                    bbox=bbox,
                    block=block_no,
                )
            )
    return _RawPage(height=height, lines=lines, tables=tables, printed_numbers=printed_numbers)


def _running_texts(raw_pages: list[_RawPage]) -> set[str]:
    """Margin text that repeats across pages, compared ignoring digits."""
    pages_seen: Counter[str] = Counter()
    for raw in raw_pages:
        pages_seen.update(
            {_shape(line.text) for line in raw.lines if _in_margin_band(line.bbox, raw.height)}
        )
    return {text for text, count in pages_seen.items() if count >= _MIN_RUNNING_TEXT_PAGES}


def _shape(text: str) -> str:
    """Text with digits removed, so "Page 3 of 9" and "Page 4 of 9" compare equal."""
    return "".join(char for char in text if not char.isdigit())


def _extract_tables(page: pymupdf.Page) -> list[Table]:
    """Find ruled tables. Degenerate one-row or one-column grids are ignored."""
    tables = []
    for table in page.find_tables().tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        rows = [[_cell_text(page, cell) for cell in row.cells] for row in table.rows]
        tables.append(Table(markdown=_to_markdown(rows), bbox=tuple(table.bbox)))
    return tables


def _cell_text(page: pymupdf.Page, cell: tuple | None) -> str:
    """Read a cell with the regular text extractor.

    ``Table.extract()`` can drop word spaces, and ``Table.to_markdown()`` may
    promote text *above* the grid (often a section heading) to the header row.
    """
    if cell is None:  # part of a merged cell
        return ""
    return " ".join(page.get_text("text", clip=cell).split()).replace("|", "/")


def _to_markdown(rows: list[list[str]]) -> str:
    """Render cell text as a Markdown table, using the first row as the header."""
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)


def _resolve_page_numbers(
    doc: pymupdf.Document, raw_pages: list[_RawPage]
) -> list[tuple[int, PageNumbering]]:
    physical = [index + 1 for index in range(len(raw_pages))]

    if doc.get_page_labels():
        labels = [page.get_label() for page in doc]
        return [
            (int(label), "pdf_label") if label.isdigit() else (pdf_page, "physical")
            for label, pdf_page in zip(labels, physical)
        ]

    offset = _printed_page_offset(raw_pages)
    if offset is None:
        return [(pdf_page, "physical") for pdf_page in physical]

    # Pages before printed page 1 (e.g. the cover) have no printed number.
    return [
        (pdf_page + offset, "printed") if pdf_page + offset >= 1 else (pdf_page, "physical")
        for pdf_page in physical
    ]


def _printed_page_offset(raw_pages: list[_RawPage]) -> int | None:
    """Return the dominant ``printed - physical`` offset, if one is consistent enough.

    Using a single document-wide offset (rather than trusting each page's own
    number) fills in pages with no visible number and ignores stray digits
    that happen to sit in the header or footer band.
    """
    votes: Counter[int] = Counter()
    for index, raw in enumerate(raw_pages):
        for offset in {number - (index + 1) for number in raw.printed_numbers}:
            votes[offset] += 1
    if not votes:
        return None
    offset, count = votes.most_common(1)[0]
    if count < len(raw_pages) * _MIN_OFFSET_AGREEMENT:
        return None
    return offset


def _in_margin_band(bbox: BBox, page_height: float) -> bool:
    _, top, _, bottom = bbox
    band = page_height * _MARGIN_BAND
    return bottom <= band or top >= page_height - band


def _contains(outer: BBox, inner: BBox) -> bool:
    """True if the centre of ``inner`` lies inside ``outer``."""
    x = (inner[0] + inner[2]) / 2
    y = (inner[1] + inner[3]) / 2
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


def _dominant_size(spans: list[dict]) -> float:
    sizes: Counter[float] = Counter()
    for span in spans:
        sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0]


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & pymupdf.TEXT_FONT_BOLD) or "bold" in span["font"].lower()
