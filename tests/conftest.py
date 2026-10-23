"""Builds a small synthetic policy handbook so tests never depend on downloaded files.

Layout (physical page -> printed page):
  1 -> (none)  cover
  2 -> 1       H1 "Section 1: Coverage basics", H2 "What Part A covers",
               H2 "What Part B covers" (its body continues onto the next page)
  3 -> 2       rest of Part B, H2 "Costs at a glance" with a ruled table
  4 -> 3       H2 "Appeals process" (13pt, while sibling H2s are 14pt), long
               enough to force length-based splitting
  5 -> 4       H1 "Section 2: Getting help", H2 wrapped over two lines

Every numbered page also carries a running footer, drawn *before* the body so
the PDF's content order differs from the visual reading order.
"""

from pathlib import Path

import pymupdf
import pytest

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
LEFT, RIGHT = 72, 540
BODY_SIZE = 11

PART_A_TEXT = (
    "Part A covers inpatient hospital stays, care in a skilled nursing facility, "
    "hospice care, and some home health care."
)
PART_B_FIRST = (
    "Part B covers services from doctors and other health care providers, "
    "outpatient care, durable medical equipment, and many preventive services."
)
PART_B_SECOND = (
    "Part B also covers ambulance services and limited outpatient prescription drugs. "
    "You usually pay 20 percent of the approved amount after you meet the deductible."
)
TABLE_ROWS = [
    ["Service", "Part A", "Part B"],
    ["Premium", "Usually free", "Monthly premium"],
    ["Deductible", "Per benefit period", "Yearly deductible"],
]
AFTER_TABLE_TEXT = "Costs may change each year, so review your plan notice every fall."
HELP_HEADING_LINES = ["Where to find help", "in your community"]
HELP_TEXT = "Local health insurance counseling programs give free, personalized advice."
FOOTER_TEXT = "Questions? Call 1-800-555-0100."


def appeals_paragraphs() -> list[str]:
    """Roughly 3,000 characters of distinct, numbered sentences."""
    paragraphs = []
    for p in range(5):
        sentences = [
            f"Appeal rule {p}.{s} explains one step of the appeals process in plain language."
            for s in range(7)
        ]
        paragraphs.append(" ".join(sentences))
    return paragraphs


class _PageWriter:
    def __init__(self, doc: pymupdf.Document, printed_number: int | None) -> None:
        self.page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.y = 90.0
        if printed_number is not None:
            self.page.insert_text((LEFT, 760), FOOTER_TEXT, fontname="helv", fontsize=9)
            self.page.insert_text((LEFT, 30), str(printed_number), fontname="helv", fontsize=10)

    def heading(self, text: str, size: float) -> None:
        self.y += size * 1.6
        self.page.insert_text((LEFT, self.y), text, fontname="hebo", fontsize=size)
        self.y += size * 0.6

    def paragraph(self, text: str) -> None:
        rect = pymupdf.Rect(LEFT, self.y, RIGHT, PAGE_HEIGHT - 72)
        spare = self.page.insert_textbox(rect, text, fontname="helv", fontsize=BODY_SIZE)
        assert spare >= 0, "paragraph does not fit on the page"
        self.y = rect.y1 - spare + BODY_SIZE  # leave a blank line between paragraphs

    def table(self, rows: list[list[str]]) -> None:
        row_height, col_width = 24, (RIGHT - LEFT) / len(rows[0])
        top = self.y + 8
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(
                    LEFT + c * col_width,
                    top + r * row_height,
                    LEFT + (c + 1) * col_width,
                    top + (r + 1) * row_height,
                )
                self.page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                self.page.insert_text(
                    (rect.x0 + 4, rect.y0 + 16), cell, fontname="helv", fontsize=10
                )
        self.y = top + len(rows) * row_height + 12


def build_handbook(path: Path) -> Path:
    doc = pymupdf.open()

    cover = _PageWriter(doc, printed_number=None)
    cover.heading("Sample Policy Handbook", 28)

    page = _PageWriter(doc, printed_number=1)
    page.heading("Section 1: Coverage basics", 20)
    page.heading("What Part A covers", 14)
    page.paragraph(PART_A_TEXT)
    page.heading("What Part B covers", 14)
    page.paragraph(PART_B_FIRST)

    page = _PageWriter(doc, printed_number=2)
    page.paragraph(PART_B_SECOND)
    page.heading("Costs at a glance", 14)
    page.table(TABLE_ROWS)
    page.paragraph(AFTER_TABLE_TEXT)

    page = _PageWriter(doc, printed_number=3)
    page.heading("Appeals process", 13)
    for text in appeals_paragraphs():
        page.paragraph(text)

    page = _PageWriter(doc, printed_number=4)
    page.heading("Section 2: Getting help", 20)
    page.page.insert_text((LEFT, page.y + 22), HELP_HEADING_LINES[0], fontname="hebo", fontsize=14)
    page.page.insert_text((LEFT, page.y + 39), HELP_HEADING_LINES[1], fontname="hebo", fontsize=14)
    page.y += 50
    page.paragraph(HELP_TEXT)

    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="session")
def handbook_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_handbook(tmp_path_factory.mktemp("pdf") / "sample-handbook.pdf")
