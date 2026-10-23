"""Domain models shared across ingest, storage, and (later) retrieval."""

from dataclasses import dataclass
from typing import Literal

PageNumbering = Literal["pdf_label", "printed", "physical", "mixed"]
"""How a page number was determined.

- ``pdf_label``: the PDF's own page labels (``/PageLabels``).
- ``printed``: the number printed on the page, inferred from header/footer text.
- ``physical``: the 1-based position of the page in the PDF file.
- ``mixed``: a chunk spanning pages whose numbers came from different sources.
"""

ContentType = Literal["text", "table"]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of a source document, with its citation metadata."""

    chunk_id: str
    text: str
    source: str
    section_title: str
    section_path: str
    page_start: int
    page_end: int
    pdf_page_start: int
    pdf_page_end: int
    page_numbering: PageNumbering
    content_type: ContentType

    @property
    def body(self) -> str:
        """The chunk text without the section-path header it starts with."""
        return self.text.removeprefix(self.section_path).lstrip()

    @property
    def page_label(self) -> str:
        """The citation page, e.g. ``"9"``, or ``"25-26"`` for a chunk spanning pages."""
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by a vector search, with its similarity score (higher is closer)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class PageText:
    """The full, cleaned text of one page, for reading a page in context."""

    source: str
    page_number: int
    pdf_page: int
    page_numbering: PageNumbering
    text: str
