"""Split a parsed document into chunks that follow its own section structure.

Strategy
--------
1. **Find headings from typography.** The body font size is the size covering
   the most characters in the document. A line is a heading when it is bold,
   noticeably larger than body text, and short. Detecting this per document
   (instead of hard-coding sizes) keeps the chunker usable on other PDFs.
2. **One section = one chunk** whenever it fits within ``max_chars``.
3. **Fall back to length-based splitting** only for long sections: split on
   paragraph and sentence boundaries, and repeat the last sentences of each
   piece at the start of the next (``overlap_chars``) so no clause loses its
   context.
4. **Tables are atomic.** Each table becomes its own chunk and is never split,
   however long it is.

Every chunk starts with its section path (e.g. ``Part B > What's covered``) so
the embedding carries the section's meaning, and records the pages it spans.

``render_pages`` turns the same cleaned element stream into one text per page,
so reading a whole page shows exactly what the chunks were built from.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from policy_rag.ingest.parse import Page, ParsedDocument, Table, TextLine
from policy_rag.models import Chunk, ContentType, PageNumbering, PageText

UNTITLED_SECTION = "Untitled section"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+")


@dataclass(frozen=True)
class ChunkerConfig:
    max_chars: int = 1500
    overlap_chars: int = 200
    # A heading must be at least this many points larger than body text.
    min_heading_size_delta: float = 1.0
    min_heading_chars: int = 3
    max_heading_chars: int = 120
    # Headings whose sizes differ by less than this are treated as the same level.
    heading_level_tolerance: float = 1.5


def chunk_document(doc: ParsedDocument, config: ChunkerConfig | None = None) -> list[Chunk]:
    config = config or ChunkerConfig()
    body_size = _body_font_size(doc.pages)
    elements = _elements(doc.pages, body_size, config)

    chunks: list[Chunk] = []
    for section in _sections(elements, config.heading_level_tolerance):
        for content_type, units in _section_pieces(section, config):
            chunks.append(
                _make_chunk(
                    chunk_id=f"{Path(doc.source).stem}-{len(chunks):04d}",
                    source=doc.source,
                    section=section,
                    content_type=content_type,
                    units=units,
                )
            )
    return chunks


def render_pages(doc: ParsedDocument, config: ChunkerConfig | None = None) -> list[PageText]:
    """The cleaned text of every page: headings, paragraphs, and Markdown tables in reading order."""
    config = config or ChunkerConfig()
    elements = _elements(doc.pages, _body_font_size(doc.pages), config)
    texts: dict[int, list[str]] = {page.pdf_page: [] for page in doc.pages}
    for element in elements:
        texts[element.page.pdf_page].append(element.text)
    return [
        PageText(
            source=doc.source,
            page_number=page.page_number,
            pdf_page=page.pdf_page,
            page_numbering=page.page_numbering,
            text="\n\n".join(texts[page.pdf_page]),
        )
        for page in doc.pages
    ]


# --- Element stream ----------------------------------------------------------


@dataclass(frozen=True)
class _PageRef:
    pdf_page: int
    page_number: int
    page_numbering: PageNumbering


@dataclass(frozen=True)
class _Element:
    kind: str  # "heading" | "paragraph" | "table"
    text: str
    page: _PageRef
    size: float = 0.0
    bottom: float = 0.0


def _body_font_size(pages: list[Page]) -> float:
    chars_by_size: dict[float, int] = {}
    for page in pages:
        for line in page.lines:
            chars_by_size[line.size] = chars_by_size.get(line.size, 0) + len(line.text)
    return max(chars_by_size, key=chars_by_size.get) if chars_by_size else 0.0


def _is_heading(line: TextLine, body_size: float, config: ChunkerConfig) -> bool:
    return (
        line.bold
        and line.size >= body_size + config.min_heading_size_delta
        and config.min_heading_chars <= len(line.text) <= config.max_heading_chars
        and any(char.isalpha() for char in line.text)
    )


def _elements(pages: list[Page], body_size: float, config: ChunkerConfig) -> list[_Element]:
    """Turn pages into an ordered stream of headings, paragraphs, and tables."""
    elements: list[_Element] = []
    for page in pages:
        ref = _PageRef(page.pdf_page, page.page_number, page.page_numbering)
        tables_after = _anchor_tables(page)
        paragraph: list[TextLine] = []

        def flush_paragraph() -> None:
            if paragraph:
                text = " ".join(line.text for line in paragraph)
                elements.append(_Element("paragraph", text, ref))
                paragraph.clear()

        def emit_tables(anchor: int) -> None:
            for table in tables_after.get(anchor, []):
                flush_paragraph()
                elements.append(_Element("table", table.markdown, ref))

        emit_tables(-1)
        for index, line in enumerate(page.lines):
            if _is_heading(line, body_size, config):
                flush_paragraph()
                previous = elements[-1] if elements else None
                if _continues_heading(previous, line, ref):
                    elements[-1] = _Element(
                        "heading", f"{previous.text} {line.text}", ref, previous.size, line.bbox[3]
                    )
                else:
                    elements.append(_Element("heading", line.text, ref, line.size, line.bbox[3]))
                emit_tables(index)
                continue

            if paragraph and paragraph[-1].block != line.block:
                flush_paragraph()
            paragraph.append(line)
            emit_tables(index)

        flush_paragraph()
    return elements


def _anchor_tables(page: Page) -> dict[int, list[Table]]:
    """Map each table to the index of the line it follows when reading the page.

    PDF content order is not always top-to-bottom (footers are often drawn
    first), so a table is anchored to the lowest line that ends above it and
    overlaps it horizontally. Tables with no such line come first (key -1).
    """
    anchored: dict[int, list[Table]] = {}
    for table in sorted(page.tables, key=lambda table: table.bbox[1]):
        left, top, right, _ = table.bbox
        above = [
            (line.bbox[3], index)
            for index, line in enumerate(page.lines)
            if line.bbox[3] <= top + 2 and line.bbox[0] < right and line.bbox[2] > left
        ]
        anchor = max(above)[1] if above else -1
        anchored.setdefault(anchor, []).append(table)
    return anchored


def _continues_heading(previous: _Element | None, line: TextLine, ref: _PageRef) -> bool:
    """A heading wrapped onto a second line arrives as two consecutive, adjacent heading lines."""
    return (
        previous is not None
        and previous.kind == "heading"
        and previous.page == ref
        and abs(previous.size - line.size) < 0.5
        and line.bbox[1] - previous.bottom < line.size
    )


# --- Sections ----------------------------------------------------------------


@dataclass
class _Section:
    title: str
    path: str
    elements: list[_Element]


def _sections(elements: list[_Element], level_tolerance: float) -> Iterator[_Section]:
    """Group elements under their nearest heading, tracking the heading hierarchy.

    A heading closes every open heading of the same level or below, so the
    stack always holds the chain of enclosing sections. Sizes within
    ``level_tolerance`` count as one level, because designers are not always
    consistent (e.g. 30pt and 31pt for sibling chapter titles).
    """
    stack: list[_Element] = []
    current = _Section(UNTITLED_SECTION, UNTITLED_SECTION, [])
    for element in elements:
        if element.kind != "heading":
            current.elements.append(element)
            continue
        if current.elements:
            yield current
        while stack and stack[-1].size < element.size + level_tolerance:
            stack.pop()
        stack.append(element)
        current = _Section(element.text, " > ".join(h.text for h in stack), [])
    if current.elements:
        yield current


# --- Splitting ---------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    """The smallest piece of text the splitter moves around."""

    text: str
    page: _PageRef
    starts_paragraph: bool = True


def _section_pieces(
    section: _Section, config: ChunkerConfig
) -> Iterator[tuple[ContentType, list[_Unit]]]:
    """Yield the chunks of one section: text runs (split if needed) and whole tables."""
    # Very long section paths must not starve the body of space.
    budget = max(config.max_chars - len(_header(section)), config.max_chars // 2)
    paragraphs: list[_Unit] = []
    for element in section.elements:
        if element.kind == "table":
            if paragraphs:
                yield from (("text", piece) for piece in _split(paragraphs, budget, config))
                paragraphs = []
            yield "table", [_Unit(element.text, element.page)]
        else:
            paragraphs.append(_Unit(element.text, element.page))
    if paragraphs:
        yield from (("text", piece) for piece in _split(paragraphs, budget, config))


def _split(paragraphs: list[_Unit], budget: int, config: ChunkerConfig) -> list[list[_Unit]]:
    if len(_join(paragraphs)) <= budget:
        return [paragraphs]

    units = [unit for paragraph in paragraphs for unit in _sentences(paragraph, budget)]
    pieces: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and len(_join([*current, unit])) > budget:
            pieces.append(current)
            current = _overlap(current, config.overlap_chars)
            if len(_join([*current, unit])) > budget:
                current = []
        current.append(unit)
    if current:
        pieces.append(current)
    return pieces


def _sentences(paragraph: _Unit, budget: int) -> list[_Unit]:
    """Split a paragraph into sentences; hard-wrap any sentence longer than the budget."""
    parts: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(paragraph.text):
        parts.extend(_wrap_words(sentence, budget) if len(sentence) > budget else [sentence])
    return [
        _Unit(text, paragraph.page, starts_paragraph=(index == 0))
        for index, text in enumerate(parts)
        if text
    ]


def _wrap_words(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) > width and current:
            lines.append(current)
            candidate = word
        current = candidate
    if current:
        lines.append(current)
    return lines


def _overlap(units: list[_Unit], overlap_chars: int) -> list[_Unit]:
    """The trailing whole sentences of a piece that fit within ``overlap_chars``."""
    tail: list[_Unit] = []
    for unit in reversed(units):
        if len(_join([unit, *tail])) > overlap_chars:
            break
        tail.insert(0, unit)
    return tail


def _join(units: list[_Unit]) -> str:
    text = ""
    for index, unit in enumerate(units):
        if index:
            text += "\n\n" if unit.starts_paragraph else " "
        text += unit.text
    return text


# --- Chunk assembly ----------------------------------------------------------


def _header(section: _Section) -> str:
    return f"{section.path}\n\n"


def _make_chunk(
    *,
    chunk_id: str,
    source: str,
    section: _Section,
    content_type: ContentType,
    units: list[_Unit],
) -> Chunk:
    pages = sorted({unit.page for unit in units}, key=lambda page: page.pdf_page)
    numbering_sources = {page.page_numbering for page in pages}
    return Chunk(
        chunk_id=chunk_id,
        text=_header(section) + _join(units),
        source=source,
        section_title=section.title,
        section_path=section.path,
        page_start=pages[0].page_number,
        page_end=pages[-1].page_number,
        pdf_page_start=pages[0].pdf_page,
        pdf_page_end=pages[-1].pdf_page,
        page_numbering=numbering_sources.pop() if len(numbering_sources) == 1 else "mixed",
        content_type=content_type,
    )
