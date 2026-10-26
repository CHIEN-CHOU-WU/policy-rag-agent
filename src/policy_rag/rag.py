"""Answer a question from retrieved excerpts, with page citations that are checked.

The model may only use the excerpts it is given and must cite their pages.
After it answers, every cited page is compared with the pages it was actually
shown, so a fabricated citation is reported instead of silently trusted.
"""

import re
from dataclasses import dataclass

from policy_rag.llm.base import LLMClient
from policy_rag.models import SearchResult
from policy_rag.retrieve import format_context, retrieve
from policy_rag.store.base import VectorStore

NOT_FOUND = "I couldn't find this in the provided documents."

# Shared with the agent, so both modes cite and decline in the same way.
ANSWER_RULES = f"""\
- Use only information in the excerpts. Do not add outside knowledge.
- Cite the page of every fact right after it, using the page attribute of the excerpt \
the fact came from: [p. 12] for one page, [pp. 25-26] for an excerpt that spans pages.
- Page numbers mentioned inside excerpt text (like "go to page 79") are cross-references, \
not sources. Never cite them, and don't copy them into your answer.
- If excerpts disagree or only partly answer the question, say what they do and don't cover.
- If the excerpts do not answer the question, reply exactly: "{NOT_FOUND}"
- Answer concisely, in plain language."""

SYSTEM_PROMPT = f"""\
You answer questions about an insurance policy document using only the excerpts provided.

Rules:
{ANSWER_RULES}"""

# A bracketed citation such as "[p. 9]", "[pp. 25-26]" or "[p. 9, p. 12]".
# Full-width brackets ("【p. 9】") are accepted too: some models (gpt-oss) write
# them even when asked for square brackets. The pages are checked all the same.
_CITATION = re.compile(r"[\[【](pp?\.[^\]】]*)[\]】]")
_PAGE_OR_RANGE = re.compile(r"(\d+)(?:\s*[-–]\s*(\d+))?")


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchResult]
    cited_pages: list[int]
    unsupported_pages: list[int]
    """Cited pages that none of the sources came from (likely fabricated)."""

    @property
    def cited_sources(self) -> list[SearchResult]:
        """The cited excerpts, one per page, in retrieval order.

        Citations name pages, not excerpts, so when several excerpts share a
        cited page only the best-ranked one is listed; the others may cover a
        different topic on the same page (Part A vs. Part B penalties, say).
        """
        cited = set(self.cited_pages)
        by_page: dict[str, SearchResult] = {}
        for source in self.sources:
            chunk = source.chunk
            if cited & set(range(chunk.page_start, chunk.page_end + 1)):
                by_page.setdefault(chunk.page_label, source)
        return list(by_page.values())

    @property
    def is_uncited(self) -> bool:
        """The model gave an answer but cited no page for it."""
        return not self.cited_pages and self.text != NOT_FOUND


def answer_question(question: str, store: VectorStore, llm: LLMClient, k: int = 5) -> Answer:
    sources = retrieve(store, question, k=k)
    if not sources:
        # Nothing to ground an answer in, so don't ask the model to guess.
        return Answer(text=NOT_FOUND, sources=[], cited_pages=[], unsupported_pages=[])

    prompt = f"<excerpts>\n{format_context(sources)}\n</excerpts>\n\nQuestion: {question}"
    text = llm.complete(SYSTEM_PROMPT, prompt)

    cited = cited_pages(text)
    shown = _source_pages(sources)
    return Answer(
        text=text,
        sources=sources,
        cited_pages=cited,
        unsupported_pages=[page for page in cited if page not in shown],
    )


def cited_pages(text: str) -> list[int]:
    """Every page cited in ``text``, with ranges expanded, sorted and de-duplicated."""
    pages: set[int] = set()
    for citation in _CITATION.findall(text):
        for start, end in _PAGE_OR_RANGE.findall(citation):
            first, last = int(start), int(end or start)
            pages.update(range(first, last + 1))
    return sorted(pages)


def _source_pages(sources: list[SearchResult]) -> set[int]:
    pages: set[int] = set()
    for source in sources:
        pages.update(range(source.chunk.page_start, source.chunk.page_end + 1))
    return pages
