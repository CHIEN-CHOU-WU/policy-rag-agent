"""The tools an agent can use to research the policy documents.

``PolicyTools`` holds the plain functions, shared by the agent (M3) and the
MCP server (M4). ``TOOL_SPECS`` and ``run_tool`` adapt them for a model:
they describe each tool, execute a call, and return text the model can read,
plus the pages it was shown, so citations can be checked afterwards.
"""

from dataclasses import dataclass, field
from typing import Any

from policy_rag.llm.base import ToolCall, ToolResult, ToolSpec
from policy_rag.models import PageText, SearchResult
from policy_rag.retrieve import format_context
from policy_rag.store.base import VectorStore
from policy_rag.store.page_store import PageStore

SEARCH_RESULTS = 5


@dataclass(frozen=True)
class PolicyTools:
    store: VectorStore
    pages: PageStore

    def search_policy(self, query: str, k: int = SEARCH_RESULTS) -> list[SearchResult]:
        """Find the excerpts most relevant to ``query``."""
        return self.store.search(query, k=k)

    def get_page(self, page_number: int) -> list[PageText]:
        """The full text of a page, by the page number printed in the document."""
        return self.pages.get(page_number)


TOOL_SPECS = [
    ToolSpec(
        name="search_policy",
        description=(
            "Search the policy documents and return the most relevant excerpts, each labeled "
            "with its page number and section. Search again with different wording if the "
            "excerpts don't answer the question."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in plain language.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="get_page",
        description=(
            "Read the full text of one page. Use it when an excerpt seems cut off, or you "
            "need the surrounding context, such as the rest of a table or list."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "page_number": {
                    "type": "integer",
                    "description": "The page number shown on an excerpt.",
                }
            },
            "required": ["page_number"],
            "additionalProperties": False,
        },
    ),
]


@dataclass(frozen=True)
class ToolOutcome:
    result: ToolResult
    pages_seen: dict[int, str] = field(default_factory=dict)
    """Each page the model was shown, with a label for listing it as a source."""


def run_tool(tools: PolicyTools, call: ToolCall) -> ToolOutcome:
    """Execute one tool call. Problems are reported to the model, never raised.

    A bad argument or a missing page is something the model can fix on its
    next turn, so it comes back as an error result instead of an exception.
    """
    match call.name:
        case "search_policy":
            return _search(tools, call)
        case "get_page":
            return _read_page(tools, call)
        case _:
            return _error(call, f"Unknown tool {call.name!r}. Use search_policy or get_page.")


def _search(tools: PolicyTools, call: ToolCall) -> ToolOutcome:
    query = call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _error(call, "search_policy needs a non-empty 'query' string.")

    results = tools.search_policy(query)
    if not results:
        return ToolOutcome(_result(call, "No excerpts matched this search."))
    pages_seen: dict[int, str] = {}
    for result in results:
        chunk = result.chunk
        for page in range(chunk.page_start, chunk.page_end + 1):
            pages_seen.setdefault(page, chunk.section_title)
    return ToolOutcome(_result(call, format_context(results)), pages_seen)


def _read_page(tools: PolicyTools, call: ToolCall) -> ToolOutcome:
    page_number = _as_int(call.arguments.get("page_number"))
    if page_number is None:
        return _error(call, "get_page needs an integer 'page_number'.")

    pages = tools.get_page(page_number)
    if not pages:
        return _error(call, f"There is no page {page_number}.")
    text = "\n\n".join(
        f'<page number="{page.page_number}" source="{page.source}">\n{page.text}\n</page>'
        for page in pages
    )
    return ToolOutcome(_result(call, text), {page_number: "Full page"})


def _as_int(value: Any) -> int | None:
    # Smaller models sometimes send numbers as strings ("66").
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _result(call: ToolCall, content: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=call.name, content=content)


def _error(call: ToolCall, message: str) -> ToolOutcome:
    return ToolOutcome(ToolResult(call_id=call.id, name=call.name, content=message, is_error=True))
