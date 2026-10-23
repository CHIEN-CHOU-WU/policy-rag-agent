"""MCP server exposing the policy tools: ``uv run python -m policy_rag.mcp_server``.

Any MCP client (Claude Code, Claude Desktop, other agents) can call
``search_policy`` and ``get_page``. The client's own model is the agent here,
so this server only provides tools; it never calls a language model itself.

The server talks MCP over stdin/stdout, so nothing may be printed to stdout.
Diagnostics go to stderr.
"""

import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from policy_rag.config import load_settings
from policy_rag.store.factory import create_page_store, create_vector_store
from policy_rag.tools import (
    TOOL_SPECS,
    PolicyTools,
    format_pages,
    format_search_results,
    no_page_message,
)

EMPTY_INDEX = "The document index is empty. Run `uv run python -m policy_rag.ingest` first."

INSTRUCTIONS = """\
Tools for answering questions about an insurance policy document (Medicare & You).
Answer only from what the tools return, and cite the page of every fact as [p. 12]. \
Page numbers mentioned inside excerpt text (like "go to page 79") are cross-references, \
not sources. If the documents don't answer the question, say so instead of guessing."""

# Descriptions come from the agent's tool specs, so both front ends describe
# the tools identically.
_SPECS = {spec.name: spec for spec in TOOL_SPECS}


def _describe(tool: str, parameter: str) -> str:
    return _SPECS[tool].input_schema["properties"][parameter]["description"]


def create_server(tools: PolicyTools) -> MCPServer:
    server = MCPServer("policy-search", instructions=INSTRUCTIONS)

    # Plain-text results: the excerpts are already formatted for a model to read.
    @server.tool(description=_SPECS["search_policy"].description, structured_output=False)
    def search_policy(
        query: Annotated[str, Field(description=_describe("search_policy", "query"))],
    ) -> str:
        if not query.strip():
            raise ToolError("search_policy needs a non-empty query.")
        _require_index(tools)
        return format_search_results(tools.search_policy(query))

    @server.tool(description=_SPECS["get_page"].description, structured_output=False)
    def get_page(
        page_number: Annotated[int, Field(description=_describe("get_page", "page_number"))],
    ) -> str:
        _require_index(tools)
        pages = tools.get_page(page_number)
        if not pages:
            raise ToolError(no_page_message(page_number))
        return format_pages(pages)

    return server


def _require_index(tools: PolicyTools) -> None:
    # ToolError reaches the model as a readable error; any other exception
    # would arrive only as a generic "Error executing tool".
    if tools.store.count() == 0:
        raise ToolError(EMPTY_INDEX)


def main() -> None:
    settings = load_settings()
    tools = PolicyTools(create_vector_store(settings), create_page_store(settings))
    if tools.store.count() == 0:
        print(f"Warning: {EMPTY_INDEX}", file=sys.stderr)
    create_server(tools).run()


if __name__ == "__main__":
    main()
