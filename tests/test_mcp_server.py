"""MCP server tests over an in-process client: no subprocess, no network."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import Client

from fakes import FakePageStore, FakeStore, make_chunk, make_page
from policy_rag.mcp_server import EMPTY_INDEX, create_server
from policy_rag.tools import NO_RESULTS, PolicyTools, format_pages, format_search_results

DENTAL = make_chunk("c1", "What isn't covered", "Most dental care isn't covered.", 58, 58)
PAGE_58 = make_page(58, "Full text of page 58.")


def with_client(tools: PolicyTools, use: Callable[[Client], Awaitable[Any]]) -> Any:
    async def run() -> Any:
        async with Client(create_server(tools)) as client:
            return await use(client)

    return asyncio.run(run())


def call(tools: PolicyTools, name: str, arguments: dict[str, Any]) -> Any:
    return with_client(tools, lambda client: client.call_tool(name, arguments))


TOOLS = PolicyTools(FakeStore([DENTAL]), FakePageStore([PAGE_58]))


def test_lists_both_tools_with_described_parameters() -> None:
    listed = with_client(TOOLS, lambda client: client.list_tools())
    schemas = {tool.name: tool.input_schema for tool in listed.tools}

    assert schemas["search_policy"]["required"] == ["query"]
    assert schemas["search_policy"]["properties"]["query"]["type"] == "string"
    assert schemas["get_page"]["properties"]["page_number"]["type"] == "integer"
    assert all(tool.description for tool in listed.tools)


def test_search_returns_the_same_text_the_agent_sees() -> None:
    result = call(TOOLS, "search_policy", {"query": "dental"})

    assert not result.is_error
    assert result.content[0].text == format_search_results(TOOLS.search_policy("dental"))
    assert '<excerpt page="58"' in result.content[0].text


def test_search_with_no_matches_is_not_an_error() -> None:
    # The index isn't empty; this particular query just finds nothing.
    tools = PolicyTools(FakeStore([DENTAL], by_query={"passport": []}), FakePageStore([]))

    result = call(tools, "search_policy", {"query": "passport"})
    assert not result.is_error
    assert result.content[0].text == NO_RESULTS


def test_get_page_returns_the_full_page() -> None:
    result = call(TOOLS, "get_page", {"page_number": 58})
    assert not result.is_error
    assert result.content[0].text == format_pages([PAGE_58])


def test_missing_page_is_a_readable_error() -> None:
    result = call(TOOLS, "get_page", {"page_number": 999})
    assert result.is_error
    assert "There is no page 999." in result.content[0].text


def test_blank_query_is_a_readable_error() -> None:
    result = call(TOOLS, "search_policy", {"query": "  "})
    assert result.is_error
    assert "non-empty query" in result.content[0].text


def test_empty_index_tells_the_user_to_run_ingest() -> None:
    result = call(PolicyTools(FakeStore([]), FakePageStore([])), "search_policy", {"query": "dental"})
    assert result.is_error
    assert EMPTY_INDEX in result.content[0].text
