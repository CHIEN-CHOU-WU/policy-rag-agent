"""Adapter tests with the vendor SDK replaced by a mock, so no network calls are made."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest

from policy_rag.llm.anthropic_client import AnthropicClient
from policy_rag.llm.base import (
    LLMError,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserMessage,
)
from policy_rag.llm.ollama_client import OllamaClient


def anthropic_client_returning(response: object) -> tuple[AnthropicClient, MagicMock]:
    client = AnthropicClient("claude-opus-5-5")
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = response
    client._client = sdk
    return client, sdk


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


SEARCH_TOOL = ToolSpec(
    name="search_policy",
    description="Search.",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)


def test_anthropic_joins_text_blocks_and_skips_thinking() -> None:
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), text_block("Covered "), text_block("[p. 9].")],
    )
    client, sdk = anthropic_client_returning(response)

    assert client.complete("system", "prompt") == "Covered [p. 9]."
    request = sdk.beta.messages.create.call_args.kwargs
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == "system"
    assert request["messages"] == [{"role": "user", "content": "prompt"}]
    assert request["fallbacks"] == "default"


def test_anthropic_refusal_raises() -> None:
    client, _ = anthropic_client_returning(SimpleNamespace(stop_reason="refusal", content=[]))
    with pytest.raises(LLMError, match="declined"):
        client.complete("system", "prompt")


def test_anthropic_missing_credentials_has_a_clear_message() -> None:
    client, sdk = anthropic_client_returning(None)
    sdk.beta.messages.create.side_effect = TypeError("Could not resolve authentication method.")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        client.complete("system", "prompt")


def test_anthropic_tool_use_round_trip() -> None:
    raw_content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="tool_use", id="toolu_1", name="search_policy", input={"query": "dental"}),
    ]
    client, sdk = anthropic_client_returning(SimpleNamespace(stop_reason="tool_use", content=raw_content))

    turn = client.chat("system", [UserMessage("q")], [SEARCH_TOOL])
    assert turn.tool_calls == [ToolCall("toolu_1", "search_policy", {"query": "dental"})]
    assert sdk.beta.messages.create.call_args.kwargs["tools"] == [
        {
            "name": "search_policy",
            "description": "Search.",
            "input_schema": SEARCH_TOOL.input_schema,
            "strict": True,
        }
    ]

    results = ToolResults([ToolResult("toolu_1", "search_policy", "excerpts", is_error=False)])
    client.chat("system", [UserMessage("q"), turn, results], [SEARCH_TOOL])
    assistant, tool_message = sdk.beta.messages.create.call_args.kwargs["messages"][1:]
    # The assistant turn is replayed verbatim, thinking block included.
    assert assistant == {"role": "assistant", "content": raw_content}
    assert tool_message == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "excerpts", "is_error": False}
        ],
    }


def test_anthropic_omits_tools_when_there_are_none() -> None:
    client, sdk = anthropic_client_returning(SimpleNamespace(stop_reason="end_turn", content=[text_block("hi")]))
    client.complete("system", "prompt")
    assert "tools" not in sdk.beta.messages.create.call_args.kwargs


def test_anthropic_truncated_reply_raises() -> None:
    client, _ = anthropic_client_returning(SimpleNamespace(stop_reason="max_tokens", content=[]))
    with pytest.raises(LLMError, match="cut off"):
        client.complete("system", "prompt")


def ollama_client_with(sdk: MagicMock) -> OllamaClient:
    client = OllamaClient("qwen3:8b", "http://localhost:11434")
    client._client = sdk
    return client


def test_ollama_sends_system_and_user_messages() -> None:
    sdk = MagicMock()
    sdk.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="  Covered [p. 9].\n", tool_calls=None)
    )

    assert ollama_client_with(sdk).complete("system", "prompt") == "Covered [p. 9]."
    request = sdk.chat.call_args.kwargs
    assert request["model"] == "qwen3:8b"
    assert request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert request["think"] is False


def test_ollama_tool_calls_get_ids_and_results_go_back_as_tool_messages() -> None:
    raw_message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name="search_policy", arguments={"query": "dental"})),
            SimpleNamespace(function=SimpleNamespace(name="get_page", arguments={"page_number": 58})),
        ],
    )
    sdk = MagicMock()
    sdk.chat.return_value = SimpleNamespace(message=raw_message)
    client = ollama_client_with(sdk)

    turn = client.chat("system", [UserMessage("q")], [SEARCH_TOOL])
    assert [call.id for call in turn.tool_calls] == ["call_0", "call_1"]
    assert sdk.chat.call_args.kwargs["tools"][0]["function"]["name"] == "search_policy"

    results = ToolResults(
        [
            ToolResult("call_0", "search_policy", "excerpts"),
            ToolResult("call_1", "get_page", "There is no page 58.", is_error=True),
        ]
    )
    client.chat("system", [UserMessage("q"), turn, results], [SEARCH_TOOL])
    messages = sdk.chat.call_args.kwargs["messages"]
    assert messages[2] is raw_message
    assert messages[3:] == [
        {"role": "tool", "tool_name": "search_policy", "content": "excerpts"},
        {"role": "tool", "tool_name": "get_page", "content": "Error: There is no page 58."},
    ]


def test_ollama_without_tools_sends_none() -> None:
    sdk = MagicMock()
    sdk.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))
    ollama_client_with(sdk).complete("system", "prompt")
    assert sdk.chat.call_args.kwargs["tools"] is None


def test_ollama_not_running_has_a_clear_message() -> None:
    sdk = MagicMock()
    sdk.chat.side_effect = ConnectionError("Failed to connect to Ollama.")
    with pytest.raises(LLMError, match="ollama serve"):
        ollama_client_with(sdk).complete("system", "prompt")


def test_ollama_missing_model_suggests_pulling_it() -> None:
    sdk = MagicMock()
    sdk.chat.side_effect = ollama.ResponseError("model not found", status_code=404)
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        ollama_client_with(sdk).complete("system", "prompt")
