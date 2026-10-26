"""OpenAI-compatible adapter tests with the SDK replaced by a mock (no network)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from policy_rag.config import load_settings
from policy_rag.llm.base import LLMError, ToolResult, ToolResults, ToolSpec, UserMessage
from policy_rag.llm.factory import create_llm_client
from policy_rag.llm.openai_compat_client import GEMINI_BASE_URL, OpenAICompatClient

SEARCH_TOOL = ToolSpec(
    name="search_policy",
    description="Search.",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)


class FakeMessage(SimpleNamespace):
    """Stands in for the SDK's message model, including extra provider fields."""

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {key: value for key, value in vars(self).items() if not (exclude_none and value is None)}


def client_returning(
    message: FakeMessage, finish_reason: str = "stop", requests_per_minute: int | None = None
) -> tuple[OpenAICompatClient, MagicMock]:
    client = OpenAICompatClient(
        GEMINI_BASE_URL, "test-key", "gemini-3.5-flash", "Gemini", requests_per_minute=requests_per_minute
    )
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
    )
    client._client = sdk
    return client, sdk


def tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_plain_reply_and_request_shape() -> None:
    client, sdk = client_returning(FakeMessage(role="assistant", content=" Covered [p. 9]. ", tool_calls=None))

    assert client.complete("system", "prompt") == "Covered [p. 9]."
    request = sdk.chat.completions.create.call_args.kwargs
    assert request["model"] == "gemini-3.5-flash"
    assert request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert "tools" not in request


def test_tool_call_round_trip_replays_the_raw_message() -> None:
    raw = FakeMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call("call_a", "search_policy", '{"query": "dental"}')],
        extra_content={"google": {"thought_signature": "abc"}},
    )
    client, sdk = client_returning(raw)

    turn = client.chat("system", [UserMessage("q")], [SEARCH_TOOL])
    assert turn.tool_calls[0].id == "call_a"
    assert turn.tool_calls[0].arguments == {"query": "dental"}
    assert sdk.chat.completions.create.call_args.kwargs["tools"][0]["function"]["name"] == "search_policy"

    results = ToolResults(
        [
            ToolResult("call_a", "search_policy", "excerpts"),
            ToolResult("call_b", "get_page", "There is no page 9.", is_error=True),
        ]
    )
    client.chat("system", [UserMessage("q"), turn, results], [SEARCH_TOOL])
    messages = sdk.chat.completions.create.call_args.kwargs["messages"]
    # Provider-specific fields (like a thought signature) survive the round trip.
    assert messages[2]["extra_content"] == {"google": {"thought_signature": "abc"}}
    assert "content" not in messages[2]
    assert messages[3:] == [
        {"role": "tool", "tool_call_id": "call_a", "content": "excerpts"},
        {"role": "tool", "tool_call_id": "call_b", "content": "Error: There is no page 9."},
    ]


@pytest.mark.parametrize("arguments", ["{not json", '"a string"', ""])
def test_malformed_arguments_become_empty(arguments: str) -> None:
    message = FakeMessage(role="assistant", content=None, tool_calls=[tool_call("c", "get_page", arguments)])
    client, _ = client_returning(message)
    assert client.chat("system", [UserMessage("q")], [SEARCH_TOOL]).tool_calls[0].arguments == {}


def test_truncated_reply_raises() -> None:
    client, _ = client_returning(FakeMessage(role="assistant", content="Cut", tool_calls=None), "length")
    with pytest.raises(LLMError, match="cut off"):
        client.complete("system", "prompt")


def test_gemini_without_a_key_has_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        create_llm_client(load_settings())


def test_calls_are_spaced_to_the_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr("time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    client, _ = client_returning(
        FakeMessage(role="assistant", content="hi", tool_calls=None), requests_per_minute=5
    )
    client.complete("system", "one")
    clock["now"] += 3.0
    client.complete("system", "two")

    assert sleeps == [9.0]


def test_groq_without_a_key_has_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        create_llm_client(load_settings())
