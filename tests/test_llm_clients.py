"""Adapter tests with the vendor SDK replaced by a mock, so no network calls are made."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest

from policy_rag.llm.anthropic_client import AnthropicClient
from policy_rag.llm.base import LLMError
from policy_rag.llm.ollama_client import OllamaClient


def anthropic_client_returning(response: object) -> tuple[AnthropicClient, MagicMock]:
    client = AnthropicClient("claude-opus-5-5")
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = response
    client._client = sdk
    return client, sdk


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


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


def ollama_client_with(sdk: MagicMock) -> OllamaClient:
    client = OllamaClient("qwen3:8b", "http://localhost:11434")
    client._client = sdk
    return client


def test_ollama_sends_system_and_user_messages() -> None:
    sdk = MagicMock()
    sdk.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="  Covered [p. 9].\n"))

    assert ollama_client_with(sdk).complete("system", "prompt") == "Covered [p. 9]."
    request = sdk.chat.call_args.kwargs
    assert request["model"] == "qwen3:8b"
    assert request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert request["think"] is False


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
