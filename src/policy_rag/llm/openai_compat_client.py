"""Adapter for providers with an OpenAI-compatible chat API (Gemini today, Groq next).

The only module that imports ``openai``. The provider is chosen by ``base_url``;
nothing here calls OpenAI's own service unless pointed at it.
"""

import json
import time
from typing import Any

import openai

from policy_rag.llm.base import (
    AssistantTurn,
    LLMClient,
    LLMError,
    Message,
    ToolCall,
    ToolResults,
    ToolSpec,
    UserMessage,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Free tiers rate-limit aggressively; the SDK retries 429s with backoff.
_MAX_RETRIES = 5


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider_name: str,
        requests_per_minute: int | None = None,
    ) -> None:
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=_MAX_RETRIES)
        self._model = model
        self._provider = provider_name
        # Free tiers cap requests per minute; spacing calls out stays under the
        # cap instead of hitting it and waiting on retries.
        self._min_interval = 60 / requests_per_minute if requests_per_minute else 0.0
        self._last_request = float("-inf")

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        api_messages: list[Any] = [{"role": "system", "content": system}]
        for message in messages:
            api_messages.extend(_to_api_messages(message))

        request: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            # Deterministic output: the answer should depend only on the documents.
            "temperature": 0,
        }
        if tools:
            request["tools"] = [_to_api_tool(tool) for tool in tools]

        self._wait_for_rate_limit()
        try:
            response = self._client.chat.completions.create(**request)
        except openai.AuthenticationError as error:
            raise LLMError(f"{self._provider} rejected the API key. Check it in .env.") from error
        except openai.RateLimitError as error:
            raise LLMError(f"{self._provider} rate limit reached. Wait a minute and retry.") from error
        except openai.APIConnectionError as error:
            raise LLMError(f"Could not reach the {self._provider} API. Check your network.") from error
        except openai.APIStatusError as error:
            raise LLMError(f"{self._provider} API error {error.status_code}: {error.message}") from error

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("The model's reply was cut off at the token limit.")
        message = choice.message
        return AssistantTurn(
            text=(message.content or "").strip(),
            tool_calls=[
                ToolCall(id=call.id, name=call.function.name, arguments=_parse_arguments(call.function.arguments))
                for call in message.tool_calls or []
            ],
            # Replayed as-is: some providers attach fields (e.g. Gemini's thought
            # signatures) that must come back unchanged.
            raw=message.model_dump(exclude_none=True),
        )


    def _wait_for_rate_limit(self) -> None:
        wait = self._last_request + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


def _parse_arguments(arguments: str) -> dict[str, Any]:
    # Malformed JSON becomes empty arguments; the tool then reports what is
    # missing, and the model can retry with valid arguments.
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_api_messages(message: Message) -> list[Any]:
    match message:
        case UserMessage(text=text):
            return [{"role": "user", "content": text}]
        case AssistantTurn(raw=raw):
            return [raw]
        case ToolResults(results=results):
            # No error flag in this API, so a failure is marked in the text itself.
            return [
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": f"Error: {result.content}" if result.is_error else result.content,
                }
                for result in results
            ]
        case _:
            # Fail loudly: a silent None here surfaces later as a confusing error.
            raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _to_api_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }
