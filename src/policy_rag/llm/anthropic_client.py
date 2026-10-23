"""Claude adapter for the ``LLMClient`` port. The only module that imports ``anthropic``."""

from typing import Any

import anthropic

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

# Answering from supplied excerpts needs little reasoning, so a low effort
# level keeps latency and cost down without hurting quality.
_EFFORT = "low"
_MAX_TOKENS = 16000
# Re-run a request that a safety classifier declines on Anthropic's
# recommended fallback model, instead of returning the refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    def __init__(self, model: str) -> None:
        # Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth` profile).
        self._client = anthropic.Anthropic()
        self._model = model

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "system": system,
            "messages": [_to_api_message(message) for message in messages],
            "output_config": {"effort": _EFFORT},
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            request["tools"] = [_to_api_tool(tool) for tool in tools]

        try:
            response = self._client.beta.messages.create(**request)
        except TypeError as error:
            # The SDK raises TypeError at request time when no credentials can be resolved.
            raise LLMError("No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env.") from error
        except anthropic.AuthenticationError as error:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from error
        except anthropic.APIConnectionError as error:
            raise LLMError("Could not reach the Anthropic API. Check your network.") from error
        except anthropic.APIStatusError as error:
            raise LLMError(f"Anthropic API error {error.status_code}: {error.message}") from error

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model's reply was cut off at the token limit.")

        return AssistantTurn(
            text="".join(block.text for block in response.content if block.type == "text").strip(),
            tool_calls=[
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                for block in response.content
                if block.type == "tool_use"
            ],
            # Replayed as-is so thinking blocks stay intact across tool rounds.
            raw=response.content,
        )


def _to_api_message(message: Message) -> dict[str, Any]:
    match message:
        case UserMessage(text=text):
            return {"role": "user", "content": text}
        case AssistantTurn(raw=raw):
            return {"role": "assistant", "content": raw}
        case ToolResults(results=results):
            # All results of one turn go back together, in a single user message.
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in results
                ],
            }
        case _:
            # Fail loudly: a silent None here surfaces later as a confusing error.
            raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _to_api_tool(tool: ToolSpec) -> dict[str, Any]:
    # strict: arguments are guaranteed to match the schema.
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "strict": True,
    }
