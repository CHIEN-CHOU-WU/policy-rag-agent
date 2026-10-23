"""Ollama adapter for the ``LLMClient`` port. The only module that imports ``ollama``."""

from typing import Any

import ollama

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


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str) -> None:
        self._client = ollama.Client(host=host)
        self._model = model
        self._host = host

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        api_messages: list[Any] = [{"role": "system", "content": system}]
        for message in messages:
            api_messages.extend(_to_api_messages(message))

        try:
            response = self._client.chat(
                model=self._model,
                messages=api_messages,
                tools=[_to_api_tool(tool) for tool in tools] or None,
                # Reasoning models (e.g. qwen3) would otherwise mix their thinking into the reply.
                think=False,
                # Deterministic output: the answer should depend only on the documents.
                options={"temperature": 0},
            )
        except ConnectionError as error:
            raise LLMError(
                f"Could not reach Ollama at {self._host}. Is it running (`ollama serve`)?"
            ) from error
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise LLMError(
                    f"Ollama model {self._model!r} is not installed. Run `ollama pull {self._model}`."
                ) from error
            raise LLMError(f"Ollama error {error.status_code}: {error.error}") from error

        message = response.message
        return AssistantTurn(
            text=(message.content or "").strip(),
            # Ollama doesn't identify tool calls, so give each one a stable id.
            tool_calls=[
                ToolCall(id=f"call_{index}", name=call.function.name, arguments=dict(call.function.arguments))
                for index, call in enumerate(message.tool_calls or [])
            ],
            raw=message,
        )


def _to_api_messages(message: Message) -> list[Any]:
    match message:
        case UserMessage(text=text):
            return [{"role": "user", "content": text}]
        case AssistantTurn(raw=raw):
            return [raw]
        case ToolResults(results=results):
            # Ollama matches results to calls by tool name and order, one message each.
            # It has no error flag, so a failure is marked in the text itself.
            return [
                {
                    "role": "tool",
                    "tool_name": result.name,
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
