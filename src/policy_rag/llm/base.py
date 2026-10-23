"""The LLM port: what the application needs from a language model.

Code outside ``policy_rag.llm`` depends only on this interface, so the model
provider (Claude, a local Ollama model, ...) is a configuration choice.

Conversations are built from provider-neutral messages. Each adapter
translates them to and from its own API, including tool calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """A model call failed. Adapters translate their SDK's errors into this."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described by a JSON Schema for its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    """One model reply: text, and the tools it wants called (if any)."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    """The provider's own reply, sent back verbatim on the next call.

    Some providers attach state to a reply that must be replayed unchanged
    (Claude's thinking blocks, for example), so adapters never rebuild a
    past assistant turn from ``text`` and ``tool_calls`` alone.
    """


@dataclass(frozen=True)
class ToolResults:
    results: list[ToolResult]


Message = UserMessage | AssistantTurn | ToolResults


class LLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Continue the conversation by one model turn."""

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's text reply to a single-turn prompt, without tools."""
        return self.chat(system, [UserMessage(prompt)], tools=[]).text
