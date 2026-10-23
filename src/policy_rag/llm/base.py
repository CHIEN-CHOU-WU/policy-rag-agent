"""The LLM port: what the application needs from a language model.

Code outside ``policy_rag.llm`` depends only on this interface, so the model
provider (Claude, a local Ollama model, ...) is a configuration choice.
"""

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """A model call failed. Adapters translate their SDK's errors into this."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Return the model's text reply to a single-turn prompt."""
