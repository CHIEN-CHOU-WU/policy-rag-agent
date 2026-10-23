"""Builds the configured ``LLMClient``. The one place that picks a model provider."""

from policy_rag.config import Settings
from policy_rag.llm.base import LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    # Adapters are imported here so only the chosen provider's SDK is loaded.
    match settings.llm_provider:
        case "anthropic":
            from policy_rag.llm.anthropic_client import AnthropicClient

            return AnthropicClient(settings.anthropic_model)
        case "ollama":
            from policy_rag.llm.ollama_client import OllamaClient

            return OllamaClient(settings.ollama_model, settings.ollama_host)
        case other:
            raise ValueError(f"Unknown LLM_PROVIDER {other!r}; expected 'anthropic' or 'ollama'.")
