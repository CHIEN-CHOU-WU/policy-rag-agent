"""Builds the configured ``LLMClient``. The one place that picks a model provider."""

from policy_rag.config import Settings
from policy_rag.llm.base import LLMClient, LLMError

PROVIDERS = ("ollama", "groq", "gemini", "anthropic")


def create_llm_client(settings: Settings) -> LLMClient:
    # Adapters are imported here so only the chosen provider's SDK is loaded.
    match settings.llm_provider:
        case "anthropic":
            from policy_rag.llm.anthropic_client import AnthropicClient

            return AnthropicClient(settings.anthropic_model)
        case "ollama":
            from policy_rag.llm.ollama_client import OllamaClient

            return OllamaClient(settings.ollama_model, settings.ollama_host)
        case "gemini":
            from policy_rag.llm.openai_compat_client import GEMINI_BASE_URL, OpenAICompatClient

            if not settings.gemini_api_key:
                raise LLMError("No Gemini API key found. Set GEMINI_API_KEY in .env.")
            return OpenAICompatClient(
                GEMINI_BASE_URL,
                settings.gemini_api_key,
                settings.gemini_model,
                "Gemini",
                requests_per_minute=settings.gemini_rpm,
            )
        case "groq":
            from policy_rag.llm.openai_compat_client import GROQ_BASE_URL, OpenAICompatClient

            if not settings.groq_api_key:
                raise LLMError("No Groq API key found. Set GROQ_API_KEY in .env.")
            return OpenAICompatClient(
                GROQ_BASE_URL,
                settings.groq_api_key,
                settings.groq_model,
                "Groq",
                requests_per_minute=settings.groq_rpm,
            )
        case other:
            raise ValueError(f"Unknown LLM_PROVIDER {other!r}; expected one of {', '.join(PROVIDERS)}.")
