"""Claude adapter for the ``LLMClient`` port. The only module that imports ``anthropic``."""

import anthropic

from policy_rag.llm.base import LLMClient, LLMError

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

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": _EFFORT},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
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
        return "".join(block.text for block in response.content if block.type == "text").strip()
