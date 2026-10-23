"""Ollama adapter for the ``LLMClient`` port. The only module that imports ``ollama``."""

import ollama

from policy_rag.llm.base import LLMClient, LLMError


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str) -> None:
        self._client = ollama.Client(host=host)
        self._model = model
        self._host = host

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # Reasoning models (e.g. qwen3) would otherwise mix their thinking into the reply.
                think=False,
                # Deterministic output: the answer should depend only on the excerpts.
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

        return (response.message.content or "").strip()
