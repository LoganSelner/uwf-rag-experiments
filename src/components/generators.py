"""Generator implementations.

Each generator wraps an LLM backend and produces answers from
formatted prompts. Generators are pure LLM wrappers — they never
see raw chunks. The pipeline calls PromptTemplate.format() first.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from components.base import BaseGenerator
from core.registry import registry
from core.types import GenerationResult

logger = logging.getLogger(__name__)


@registry.register("generation", "ollama")
class OllamaGenerator(BaseGenerator):
    """Generates answers using a local Ollama instance.

    Communicates via Ollama's HTTP API (no extra dependencies).

    Config params:
        llm.model_name: Ollama model tag (e.g. "qwen2.5:32b")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (optional)
        base_url: Ollama API base URL (default: "http://localhost:11434")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        llm = self.config.get("llm", {})
        self._model: str = llm.get("model_name", "")
        self._temperature: float = llm.get("temperature", 0.0)
        self._max_tokens: int | None = llm.get("max_tokens")
        self._base_url: str = self.config.get("base_url", "http://localhost:11434")

    def generate(self, prompt: str | list[dict[str, str]]) -> GenerationResult:
        """Call Ollama's chat API and return a GenerationResult."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }
        if self._max_tokens is not None:
            payload["options"]["num_predict"] = self._max_tokens

        response = self._call_api(payload)
        answer = response.get("message", {}).get("content", "")

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            metadata={
                "model": self._model,
                "provider": "ollama",
                "total_duration": response.get("total_duration"),
                "eval_count": response.get("eval_count"),
            },
        )

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP POST to Ollama's /api/chat endpoint."""
        url = f"{self._base_url}/api/chat"
        data = json.dumps(payload).encode()
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req) as resp:
                body = resp.read().decode()
            return json.loads(body)
        except URLError as e:
            logger.error("Ollama API call failed: %s", e)
            raise RuntimeError(
                f"Failed to connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
