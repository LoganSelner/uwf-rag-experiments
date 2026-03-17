"""Generator implementations.

Each generator wraps an LLM backend and produces answers from
formatted prompts. Generators are pure LLM wrappers — they never
see raw chunks. The pipeline calls PromptTemplate.format() first.
"""

from __future__ import annotations

import json
import logging
import os
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


@registry.register("generation", "edenai")
class EdenAIGenerator(BaseGenerator):
    """Generates answers using Eden AI's chat gateway.

    Eden AI routes requests to a sub-provider (OpenAI, Google,
    Anthropic, etc.) so a single API key unlocks multiple backends.

    Config params:
        llm.model_name: Model name (e.g. "gpt-4o", "claude-3-5-sonnet")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (default: 1024)
        sub_provider: Eden AI sub-provider (e.g. "openai") — required
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        llm = self.config.get("llm", {})
        self._model: str = llm.get("model_name", "")
        self._temperature: float = llm.get("temperature", 0.0)
        self._max_tokens: int = llm.get("max_tokens") or 1024

        self._sub_provider: str = self.config.get("sub_provider", "")
        if not self._sub_provider:
            raise ValueError(
                "EdenAIGenerator requires 'sub_provider' in config "
                "(e.g. 'openai', 'google', 'anthropic'). "
                "Set it via generation.params.sub_provider in YAML."
            )

        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("EDENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "EDENAI_API_KEY environment variable is required. "
                "Set it in .env or your shell environment."
            )

        from langchain_community.chat_models.edenai import ChatEdenAI

        self._client: Any = ChatEdenAI(
            provider=self._sub_provider,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            edenai_api_key=api_key,  # type: ignore[arg-type]
        )

    def generate(self, prompt: str | list[dict[str, str]]) -> GenerationResult:
        """Call Eden AI's chat API and return a GenerationResult."""
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
        )

        if isinstance(prompt, str):
            messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        else:
            messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

        try:
            result = self._client.invoke(messages)
        except Exception as e:
            logger.error("Eden AI generation failed: %s", e)
            raise RuntimeError(
                f"Eden AI API call failed "
                f"(provider={self._sub_provider}, "
                f"model={self._model}): {e}"
            ) from e

        answer = (
            result.content if isinstance(result.content, str) else str(result.content)
        )

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            metadata={
                "model": self._model,
                "provider": "edenai",
                "sub_provider": self._sub_provider,
            },
        )
