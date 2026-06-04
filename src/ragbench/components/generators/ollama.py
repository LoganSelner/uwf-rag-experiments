"""Ollama generator — local LLM via Ollama's HTTP chat API (no extra deps)."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen

from pydantic import Field

from ragbench.components._tool_protocol import parse_arguments, to_openai_tools
from ragbench.components.generators.base import _SDKGenerator
from ragbench.core.registry import registry
from ragbench.core.retry import retry_transient
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec


@registry.register("generation", "ollama")
class OllamaGenerator(_SDKGenerator):
    """Generates answers using a local Ollama instance.

    Communicates via Ollama's HTTP API (no extra dependencies).

    Config params:
        llm.model_name: Ollama model tag (e.g. "qwen2.5:32b")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (optional)
        base_url: Ollama API base URL (default: "http://localhost:11434")
    """

    provider_label = "ollama"

    class Params(_SDKGenerator.Params):
        # No explicit base_url in config → fall back to $OLLAMA_BASE_URL, then
        # localhost. Lets a WSL→Windows host (or a remote Ollama) be set via env
        # without baking a machine-specific URL into a tracked config.
        base_url: str = Field(
            default_factory=lambda: os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_url = self.Params.model_validate(self.config).base_url

    def generate(
        self,
        prompt: str | list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> GenerationResult:
        """Call Ollama's chat API and return a GenerationResult."""
        messages = self._to_ollama_messages(prompt)

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
        if tools:
            payload["tools"] = to_openai_tools(tools)

        try:
            response = self._call_api(payload)
        except Exception as e:
            raise RuntimeError(
                f"Ollama call failed at {self._base_url} after retries: {e}"
            ) from e
        message = response.get("message", {})
        answer = message.get("content", "")
        tool_calls = self._extract_tool_calls(message)

        return self._assemble(
            answer,
            tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            total_duration=response.get("total_duration"),
            # Normalize Ollama's native counters to the cross-provider keys
            # (prompt_eval_count = prompt tokens, eval_count = completion tokens).
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

    @staticmethod
    def _to_ollama_messages(prompt: str | list[Message]) -> list[dict[str, Any]]:
        """Translate the neutral message schema to Ollama's chat shape.

        Ollama omits tool-call ids and matches results by order/name, so ids
        and ``tool_call_id`` are dropped, and assistant ``tool_calls`` carry
        ``arguments`` as a parsed object (not a JSON string). Plain
        system/user/assistant turns pass through unchanged — identical to the
        linear pipeline's payload before tool support.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        out: list[dict[str, Any]] = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "assistant" and msg.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": parse_arguments(
                                        tc["function"]["arguments"]
                                    ),
                                }
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            else:
                out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _extract_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Parse Ollama ``message.tool_calls`` into our ``ToolCall`` list.

        Ollama returns no ids (the agent loop synthesizes them) and
        ``arguments`` already as an object.
        """
        raw = message.get("tool_calls") or []
        out: list[ToolCall] = []
        for tc in raw:
            fn = tc.get("function", {})
            out.append(
                ToolCall(
                    id="",
                    name=fn.get("name", ""),
                    arguments=parse_arguments(fn.get("arguments")),
                )
            )
        return out

    @retry_transient
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
        with urlopen(req) as resp:
            body = resp.read().decode()
        return json.loads(body)
