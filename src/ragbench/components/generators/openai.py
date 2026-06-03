"""OpenAI generator — uses the ``openai`` SDK directly."""

from __future__ import annotations

from typing import Any

from ragbench.components._tool_protocol import parse_arguments, to_openai_tools
from ragbench.components.generators.base import _SDKGenerator
from ragbench.core.registry import registry
from ragbench.core.retry import retry_transient
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec


@registry.register("generation", "openai")
class OpenAIGenerator(_SDKGenerator):
    """Generates answers using OpenAI's chat completion API.

    Uses the ``openai`` SDK directly.

    Config params:
        llm.model_name: OpenAI model (e.g. "gpt-4o", "gpt-3.5-turbo")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (optional)
    """

    provider_label = "openai"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        api_key = self._require_env("OPENAI_API_KEY")

        import openai

        self._client: Any = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str | list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> GenerationResult:
        """Call OpenAI's chat completion API and return a GenerationResult."""
        if isinstance(prompt, str):
            messages: list[Message] = [{"role": "user", "content": prompt}]
        else:
            # The neutral schema is already OpenAI's chat shape (assistant
            # tool_calls + tool-role results), so it passes through verbatim.
            messages = prompt

        try:
            response = self._call_api(messages, tools)
        except Exception as e:
            raise RuntimeError(
                f"OpenAI API call failed (model={self._model}): {e}"
            ) from e

        choice = response.choices[0]
        answer = choice.message.content or ""
        tool_calls = self._extract_tool_calls(choice.message)

        extra: dict[str, Any] = {}
        if response.usage:
            extra["prompt_tokens"] = response.usage.prompt_tokens
            extra["completion_tokens"] = response.usage.completion_tokens
            extra["total_tokens"] = response.usage.total_tokens

        return self._assemble(
            answer,
            tool_calls,
            finish_reason=choice.finish_reason or "",
            **extra,
        )

    @retry_transient
    def _call_api(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Any:
        """Call OpenAI chat completions with automatic retry."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[ToolCall]:
        """Parse OpenAI ``message.tool_calls`` into our ``ToolCall`` list."""
        raw = getattr(message, "tool_calls", None) or []
        return [
            ToolCall(
                id=tc.id or "",
                name=tc.function.name,
                arguments=parse_arguments(tc.function.arguments),
            )
            for tc in raw
        ]
