"""Eden AI generator — cloud LLM gateway via ``langchain-community``'s ChatEdenAI."""

from __future__ import annotations

from typing import Any

from ragbench.components._tool_protocol import parse_arguments, to_openai_tools
from ragbench.components.generators.base import _SDKGenerator
from ragbench.core.registry import registry
from ragbench.core.retry import retry_transient
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec


@registry.register("generation", "edenai")
class EdenAIGenerator(_SDKGenerator):
    """Generates answers using Eden AI's chat gateway.

    Eden AI routes requests to a sub-provider (OpenAI, Google,
    Anthropic, etc.) so a single API key unlocks multiple backends.

    Config params:
        llm.model_name: Model name (e.g. "gpt-4o", "claude-3-5-sonnet")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (default: 1024)
        sub_provider: Eden AI sub-provider (e.g. "openai") — required
    """

    provider_label = "edenai"

    class Params(_SDKGenerator.Params):
        sub_provider: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Eden AI requires an explicit max_tokens; default to 1024 when unset.
        self._max_tokens = self._max_tokens or 1024

        self._sub_provider = self.Params.model_validate(self.config).sub_provider
        if not self._sub_provider:
            raise ValueError(
                "EdenAIGenerator requires 'sub_provider' in config "
                "(e.g. 'openai', 'google', 'anthropic'). "
                "Set it via generation.params.sub_provider in YAML."
            )

        api_key = self._require_env("EDENAI_API_KEY")

        from langchain_community.chat_models.edenai import ChatEdenAI

        self._client: Any = ChatEdenAI(
            provider=self._sub_provider,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            edenai_api_key=api_key,  # type: ignore[arg-type]
        )

    @retry_transient
    def _invoke_with_retry(
        self, messages: list[Any], tools: list[ToolSpec] | None = None
    ) -> Any:
        """Call Eden AI with automatic retry on transient failures.

        When ``tools`` is provided, bind them so the model may emit tool calls
        (Eden AI runs the OpenAI-format schema through ``convert_to_openai_tool``).
        ``tool_choice`` is left at its default so the model decides whether to
        call a tool — the ReAct loop wants that latitude.
        """
        client = self._client
        if tools:
            client = client.bind_tools(to_openai_tools(tools))
        return client.invoke(messages)

    def generate(
        self,
        prompt: str | list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> GenerationResult:
        """Call Eden AI's chat API and return a GenerationResult."""
        messages = self._to_langchain_messages(prompt)

        try:
            result = self._invoke_with_retry(messages, tools)
        except Exception as e:
            raise RuntimeError(
                f"Eden AI API call failed after retries "
                f"(provider={self._sub_provider}, "
                f"model={self._model}): {e}"
            ) from e

        answer = (
            result.content if isinstance(result.content, str) else str(result.content)
        )
        tool_calls = self._extract_tool_calls(result)

        return self._assemble(
            answer,
            tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            sub_provider=self._sub_provider,
        )

    @staticmethod
    def _to_langchain_messages(prompt: str | list[Message]) -> list[Any]:
        """Translate the neutral message schema to LangChain message objects.

        Eden AI's converter is keyed on LangChain types: assistant tool calls
        ride on ``AIMessage.tool_calls`` (``{name, args, id}`` with ``args`` a
        parsed dict) and tool results on ``ToolMessage.tool_call_id``. The
        system message stays at index 0 and the latest ``HumanMessage`` is the
        live query, so tool results are appended as ToolMessages without
        injecting synthetic human turns.
        """
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        if isinstance(prompt, str):
            return [HumanMessage(content=prompt)]

        messages: list[BaseMessage] = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                raw_calls = msg.get("tool_calls")
                if raw_calls:
                    messages.append(
                        AIMessage(
                            content=content,
                            tool_calls=[
                                {
                                    "name": tc["function"]["name"],
                                    "args": parse_arguments(
                                        tc["function"]["arguments"]
                                    ),
                                    "id": tc.get("id", ""),
                                    "type": "tool_call",
                                }
                                for tc in raw_calls
                            ],
                        )
                    )
                else:
                    messages.append(AIMessage(content=content))
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
            else:
                messages.append(HumanMessage(content=content))
        return messages

    @staticmethod
    def _extract_tool_calls(result: Any) -> list[ToolCall]:
        """Map LangChain ``AIMessage.tool_calls`` to our ``ToolCall`` list."""
        raw = getattr(result, "tool_calls", None) or []
        return [
            ToolCall(
                id=tc.get("id") or "",
                name=tc.get("name", ""),
                arguments=tc.get("args") or {},
            )
            for tc in raw
        ]
