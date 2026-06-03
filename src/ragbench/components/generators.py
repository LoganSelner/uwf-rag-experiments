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
from urllib.request import Request, urlopen

from pydantic import Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ragbench.components._tool_protocol import parse_arguments, to_openai_tools
from ragbench.components.base import BaseGenerator, ComponentParams
from ragbench.core.config import LLMConfig
from ragbench.core.registry import registry
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

_NON_RETRYABLE = (TypeError, ValueError, KeyError, AttributeError, SyntaxError)


_retry_decorator = retry(
    retry=retry_if_exception(lambda e: not isinstance(e, _NON_RETRYABLE)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def build_generator(llm: LLMConfig) -> BaseGenerator:
    """Construct a generator from an :class:`LLMConfig` via the registry.

    The single construction path for every generator in the harness — used by
    the linear pipeline, the agent's reasoning model, and any component that
    needs an LLM (query transformers, the contextual enricher). ``llm.provider``
    is the ``generation`` registry name; ``llm.params`` carries provider-specific
    extras (EdenAI ``sub_provider``, Ollama ``base_url``) that land at the top
    level of the generator config, alongside the nested ``llm`` block each
    generator reads.
    """
    gen_cls = registry.get("generation", llm.provider)
    config: dict[str, Any] = {
        **llm.params,
        "llm": {
            "provider": llm.provider,
            "model_name": llm.model_name,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
        },
    }
    return gen_cls(config=config)


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

    class Params(ComponentParams):
        llm: LLMConfig = Field(default_factory=LLMConfig)
        base_url: str = "http://localhost:11434"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model = self.p.llm.model_name
        self._temperature = self.p.llm.temperature
        self._max_tokens = self.p.llm.max_tokens
        self._base_url = self.p.base_url

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

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata={
                "model": self._model,
                "provider": "ollama",
                "total_duration": response.get("total_duration"),
                "eval_count": response.get("eval_count"),
            },
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

    @_retry_decorator
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


@registry.register("generation", "google")
class GoogleGenerator(BaseGenerator):
    """Generates answers using Google's Gemini API.

    Wraps the ``google-genai`` SDK.

    Config params:
        llm.model_name: Gemini model (e.g. "gemini-2.0-flash")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max output tokens (optional)
    """

    class Params(ComponentParams):
        llm: LLMConfig = Field(default_factory=LLMConfig)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model = self.p.llm.model_name
        self._temperature = self.p.llm.temperature
        self._max_tokens = self.p.llm.max_tokens

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is "
                "required. Set it in .env or your shell environment."
            )

        from google import genai

        self._client: Any = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str | list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> GenerationResult:
        """Call Gemini's generate_content API and return a GenerationResult."""
        from google.genai import types

        system_instruction, contents = self._to_google_contents(prompt)

        gen_config = types.GenerateContentConfig(
            temperature=self._temperature,
        )
        if self._max_tokens is not None:
            gen_config.max_output_tokens = self._max_tokens
        if system_instruction is not None:
            gen_config.system_instruction = system_instruction
        if tools:
            gen_config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=s.name,
                            description=s.description,
                            # google-genai coerces a JSON-schema dict into its
                            # Schema type at runtime; the annotation is stricter.
                            parameters=s.parameters or None,  # type: ignore[arg-type]
                        )
                        for s in tools
                    ]
                )
            ]

        try:
            response = self._call_api(contents, gen_config)
        except Exception as e:
            raise RuntimeError(
                f"Google API call failed (model={self._model}): {e}"
            ) from e

        answer, tool_calls = self._extract_response(response)

        metadata: dict[str, Any] = {
            "model": self._model,
            "provider": "google",
        }
        if response.usage_metadata:
            metadata["prompt_tokens"] = response.usage_metadata.prompt_token_count
            metadata["completion_tokens"] = (
                response.usage_metadata.candidates_token_count
            )

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata=metadata,
        )

    @staticmethod
    def _to_google_contents(
        prompt: str | list[Message],
    ) -> tuple[str | None, list[Any]]:
        """Translate the neutral schema to Gemini Contents + system instruction.

        Gemini has no "tool" role and keys tool results by **function name**,
        not call id: an assistant turn becomes a ``model`` Content with
        ``function_call`` parts, and a tool result becomes a ``user`` Content
        with a ``function_response`` part. A forward id→name map (built from the
        preceding assistant turn) resolves each tool result's ``tool_call_id``
        back to its function name.
        """
        from google.genai import types

        if isinstance(prompt, str):
            return None, [types.Content(role="user", parts=[types.Part(text=prompt)])]

        system_instruction: str | None = None
        contents: list[Any] = []
        id_to_name: dict[str, str] = {}
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                system_instruction = content
            elif role == "assistant" and msg.get("tool_calls"):
                parts = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    id_to_name[tc.get("id", "")] = name
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=name,
                                args=parse_arguments(tc["function"]["arguments"]),
                            )
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                call_id = msg.get("tool_call_id", "")
                name = id_to_name.get(call_id, call_id)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=name,
                                    response={"result": content},
                                )
                            )
                        ],
                    )
                )
            else:
                api_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(role=api_role, parts=[types.Part(text=content)])
                )
        return system_instruction, contents

    @staticmethod
    def _extract_response(response: Any) -> tuple[str, list[ToolCall]]:
        """Pull answer text and tool calls from a Gemini response.

        Iterates parts directly (rather than ``response.text``) so a response
        that contains only ``function_call`` parts doesn't error. Gemini omits
        call ids — the agent loop synthesizes them.
        """
        answer_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = response.candidates or []
        if candidates and candidates[0].content:
            for part in candidates[0].content.parts or []:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ToolCall(id="", name=fc.name, arguments=dict(fc.args or {}))
                    )
                elif getattr(part, "text", None):
                    answer_parts.append(part.text)
        return "".join(answer_parts), tool_calls

    @_retry_decorator
    def _call_api(self, contents: list[Any], config: Any) -> Any:
        """Call Gemini with automatic retry on transient failures."""
        return self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )


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

    class Params(ComponentParams):
        llm: LLMConfig = Field(default_factory=LLMConfig)
        sub_provider: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model = self.p.llm.model_name
        self._temperature = self.p.llm.temperature
        self._max_tokens = self.p.llm.max_tokens or 1024

        self._sub_provider = self.p.sub_provider
        if not self._sub_provider:
            raise ValueError(
                "EdenAIGenerator requires 'sub_provider' in config "
                "(e.g. 'openai', 'google', 'anthropic'). "
                "Set it via generation.params.sub_provider in YAML."
            )

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

    @_retry_decorator
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

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata={
                "model": self._model,
                "provider": "edenai",
                "sub_provider": self._sub_provider,
            },
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


@registry.register("generation", "openai")
class OpenAIGenerator(BaseGenerator):
    """Generates answers using OpenAI's chat completion API.

    Uses the ``openai`` SDK directly.

    Config params:
        llm.model_name: OpenAI model (e.g. "gpt-4o", "gpt-3.5-turbo")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max tokens to generate (optional)
    """

    class Params(ComponentParams):
        llm: LLMConfig = Field(default_factory=LLMConfig)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model = self.p.llm.model_name
        self._temperature = self.p.llm.temperature
        self._max_tokens = self.p.llm.max_tokens

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in .env or your shell environment."
            )

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

        metadata: dict[str, Any] = {
            "model": self._model,
            "provider": "openai",
        }
        if response.usage:
            metadata["prompt_tokens"] = response.usage.prompt_tokens
            metadata["completion_tokens"] = response.usage.completion_tokens
            metadata["total_tokens"] = response.usage.total_tokens

        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            metadata=metadata,
        )

    @_retry_decorator
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
