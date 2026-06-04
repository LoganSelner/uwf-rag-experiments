"""Google Gemini generator — wraps the ``google-genai`` SDK."""

from __future__ import annotations

from typing import Any

from ragbench.components._tool_protocol import parse_arguments
from ragbench.components.generators.base import _SDKGenerator
from ragbench.core.registry import registry
from ragbench.core.retry import retry_transient
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec


@registry.register("generation", "google")
class GoogleGenerator(_SDKGenerator):
    """Generates answers using Google's Gemini API.

    Wraps the ``google-genai`` SDK.

    Config params:
        llm.model_name: Gemini model (e.g. "gemini-2.0-flash")
        llm.temperature: Sampling temperature (default: 0.0)
        llm.max_tokens: Max output tokens (optional)
    """

    provider_label = "google"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        api_key = self._require_env("GOOGLE_API_KEY", "GEMINI_API_KEY")

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

        extra: dict[str, Any] = {}
        if response.usage_metadata:
            extra["prompt_tokens"] = response.usage_metadata.prompt_token_count
            extra["completion_tokens"] = response.usage_metadata.candidates_token_count

        return self._assemble(
            answer,
            tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            **extra,
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

    @retry_transient
    def _call_api(self, contents: list[Any], config: Any) -> Any:
        """Call Gemini with automatic retry on transient failures."""
        return self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
