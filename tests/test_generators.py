"""Tests for the ragbench.components.generators package — Ollama, Google, EdenAI,
and OpenAI generators."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragbench.components._tool_protocol import to_openai_tools
from ragbench.components.generators import (
    EdenAIGenerator,
    GoogleGenerator,
    OllamaGenerator,
    OpenAIGenerator,
)
from ragbench.core.types import GenerationResult, Message, ToolCall, ToolSpec


def _tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="knowledge_base",
            description="Search the knowledge base.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]


def _google_response(
    text: str = "",
    function_calls: list[tuple[str, dict]] | None = None,
    usage: object | None = None,
) -> MagicMock:
    """Build a mock google-genai response with text and/or function_call parts.

    Mirrors the real shape (``candidates[0].content.parts`` with ``.text`` or
    ``.function_call``) so the part-iterating extractor is exercised properly.
    """
    parts = []
    if text:
        p = MagicMock()
        p.function_call = None
        p.text = text
        parts.append(p)
    for name, args in function_calls or []:
        p = MagicMock()
        fc = MagicMock()
        fc.name = name
        fc.args = args
        p.function_call = fc
        p.text = None
        parts.append(p)
    candidate = MagicMock()
    candidate.content.parts = parts
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    return resp


# -----------------------------------------------------------------------
# OllamaGenerator
# -----------------------------------------------------------------------


class TestOllamaGenerator:
    def _make_generator(self, **overrides) -> OllamaGenerator:
        config = {
            "llm": {"model_name": "qwen2.5:32b", "temperature": 0.0},
            "base_url": "http://localhost:11434",
            **overrides,
        }
        return OllamaGenerator(config)

    def test_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        gen = OllamaGenerator()
        assert gen._model == ""
        assert gen._temperature == 0.0
        assert gen._max_tokens is None
        assert gen._base_url == "http://localhost:11434"

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No explicit base_url → fall back to $OLLAMA_BASE_URL.
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://172.30.32.1:11434")
        gen = OllamaGenerator({"llm": {"model_name": "m"}})
        assert gen._base_url == "http://172.30.32.1:11434"

    def test_explicit_base_url_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        gen = OllamaGenerator(
            {"llm": {"model_name": "m"}, "base_url": "http://explicit:11434"}
        )
        assert gen._base_url == "http://explicit:11434"

    @patch.object(OllamaGenerator, "_call_api")
    def test_string_prompt(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {
            "message": {"content": "The answer is 42."},
            "total_duration": 1000,
            "eval_count": 10,
        }
        gen = self._make_generator()
        result = gen.generate("What is the answer?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is 42."
        # String prompt gets wrapped in a user message
        call_payload = mock_api.call_args[0][0]
        assert call_payload["messages"] == [
            {"role": "user", "content": "What is the answer?"}
        ]

    @patch.object(OllamaGenerator, "_call_api")
    def test_message_list_prompt(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {"message": {"content": "Response"}}
        gen = self._make_generator()
        messages: list[Message] = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        gen.generate(messages)
        call_payload = mock_api.call_args[0][0]
        assert call_payload["messages"] == messages

    @patch.object(OllamaGenerator, "_call_api")
    def test_metadata_populated(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {
            "message": {"content": "ans"},
            "total_duration": 5000,
            "eval_count": 20,
        }
        gen = self._make_generator()
        result = gen.generate("Q?")
        assert result.metadata["model"] == "qwen2.5:32b"
        assert result.metadata["provider"] == "ollama"
        assert result.metadata["total_duration"] == 5000
        assert result.metadata["eval_count"] == 20

    @patch.object(OllamaGenerator, "_call_api")
    def test_api_error_raises_runtime(self, mock_api: MagicMock) -> None:
        mock_api.side_effect = OSError("Connection refused")
        gen = self._make_generator()
        with pytest.raises(RuntimeError, match="Ollama call failed"):
            gen.generate("Q?")

    def test_max_tokens_in_payload(self) -> None:
        gen = self._make_generator(
            llm={"model_name": "m", "temperature": 0.0, "max_tokens": 256}
        )
        assert gen._max_tokens == 256


# -----------------------------------------------------------------------
# GoogleGenerator
# -----------------------------------------------------------------------


class TestGoogleGenerator:
    def _make_generator(self, **overrides: object) -> GoogleGenerator:
        config: dict = {
            "llm": {"model_name": "gemini-2.0-flash", "temperature": 0.0},
            **overrides,
        }
        return GoogleGenerator(config)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_config_defaults(self, mock_client_cls: MagicMock) -> None:
        gen = GoogleGenerator()
        assert gen._model == ""
        assert gen._temperature == 0.0
        assert gen._max_tokens is None

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_string_prompt(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _google_response(
            text="The answer is 42."
        )

        gen = self._make_generator()
        result = gen.generate("What is the answer?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is 42."
        # String prompt should produce a single user content part
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_message_list_prompt(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        gen = self._make_generator()
        messages: list[Message] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        gen.generate(messages)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        assert len(contents) == 3
        # "assistant" should be mapped to "model"
        assert contents[1].role == "model"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_system_instruction_extracted(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        gen = self._make_generator()
        messages: list[Message] = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        gen.generate(messages)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        # System message should NOT be in contents
        assert len(contents) == 1
        assert contents[0].role == "user"
        # System instruction should be in config
        config = call_args.kwargs["config"]
        assert config.system_instruction == "Be helpful."

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_metadata_populated(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "ans"
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_response.usage_metadata = mock_usage
        mock_client.models.generate_content.return_value = mock_response

        gen = self._make_generator()
        result = gen.generate("Q?")

        assert result.metadata["model"] == "gemini-2.0-flash"
        assert result.metadata["provider"] == "google"
        assert result.metadata["prompt_tokens"] == 10
        assert result.metadata["completion_tokens"] == 20

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_api_error_raises_runtime(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = OSError("Connection refused")

        gen = self._make_generator()
        with pytest.raises(RuntimeError, match="Google API call failed"):
            gen.generate("Q?")

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False)
    @patch("google.genai.Client")
    def test_missing_api_key_raises(self, mock_client_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleGenerator()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_max_tokens_passed(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "ans"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        gen = self._make_generator(
            llm={"model_name": "m", "temperature": 0.0, "max_tokens": 256}
        )
        gen.generate("Q?")

        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config.max_output_tokens == 256


# -----------------------------------------------------------------------
# EdenAIGenerator — constructor validation only (no real API calls)
# -----------------------------------------------------------------------


class TestEdenAIGeneratorInit:
    def test_missing_sub_provider_raises(self) -> None:
        # The sub_provider check precedes any env/key access, so no env patch
        # is needed — construction fails before EDENAI_API_KEY is read.
        with pytest.raises(ValueError, match="sub_provider"):
            EdenAIGenerator({"llm": {"model_name": "gpt-4o"}})

    @patch.dict("os.environ", {"EDENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="EDENAI_API_KEY"):
            from ragbench.components.generators import EdenAIGenerator

            EdenAIGenerator(
                {
                    "llm": {"model_name": "gpt-4o"},
                    "sub_provider": "openai",
                }
            )


# -----------------------------------------------------------------------
# OpenAIGenerator
# -----------------------------------------------------------------------


class TestOpenAIGenerator:
    def _make_generator(self, **overrides: object) -> OpenAIGenerator:
        config: dict = {
            "llm": {"model_name": "gpt-4o", "temperature": 0.0},
            **overrides,
        }
        return OpenAIGenerator(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_config_defaults(self, mock_client_cls: MagicMock) -> None:
        gen = OpenAIGenerator()
        assert gen._model == ""
        assert gen._temperature == 0.0
        assert gen._max_tokens is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_string_prompt(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        gen = self._make_generator()
        result = gen.generate("What is the answer?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is 42."
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "What is the answer?"}
        ]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_message_list_prompt(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        gen = self._make_generator()
        messages: list[Message] = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        gen.generate(messages)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == messages

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_metadata_populated(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ans"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response

        gen = self._make_generator()
        result = gen.generate("Q?")

        assert result.metadata["model"] == "gpt-4o"
        assert result.metadata["provider"] == "openai"
        assert result.metadata["prompt_tokens"] == 10
        assert result.metadata["completion_tokens"] == 20
        assert result.metadata["total_tokens"] == 30

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_api_error_raises_runtime(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OSError("Connection refused")

        gen = self._make_generator()
        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            gen.generate("Q?")

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    @patch("openai.OpenAI")
    def test_missing_api_key_raises(self, mock_client_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIGenerator()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_max_tokens_passed(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ans"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        gen = self._make_generator(
            llm={"model_name": "m", "temperature": 0.0, "max_tokens": 256}
        )
        gen.generate("Q?")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256


# -----------------------------------------------------------------------
# Native tool calling — per provider
# -----------------------------------------------------------------------


class TestOllamaToolCalling:
    def _make_generator(self) -> OllamaGenerator:
        return OllamaGenerator({"llm": {"model_name": "qwen2.5", "temperature": 0.0}})

    @patch.object(OllamaGenerator, "_call_api")
    def test_tools_in_payload(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {"message": {"content": "ans"}}
        self._make_generator().generate("Q?", tools=_tool_specs())
        assert mock_api.call_args[0][0]["tools"] == to_openai_tools(_tool_specs())

    @patch.object(OllamaGenerator, "_call_api")
    def test_no_tools_omits_payload_key(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {"message": {"content": "ans"}}
        self._make_generator().generate("Q?")
        assert "tools" not in mock_api.call_args[0][0]

    @patch.object(OllamaGenerator, "_call_api")
    def test_tool_call_response_parsed(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "knowledge_base",
                            "arguments": {"query": "grade forgiveness"},
                        }
                    }
                ],
            }
        }
        result = self._make_generator().generate("Q?", tools=_tool_specs())
        assert result.tool_calls == [
            ToolCall(
                id="",
                name="knowledge_base",
                arguments={"query": "grade forgiveness"},
            )
        ]
        assert result.finish_reason == "tool_calls"

    @patch.object(OllamaGenerator, "_call_api")
    def test_multiturn_strips_ids_and_objectifies_args(
        self, mock_api: MagicMock
    ) -> None:
        mock_api.return_value = {"message": {"content": "final"}}
        messages: list[Message] = [
            {"role": "user", "content": "Q?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0_0",
                        "type": "function",
                        "function": {"name": "kb", "arguments": '{"query": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0_0", "content": "ctx"},
        ]
        self._make_generator().generate(messages, tools=_tool_specs())
        sent = mock_api.call_args[0][0]["messages"]
        assistant = sent[1]
        assert "id" not in assistant["tool_calls"][0]
        assert assistant["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
        assert sent[2] == {"role": "tool", "content": "ctx"}


class TestOpenAIToolCalling:
    def _make_generator(self) -> OpenAIGenerator:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=False):
            with patch("openai.OpenAI"):
                return OpenAIGenerator({"llm": {"model_name": "gpt-4o"}})

    @staticmethod
    def _resp(
        content: str | None = None,
        tool_calls: list | None = None,
        finish_reason: str = "stop",
    ) -> MagicMock:
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        choice.message.tool_calls = tool_calls
        choice.finish_reason = finish_reason
        resp.choices = [choice]
        resp.usage = None
        return resp

    def test_tools_and_choice_in_kwargs(self) -> None:
        gen = self._make_generator()
        gen._client.chat.completions.create.return_value = self._resp(content="ans")
        gen.generate("Q?", tools=_tool_specs())
        kwargs = gen._client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == to_openai_tools(_tool_specs())
        assert kwargs["tool_choice"] == "auto"

    def test_no_tools_omits_kwargs(self) -> None:
        gen = self._make_generator()
        gen._client.chat.completions.create.return_value = self._resp(content="ans")
        gen.generate("Q?")
        kwargs = gen._client.chat.completions.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_tool_call_response_parsed(self) -> None:
        gen = self._make_generator()
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function.name = "knowledge_base"
        tc.function.arguments = '{"query": "grade forgiveness"}'
        gen._client.chat.completions.create.return_value = self._resp(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        )
        result = gen.generate("Q?", tools=_tool_specs())
        assert result.tool_calls == [
            ToolCall(
                id="call_abc",
                name="knowledge_base",
                arguments={"query": "grade forgiveness"},
            )
        ]
        assert result.finish_reason == "tool_calls"
        assert result.answer == ""


class TestGoogleToolCalling:
    def _make_generator(self) -> GoogleGenerator:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "k"}, clear=False):
            with patch("google.genai.Client"):
                return GoogleGenerator({"llm": {"model_name": "gemini-2.0-flash"}})

    def test_tools_in_config(self) -> None:
        gen = self._make_generator()
        gen._client.models.generate_content.return_value = _google_response(text="ans")
        gen.generate("Q?", tools=_tool_specs())
        config = gen._client.models.generate_content.call_args.kwargs["config"]
        decls = config.tools[0].function_declarations
        assert [d.name for d in decls] == ["knowledge_base"]

    def test_function_call_response_parsed(self) -> None:
        gen = self._make_generator()
        gen._client.models.generate_content.return_value = _google_response(
            function_calls=[("knowledge_base", {"query": "grade forgiveness"})]
        )
        result = gen.generate("Q?", tools=_tool_specs())
        assert result.tool_calls == [
            ToolCall(
                id="",
                name="knowledge_base",
                arguments={"query": "grade forgiveness"},
            )
        ]
        assert result.finish_reason == "tool_calls"

    def test_tool_result_keyed_by_function_name(self) -> None:
        gen = self._make_generator()
        gen._client.models.generate_content.return_value = _google_response(
            text="final"
        )
        messages: list[Message] = [
            {"role": "user", "content": "Q?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "knowledge_base",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ctx"},
        ]
        gen.generate(messages)
        contents = gen._client.models.generate_content.call_args.kwargs["contents"]
        fr_part = contents[-1].parts[0]
        # The tool result is keyed by function NAME (resolved from the id), not id.
        assert fr_part.function_response.name == "knowledge_base"
        assert fr_part.function_response.response == {"result": "ctx"}


class TestEdenAIToolCalling:
    def _make_generator(self) -> EdenAIGenerator:
        with patch.dict("os.environ", {"EDENAI_API_KEY": "k"}, clear=False):
            with patch("langchain_community.chat_models.edenai.ChatEdenAI"):
                return EdenAIGenerator(
                    {"llm": {"model_name": "gpt-4.1"}, "sub_provider": "openai"}
                )

    @staticmethod
    def _result(content: str = "", tool_calls: list | None = None) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls if tool_calls is not None else []
        return msg

    def test_bind_tools_called(self) -> None:
        gen = self._make_generator()
        gen._client.bind_tools.return_value.invoke.return_value = self._result("ans")
        gen.generate("Q?", tools=_tool_specs())
        gen._client.bind_tools.assert_called_once_with(to_openai_tools(_tool_specs()))
        gen._client.bind_tools.return_value.invoke.assert_called_once()

    def test_no_tools_uses_plain_client(self) -> None:
        gen = self._make_generator()
        gen._client.invoke.return_value = self._result("ans")
        gen.generate("Q?")
        gen._client.bind_tools.assert_not_called()
        gen._client.invoke.assert_called_once()

    def test_tool_call_response_parsed(self) -> None:
        gen = self._make_generator()
        gen._client.bind_tools.return_value.invoke.return_value = self._result(
            content="",
            tool_calls=[
                {
                    "name": "knowledge_base",
                    "args": {"query": "grade forgiveness"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        result = gen.generate("Q?", tools=_tool_specs())
        assert result.tool_calls == [
            ToolCall(
                id="call_1",
                name="knowledge_base",
                arguments={"query": "grade forgiveness"},
            )
        ]
        assert result.finish_reason == "tool_calls"

    def test_multiturn_builds_langchain_objects(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        gen = self._make_generator()
        gen._client.bind_tools.return_value.invoke.return_value = self._result("final")
        messages: list[Message] = [
            {"role": "user", "content": "Q?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "knowledge_base",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ctx"},
        ]
        gen.generate(messages, tools=_tool_specs())
        sent = gen._client.bind_tools.return_value.invoke.call_args[0][0]
        ai = sent[1]
        assert isinstance(ai, AIMessage)
        assert ai.tool_calls[0]["name"] == "knowledge_base"
        assert ai.tool_calls[0]["args"] == {"query": "x"}
        assert ai.tool_calls[0]["id"] == "c1"
        tool_msg = sent[2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "c1"
        assert tool_msg.content == "ctx"
