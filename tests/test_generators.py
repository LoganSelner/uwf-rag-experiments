"""Tests for src/components/generators.py — Ollama, Google, and EdenAI generators."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from components.generators import GoogleGenerator, OllamaGenerator, OpenAIGenerator
from core.types import GenerationResult

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

    def test_config_defaults(self) -> None:
        gen = OllamaGenerator()
        assert gen._model == ""
        assert gen._temperature == 0.0
        assert gen._max_tokens is None
        assert gen._base_url == "http://localhost:11434"

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
        messages = [
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
        mock_response = MagicMock()
        mock_response.text = "The answer is 42."
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

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
        messages = [
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
        messages = [
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
        with pytest.raises(ValueError, match="sub_provider"):
            # Patch dotenv so it doesn't touch the real env
            with patch("components.generators.os.environ.get", return_value="fake-key"):
                from components.generators import EdenAIGenerator

                EdenAIGenerator({"llm": {"model_name": "gpt-4o"}})

    @patch.dict("os.environ", {"EDENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="EDENAI_API_KEY"):
            from components.generators import EdenAIGenerator

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
        messages = [
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
