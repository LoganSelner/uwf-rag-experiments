"""Tests for src/components/generators.py — Ollama and EdenAI generators."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from components.generators import OllamaGenerator, _is_retryable
from core.types import GenerationResult

# -----------------------------------------------------------------------
# _is_retryable
# -----------------------------------------------------------------------


class TestIsRetryable:
    def test_network_error_retryable(self) -> None:
        assert _is_retryable(OSError("Connection refused")) is True

    def test_runtime_error_retryable(self) -> None:
        assert _is_retryable(RuntimeError("500 Internal Server Error")) is True

    def test_type_error_not_retryable(self) -> None:
        assert _is_retryable(TypeError("bad type")) is False

    def test_value_error_not_retryable(self) -> None:
        assert _is_retryable(ValueError("bad value")) is False

    def test_key_error_not_retryable(self) -> None:
        assert _is_retryable(KeyError("missing")) is False

    def test_attribute_error_not_retryable(self) -> None:
        assert _is_retryable(AttributeError("no attr")) is False


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
