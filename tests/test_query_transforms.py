"""Tests for src/components/query_transforms.py — contextualizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from components.query_transforms import (
    _DEFAULT_SYSTEM_PROMPT,
    ContextualizerQueryTransformer,
)
from core.types import GenerationResult, TransformedQuery


def _mock_generator_cls(answer: str = "Standalone question") -> MagicMock:
    """Return a mock generator class whose instances return *answer*."""
    mock_gen = MagicMock()
    mock_gen.generate.return_value = GenerationResult(query="", answer=answer)
    mock_cls = MagicMock(return_value=mock_gen)
    return mock_cls


class TestContextualizerQueryTransformer:
    @patch("components.query_transforms.registry")
    def test_no_history_returns_query_unchanged(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        result = qt.transform("What is UWF?")

        assert result == [TransformedQuery(text="What is UWF?")]
        # No LLM call when history is empty
        qt._generator.generate.assert_not_called()

    @patch("components.query_transforms.registry")
    def test_empty_history_returns_query_unchanged(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        result = qt.transform("What is UWF?", history=[])

        assert result == [TransformedQuery(text="What is UWF?")]
        qt._generator.generate.assert_not_called()

    @patch("components.query_transforms.registry")
    def test_with_history_calls_generator(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        history = [
            {"role": "user", "content": "Tell me about UWF."},
            {"role": "assistant", "content": "UWF is a university."},
        ]
        qt.transform("What programs do they offer?", history=history)

        qt._generator.generate.assert_called_once()

    @patch("components.query_transforms.registry")
    def test_with_history_returns_reformulated_query(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls(
            "What programs does UWF offer?"
        )

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        history = [
            {"role": "user", "content": "Tell me about UWF."},
            {"role": "assistant", "content": "UWF is a university."},
        ]
        result = qt.transform("What programs do they offer?", history=history)

        assert result == [TransformedQuery(text="What programs does UWF offer?")]

    @patch("components.query_transforms.registry")
    def test_system_prompt_default(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        history = [{"role": "user", "content": "Hi"}]
        qt.transform("Follow up", history=history)

        call_args = qt._generator.generate.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == _DEFAULT_SYSTEM_PROMPT

    @patch("components.query_transforms.registry")
    def test_system_prompt_override(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        custom_prompt = "Rewrite the question."
        qt = ContextualizerQueryTransformer(
            {
                "generator_type": "google",
                "llm": {"model_name": "m"},
                "system_prompt": custom_prompt,
            }
        )
        history = [{"role": "user", "content": "Hi"}]
        qt.transform("Follow up", history=history)

        call_args = qt._generator.generate.call_args[0][0]
        assert call_args[0]["content"] == custom_prompt

    @patch("components.query_transforms.registry")
    def test_generator_receives_extra_params(self, mock_registry: MagicMock) -> None:
        mock_cls = _mock_generator_cls()
        mock_registry.get.return_value = mock_cls

        ContextualizerQueryTransformer(
            {
                "generator_type": "edenai",
                "system_prompt": "Custom prompt.",
                "sub_provider": "openai",
                "base_url": "http://custom:8080",
                "llm": {"model_name": "gpt-4", "temperature": 0.5},
            }
        )

        mock_cls.assert_called_once_with(
            config={
                "sub_provider": "openai",
                "base_url": "http://custom:8080",
                "llm": {"model_name": "gpt-4", "temperature": 0.5},
            }
        )

    def test_missing_generator_type_raises(self) -> None:
        with pytest.raises(ValueError, match="generator_type"):
            ContextualizerQueryTransformer({"llm": {"model_name": "m"}})

    @patch("components.query_transforms.registry")
    def test_result_is_single_element_list(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Reformulated")

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        history = [{"role": "user", "content": "Hi"}]
        result = qt.transform("Follow up", history=history)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TransformedQuery)
        assert result[0].branch is None

    @patch("components.query_transforms.registry")
    def test_messages_include_history_and_query(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()

        qt = ContextualizerQueryTransformer(
            {"generator_type": "google", "llm": {"model_name": "m"}}
        )
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        qt.transform("Second question", history=history)

        messages = qt._generator.generate.call_args[0][0]
        # system + 2 history turns + user query
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3] == {"role": "user", "content": "Second question"}
