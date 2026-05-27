"""Tests for src/components/query_transforms.py — contextualizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from components.query_transforms import (
    _DEFAULT_HYDE_SYSTEM_PROMPT,
    _DEFAULT_SYSTEM_PROMPT,
    ContextualizerQueryTransformer,
    HyDEQueryTransformer,
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


class TestHyDEQueryTransformer:
    @patch("components.query_transforms.registry")
    def test_default_emits_one_hypothetical_branch_dense(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls("A hypothetical doc.")
        qt = HyDEQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        result = qt.transform("What is grade forgiveness?")
        assert result == [TransformedQuery(text="A hypothetical doc.", branch="dense")]

    @patch("components.query_transforms.registry")
    def test_include_original_prepends_original(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hypothesis.")
        qt = HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "include_original": True,
                "original_branch": "bm25",
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Query?")
        assert result == [
            TransformedQuery(text="Query?", branch="bm25"),
            TransformedQuery(text="Hypothesis.", branch="dense"),
        ]

    @patch("components.query_transforms.registry")
    def test_original_branch_defaults_to_none(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        qt = HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "include_original": True,
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Q?")
        # branch=None means "broadcast to all hybrid children" or "ignored
        # by single retrievers".
        assert result[0].branch is None
        assert result[1].branch == "dense"

    @patch("components.query_transforms.registry")
    def test_num_hypotheticals_makes_n_calls(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        qt = HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "num_hypotheticals": 3,
                "llm": {"model_name": "m", "temperature": 0.5},
            }
        )
        result = qt.transform("Q?")
        assert qt._generator.generate.call_count == 3
        assert len(result) == 3
        assert all(tq.branch == "dense" for tq in result)

    @patch("components.query_transforms.registry")
    def test_branch_override(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        qt = HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "branch": "splade",
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Q?")
        assert result[0].branch == "splade"

    @patch("components.query_transforms.registry")
    def test_history_is_ignored_for_hyde(self, mock_registry: MagicMock) -> None:
        # HyDE generates a fresh hypothetical from the standalone query;
        # conversation history is not part of the prompt by design.
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        qt = HyDEQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        qt.transform("Q?", history=[{"role": "user", "content": "prior"}])
        messages = qt._generator.generate.call_args[0][0]
        assert len(messages) == 2  # system + user only
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("components.query_transforms.registry")
    def test_default_system_prompt_used(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        qt = HyDEQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        qt.transform("Q?")
        messages = qt._generator.generate.call_args[0][0]
        assert messages[0]["content"] == _DEFAULT_HYDE_SYSTEM_PROMPT

    @patch("components.query_transforms.registry")
    def test_system_prompt_override(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("Hyp.")
        custom = "Generate a fake answer."
        qt = HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "system_prompt": custom,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?")
        messages = qt._generator.generate.call_args[0][0]
        assert messages[0]["content"] == custom

    @patch("components.query_transforms.registry")
    def test_generator_receives_provider_params(self, mock_registry: MagicMock) -> None:
        # HyDE-specific keys are stripped before passing config to the
        # generator; provider params (sub_provider, base_url, llm) pass through.
        mock_cls = _mock_generator_cls("Hyp.")
        mock_registry.get.return_value = mock_cls
        HyDEQueryTransformer(
            {
                "generator_type": "edenai",
                "sub_provider": "openai",
                "base_url": "http://example:8080",
                "num_hypotheticals": 1,
                "include_original": True,
                "branch": "dense",
                "original_branch": "bm25",
                "system_prompt": "ignored by generator",
                "llm": {"model_name": "gpt-4.1", "temperature": 0.0},
            }
        )
        mock_cls.assert_called_once_with(
            config={
                "sub_provider": "openai",
                "base_url": "http://example:8080",
                "llm": {"model_name": "gpt-4.1", "temperature": 0.0},
            }
        )

    def test_missing_generator_type_raises(self) -> None:
        with pytest.raises(ValueError, match="generator_type"):
            HyDEQueryTransformer({"llm": {"model_name": "m"}})

    @patch("components.query_transforms.registry")
    def test_num_hypotheticals_must_be_positive(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()
        with pytest.raises(ValueError, match="num_hypotheticals"):
            HyDEQueryTransformer(
                {
                    "generator_type": "edenai",
                    "num_hypotheticals": 0,
                    "llm": {"model_name": "m"},
                }
            )

    @patch("components.query_transforms.registry")
    def test_empty_generator_output_falls_back_to_original(
        self, mock_registry: MagicMock
    ) -> None:
        # LLM returns whitespace-only → no hypothetical; transformer
        # gracefully falls back to the original query.
        mock_registry.get.return_value = _mock_generator_cls("   ")
        qt = HyDEQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        result = qt.transform("Q?")
        assert result == [TransformedQuery(text="Q?")]
