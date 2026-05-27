"""Tests for src/components/query_transforms.py — contextualizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from components.query_transforms import (
    _DEFAULT_HYDE_SYSTEM_PROMPT,
    _DEFAULT_MULTI_QUERY_SYSTEM_PROMPT,
    _DEFAULT_SYSTEM_PROMPT,
    ContextualizerQueryTransformer,
    HyDEQueryTransformer,
    MultiQueryQueryTransformer,
    _parse_numbered_list,
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


class TestParseNumberedList:
    def test_period_prefix(self) -> None:
        text = "1. first reformulation\n2. second\n3. third"
        assert _parse_numbered_list(text, expected=3) == [
            "first reformulation",
            "second",
            "third",
        ]

    def test_paren_prefix(self) -> None:
        text = "1) first\n2) second"
        assert _parse_numbered_list(text, expected=2) == ["first", "second"]

    def test_colon_prefix(self) -> None:
        text = "1: first\n2: second"
        assert _parse_numbered_list(text, expected=2) == ["first", "second"]

    def test_hyphen_bullet(self) -> None:
        text = "- alpha\n- beta\n- gamma"
        assert _parse_numbered_list(text, expected=3) == ["alpha", "beta", "gamma"]

    def test_asterisk_bullet(self) -> None:
        text = "* one\n* two"
        assert _parse_numbered_list(text, expected=2) == ["one", "two"]

    def test_strips_surrounding_quotes(self) -> None:
        text = "1. \"quoted reformulation\"\n2. 'single-quoted'"
        assert _parse_numbered_list(text, expected=2) == [
            "quoted reformulation",
            "single-quoted",
        ]

    def test_blank_lines_tolerated(self) -> None:
        text = "1. first\n\n\n2. second\n\n"
        assert _parse_numbered_list(text, expected=2) == ["first", "second"]

    def test_truncates_to_expected(self) -> None:
        text = "1. a\n2. b\n3. c\n4. d\n5. e"
        assert _parse_numbered_list(text, expected=3) == ["a", "b", "c"]

    def test_fallback_to_bare_lines_when_no_numbering(self) -> None:
        # Some LLMs (Ollama qwen3) occasionally drop the numbering.
        text = "first reformulation\nsecond reformulation"
        assert _parse_numbered_list(text, expected=2) == [
            "first reformulation",
            "second reformulation",
        ]

    def test_empty_text_returns_empty(self) -> None:
        assert _parse_numbered_list("", expected=4) == []
        assert _parse_numbered_list("   \n  \n", expected=4) == []

    def test_under_expected_returns_what_we_have(self) -> None:
        # 1 line, 4 expected — return the 1 we got, don't fail.
        result = _parse_numbered_list("1. only", expected=4)
        assert result == ["only"]

    def test_mixed_prefix_styles(self) -> None:
        text = "1. period\n2) paren\n- bullet"
        assert _parse_numbered_list(text, expected=3) == ["period", "paren", "bullet"]


class TestMultiQueryQueryTransformer:
    @patch("components.query_transforms.registry")
    def test_default_emits_original_plus_n_reformulations(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls(
            "1. reformulation A\n"
            "2. reformulation B\n"
            "3. reformulation C\n"
            "4. reformulation D"
        )
        qt = MultiQueryQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        result = qt.transform("Original Q?")
        # Default include_original=True, num_queries=4 → 5 total.
        assert len(result) == 5
        assert result[0] == TransformedQuery(text="Original Q?")
        assert [tq.text for tq in result[1:]] == [
            "reformulation A",
            "reformulation B",
            "reformulation C",
            "reformulation D",
        ]
        assert all(tq.branch is None for tq in result)

    @patch("components.query_transforms.registry")
    def test_include_original_false_omits_original(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls("1. only reformulation")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 1,
                "include_original": False,
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Original")
        assert result == [TransformedQuery(text="only reformulation")]

    @patch("components.query_transforms.registry")
    def test_single_llm_call(self, mock_registry: MagicMock) -> None:
        # Multi-query batches into a single generation call, unlike HyDE's
        # per-hypothetical loop.
        mock_registry.get.return_value = _mock_generator_cls("1. a\n2. b")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 2,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?")
        assert qt._generator.generate.call_count == 1

    @patch("components.query_transforms.registry")
    def test_num_queries_substituted_in_prompt(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("1. a\n2. b\n3. c")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 3,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?")
        messages = qt._generator.generate.call_args[0][0]
        assert "3" in messages[0]["content"]
        assert "{num_queries}" not in messages[0]["content"]

    @patch("components.query_transforms.registry")
    def test_branch_override_tags_all_queries(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("1. a\n2. b")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 2,
                "branch": "dense",
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Q?")
        # Original + 2 reformulations, all tagged "dense".
        assert all(tq.branch == "dense" for tq in result)

    @patch("components.query_transforms.registry")
    def test_history_is_ignored(self, mock_registry: MagicMock) -> None:
        # Multi-query, like HyDE, generates from the standalone query;
        # conversation history is not part of the prompt.
        mock_registry.get.return_value = _mock_generator_cls("1. a\n2. b")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 2,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?", history=[{"role": "user", "content": "prior"}])
        messages = qt._generator.generate.call_args[0][0]
        assert len(messages) == 2  # system + user
        assert messages[1]["role"] == "user"

    @patch("components.query_transforms.registry")
    def test_default_system_prompt_used(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("1. a")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 1,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?")
        # Default prompt contains the templated substitution.
        rendered = _DEFAULT_MULTI_QUERY_SYSTEM_PROMPT.replace("{num_queries}", "1")
        assert qt._generator.generate.call_args[0][0][0]["content"] == rendered

    @patch("components.query_transforms.registry")
    def test_system_prompt_override(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls("1. a")
        custom = "Give me {num_queries} variants."
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "system_prompt": custom,
                "num_queries": 3,
                "llm": {"model_name": "m"},
            }
        )
        qt.transform("Q?")
        assert (
            qt._generator.generate.call_args[0][0][0]["content"]
            == "Give me 3 variants."
        )

    @patch("components.query_transforms.registry")
    def test_under_count_returns_what_we_have(self, mock_registry: MagicMock) -> None:
        # LLM emits 2 reformulations but we asked for 4. We get what
        # we got; the pipeline still RRF-fuses 2 lists.
        mock_registry.get.return_value = _mock_generator_cls("1. one\n2. two")
        qt = MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "num_queries": 4,
                "include_original": False,
                "llm": {"model_name": "m"},
            }
        )
        result = qt.transform("Q?")
        assert [tq.text for tq in result] == ["one", "two"]

    @patch("components.query_transforms.registry")
    def test_empty_output_falls_back_to_original(
        self, mock_registry: MagicMock
    ) -> None:
        mock_registry.get.return_value = _mock_generator_cls("   ")
        qt = MultiQueryQueryTransformer(
            {"generator_type": "edenai", "llm": {"model_name": "m"}}
        )
        result = qt.transform("Q?")
        assert result == [TransformedQuery(text="Q?")]

    @patch("components.query_transforms.registry")
    def test_generator_receives_provider_params(self, mock_registry: MagicMock) -> None:
        # Multi-query-specific keys are stripped before the generator
        # is constructed; provider params (sub_provider, base_url, llm)
        # pass through.
        mock_cls = _mock_generator_cls("1. a")
        mock_registry.get.return_value = mock_cls
        MultiQueryQueryTransformer(
            {
                "generator_type": "edenai",
                "sub_provider": "openai",
                "base_url": "http://example:8080",
                "num_queries": 4,
                "include_original": True,
                "branch": "dense",
                "system_prompt": "custom",
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
            MultiQueryQueryTransformer({"llm": {"model_name": "m"}})

    @patch("components.query_transforms.registry")
    def test_num_queries_must_be_positive(self, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = _mock_generator_cls()
        with pytest.raises(ValueError, match="num_queries"):
            MultiQueryQueryTransformer(
                {
                    "generator_type": "edenai",
                    "num_queries": 0,
                    "llm": {"model_name": "m"},
                }
            )
