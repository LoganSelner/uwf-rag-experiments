"""Tests for src/ragbench/components/query_transforms.py.

The LLM-backed transformers receive their reasoning generator via constructor
injection (the pipeline builds it through ``build_generator``), so tests pass a
mock generator directly — no registry patching.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragbench.components.query_transforms import (
    _DEFAULT_HYDE_SYSTEM_PROMPT,
    _DEFAULT_MULTI_QUERY_SYSTEM_PROMPT,
    _DEFAULT_SYSTEM_PROMPT,
    ContextualizerQueryTransformer,
    HyDEQueryTransformer,
    MultiQueryQueryTransformer,
    _parse_numbered_list,
)
from ragbench.core.types import GenerationResult, Message, TransformedQuery


def _mock_generator(answer: str = "Standalone question") -> MagicMock:
    """Return a mock generator whose ``generate()`` returns *answer*."""
    gen = MagicMock()
    gen.generate.return_value = GenerationResult(query="", answer=answer)
    return gen


class TestContextualizerQueryTransformer:
    def test_no_history_returns_query_unchanged(self) -> None:
        gen = _mock_generator()
        qt = ContextualizerQueryTransformer({}, generator=gen)
        result = qt.transform("What is UWF?")

        assert result == [TransformedQuery(text="What is UWF?")]
        # No LLM call when history is empty.
        gen.generate.assert_not_called()

    def test_empty_history_returns_query_unchanged(self) -> None:
        gen = _mock_generator()
        qt = ContextualizerQueryTransformer({}, generator=gen)
        result = qt.transform("What is UWF?", history=[])

        assert result == [TransformedQuery(text="What is UWF?")]
        gen.generate.assert_not_called()

    def test_with_history_calls_generator(self) -> None:
        gen = _mock_generator()
        qt = ContextualizerQueryTransformer({}, generator=gen)
        history: list[Message] = [
            {"role": "user", "content": "Tell me about UWF."},
            {"role": "assistant", "content": "UWF is a university."},
        ]
        qt.transform("What programs do they offer?", history=history)

        gen.generate.assert_called_once()

    def test_with_history_returns_reformulated_query(self) -> None:
        gen = _mock_generator("What programs does UWF offer?")
        qt = ContextualizerQueryTransformer({}, generator=gen)
        history: list[Message] = [
            {"role": "user", "content": "Tell me about UWF."},
            {"role": "assistant", "content": "UWF is a university."},
        ]
        result = qt.transform("What programs do they offer?", history=history)

        assert result == [TransformedQuery(text="What programs does UWF offer?")]

    def test_system_prompt_default(self) -> None:
        gen = _mock_generator()
        qt = ContextualizerQueryTransformer({}, generator=gen)
        history: list[Message] = [{"role": "user", "content": "Hi"}]
        qt.transform("Follow up", history=history)

        call_args = gen.generate.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == _DEFAULT_SYSTEM_PROMPT

    def test_system_prompt_override(self) -> None:
        gen = _mock_generator()
        custom_prompt = "Rewrite the question."
        qt = ContextualizerQueryTransformer(
            {"system_prompt": custom_prompt}, generator=gen
        )
        history: list[Message] = [{"role": "user", "content": "Hi"}]
        qt.transform("Follow up", history=history)

        call_args = gen.generate.call_args[0][0]
        assert call_args[0]["content"] == custom_prompt

    def test_missing_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a generator"):
            ContextualizerQueryTransformer({})

    def test_result_is_single_element_list(self) -> None:
        gen = _mock_generator("Reformulated")
        qt = ContextualizerQueryTransformer({}, generator=gen)
        history: list[Message] = [{"role": "user", "content": "Hi"}]
        result = qt.transform("Follow up", history=history)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TransformedQuery)
        assert result[0].branch is None

    def test_messages_include_history_and_query(self) -> None:
        gen = _mock_generator()
        qt = ContextualizerQueryTransformer({}, generator=gen)
        history: list[Message] = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        qt.transform("Second question", history=history)

        messages = gen.generate.call_args[0][0]
        # system + 2 history turns + user query
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3] == {"role": "user", "content": "Second question"}


class TestHyDEQueryTransformer:
    def test_default_emits_one_hypothetical_branch_none(self) -> None:
        # Default branch is None (broadcast) — consistent with all other
        # transformers. Routing to a hybrid child is opt-in.
        gen = _mock_generator("A hypothetical doc.")
        qt = HyDEQueryTransformer({}, generator=gen)
        result = qt.transform("What is grade forgiveness?")
        assert result == [TransformedQuery(text="A hypothetical doc.", branch=None)]

    def test_include_original_canonical_hybrid_routing(self) -> None:
        # Canonical HyDE+hybrid recipe: hypothetical → dense, original → bm25.
        # Both branches are set explicitly (routing is opt-in).
        gen = _mock_generator("Hypothesis.")
        qt = HyDEQueryTransformer(
            {
                "include_original": True,
                "branch": "dense",
                "original_branch": "bm25",
            },
            generator=gen,
        )
        result = qt.transform("Query?")
        assert result == [
            TransformedQuery(text="Query?", branch="bm25"),
            TransformedQuery(text="Hypothesis.", branch="dense"),
        ]

    def test_branches_default_to_none(self) -> None:
        # With no explicit branch / original_branch, both the original and
        # the hypothetical broadcast (branch=None).
        gen = _mock_generator("Hyp.")
        qt = HyDEQueryTransformer({"include_original": True}, generator=gen)
        result = qt.transform("Q?")
        assert result[0].branch is None  # original
        assert result[1].branch is None  # hypothetical

    def test_num_hypotheticals_makes_n_calls(self) -> None:
        gen = _mock_generator("Hyp.")
        qt = HyDEQueryTransformer({"num_hypotheticals": 3}, generator=gen)
        result = qt.transform("Q?")
        assert gen.generate.call_count == 3
        assert len(result) == 3
        assert all(tq.branch is None for tq in result)

    def test_branch_override(self) -> None:
        gen = _mock_generator("Hyp.")
        qt = HyDEQueryTransformer({"branch": "splade"}, generator=gen)
        result = qt.transform("Q?")
        assert result[0].branch == "splade"

    def test_history_is_ignored_for_hyde(self) -> None:
        # HyDE generates a fresh hypothetical from the standalone query;
        # conversation history is not part of the prompt by design.
        gen = _mock_generator("Hyp.")
        qt = HyDEQueryTransformer({}, generator=gen)
        qt.transform("Q?", history=[{"role": "user", "content": "prior"}])
        messages = gen.generate.call_args[0][0]
        assert len(messages) == 2  # system + user only
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_default_system_prompt_used(self) -> None:
        gen = _mock_generator("Hyp.")
        qt = HyDEQueryTransformer({}, generator=gen)
        qt.transform("Q?")
        messages = gen.generate.call_args[0][0]
        assert messages[0]["content"] == _DEFAULT_HYDE_SYSTEM_PROMPT

    def test_system_prompt_override(self) -> None:
        gen = _mock_generator("Hyp.")
        custom = "Generate a fake answer."
        qt = HyDEQueryTransformer({"system_prompt": custom}, generator=gen)
        qt.transform("Q?")
        messages = gen.generate.call_args[0][0]
        assert messages[0]["content"] == custom

    def test_missing_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a generator"):
            HyDEQueryTransformer({})

    def test_num_hypotheticals_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_hypotheticals"):
            HyDEQueryTransformer({"num_hypotheticals": 0}, generator=_mock_generator())

    def test_empty_generator_output_falls_back_to_original(self) -> None:
        # LLM returns whitespace-only → no hypothetical; transformer
        # gracefully falls back to the original query.
        gen = _mock_generator("   ")
        qt = HyDEQueryTransformer({}, generator=gen)
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
    def test_default_emits_original_plus_n_reformulations(self) -> None:
        gen = _mock_generator(
            "1. reformulation A\n"
            "2. reformulation B\n"
            "3. reformulation C\n"
            "4. reformulation D"
        )
        qt = MultiQueryQueryTransformer({}, generator=gen)
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

    def test_include_original_false_omits_original(self) -> None:
        gen = _mock_generator("1. only reformulation")
        qt = MultiQueryQueryTransformer(
            {"num_queries": 1, "include_original": False}, generator=gen
        )
        result = qt.transform("Original")
        assert result == [TransformedQuery(text="only reformulation")]

    def test_single_llm_call(self) -> None:
        # Multi-query batches into a single generation call, unlike HyDE's
        # per-hypothetical loop.
        gen = _mock_generator("1. a\n2. b")
        qt = MultiQueryQueryTransformer({"num_queries": 2}, generator=gen)
        qt.transform("Q?")
        assert gen.generate.call_count == 1

    def test_num_queries_substituted_in_prompt(self) -> None:
        gen = _mock_generator("1. a\n2. b\n3. c")
        qt = MultiQueryQueryTransformer({"num_queries": 3}, generator=gen)
        qt.transform("Q?")
        messages = gen.generate.call_args[0][0]
        assert "3" in messages[0]["content"]
        assert "{num_queries}" not in messages[0]["content"]

    def test_branch_override_tags_all_queries(self) -> None:
        gen = _mock_generator("1. a\n2. b")
        qt = MultiQueryQueryTransformer(
            {"num_queries": 2, "branch": "dense"}, generator=gen
        )
        result = qt.transform("Q?")
        # Original + 2 reformulations, all tagged "dense".
        assert all(tq.branch == "dense" for tq in result)

    def test_history_is_ignored(self) -> None:
        # Multi-query, like HyDE, generates from the standalone query;
        # conversation history is not part of the prompt.
        gen = _mock_generator("1. a\n2. b")
        qt = MultiQueryQueryTransformer({"num_queries": 2}, generator=gen)
        qt.transform("Q?", history=[{"role": "user", "content": "prior"}])
        messages = gen.generate.call_args[0][0]
        assert len(messages) == 2  # system + user
        assert messages[1]["role"] == "user"

    def test_default_system_prompt_used(self) -> None:
        gen = _mock_generator("1. a")
        qt = MultiQueryQueryTransformer({"num_queries": 1}, generator=gen)
        qt.transform("Q?")
        # Default prompt contains the templated substitution.
        rendered = _DEFAULT_MULTI_QUERY_SYSTEM_PROMPT.replace("{num_queries}", "1")
        assert gen.generate.call_args[0][0][0]["content"] == rendered

    def test_system_prompt_override(self) -> None:
        gen = _mock_generator("1. a")
        custom = "Give me {num_queries} variants."
        qt = MultiQueryQueryTransformer(
            {"system_prompt": custom, "num_queries": 3}, generator=gen
        )
        qt.transform("Q?")
        assert gen.generate.call_args[0][0][0]["content"] == "Give me 3 variants."

    def test_under_count_returns_what_we_have(self) -> None:
        # LLM emits 2 reformulations but we asked for 4. We get what
        # we got; the pipeline still RRF-fuses 2 lists.
        gen = _mock_generator("1. one\n2. two")
        qt = MultiQueryQueryTransformer(
            {"num_queries": 4, "include_original": False}, generator=gen
        )
        result = qt.transform("Q?")
        assert [tq.text for tq in result] == ["one", "two"]

    def test_empty_output_falls_back_to_original(self) -> None:
        gen = _mock_generator("   ")
        qt = MultiQueryQueryTransformer({}, generator=gen)
        result = qt.transform("Q?")
        assert result == [TransformedQuery(text="Q?")]

    def test_missing_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a generator"):
            MultiQueryQueryTransformer({})

    def test_num_queries_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_queries"):
            MultiQueryQueryTransformer({"num_queries": 0}, generator=_mock_generator())
