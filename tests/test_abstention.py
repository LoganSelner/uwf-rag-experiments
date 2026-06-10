"""Tests for evaluation/abstention.py — abstention classifiers (Phase E item 13)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragbench.core.types import EvalSample
from ragbench.evaluation.abstention import (
    ABSTENTION_METRIC_NAME,
    LLMAbstentionClassifier,
    PhraseAbstentionClassifier,
    build_abstention_classifier,
)


def _sample(response: str, answerable: bool | None = None) -> EvalSample:
    meta: dict = {} if answerable is None else {"answerable": answerable}
    return EvalSample(
        id="1",
        query="q",
        response=response,
        retrieved_contexts=[],
        reference="r",
        metadata=meta,
    )


class TestPhraseAbstentionClassifier:
    def test_direct_refusal(self) -> None:
        out = PhraseAbstentionClassifier().classify(
            [_sample("I don't know the answer to that.")]
        )
        assert out == [1.0]

    def test_indirect_redirect_counts(self) -> None:
        # The hard case: a redirect with no literal "I refuse".
        out = PhraseAbstentionClassifier().classify(
            [_sample("Please check MyUWF for your current balance.")]
        )
        assert out == [1.0]

    def test_substantive_answer(self) -> None:
        out = PhraseAbstentionClassifier().classify(
            [_sample("Grade forgiveness is allowed three times.")]
        )
        assert out == [0.0]

    def test_empty_response(self) -> None:
        assert PhraseAbstentionClassifier().classify([_sample("")]) == [0.0]


class TestLLMAbstentionClassifier:
    def test_delegates_to_aspect_critics(self) -> None:
        samples = [_sample("a1"), _sample("a2")]
        verdicts = [{ABSTENTION_METRIC_NAME: 1.0}, {ABSTENTION_METRIC_NAME: None}]
        with patch(
            "ragbench.evaluation.abstention._ragas_adapter.run_aspect_critics",
            return_value=verdicts,
        ) as mock_run:
            classifier = LLMAbstentionClassifier(
                llm=MagicMock(), run_config=MagicMock()
            )
            out = classifier.classify(samples)
        assert out == [1.0, None]  # NaN verdict surfaced as None (unknown)
        assert mock_run.called

    def test_empty_samples_short_circuits(self) -> None:
        classifier = LLMAbstentionClassifier(llm=MagicMock(), run_config=MagicMock())
        assert classifier.classify([]) == []


class TestBuildAbstentionClassifier:
    def test_phrase(self) -> None:
        assert isinstance(
            build_abstention_classifier("phrase"), PhraseAbstentionClassifier
        )

    def test_llm(self) -> None:
        classifier = build_abstention_classifier(
            "llm", llm=MagicMock(), run_config=MagicMock()
        )
        assert isinstance(classifier, LLMAbstentionClassifier)

    def test_llm_without_llm_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a judge"):
            build_abstention_classifier("llm")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown abstention classifier"):
            build_abstention_classifier("bogus")
