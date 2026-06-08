"""Tests for evaluation/conflict.py — knowledge-conflict scoring (item 15)."""

from __future__ import annotations

from ragbench.evaluation.conflict import (
    CORPUS_PREFERENCE_NAME,
    ERROR_DETECTION_NAME,
    conflict_specs,
    corpus_fact_retrieved,
)


class TestConflictSpecs:
    def test_two_named_specs_with_definitions(self) -> None:
        specs = conflict_specs()
        assert {s.name for s in specs} == {CORPUS_PREFERENCE_NAME, ERROR_DETECTION_NAME}
        assert all(s.definition for s in specs)


class TestCorpusFactRetrieved:
    def test_true_when_fact_present(self) -> None:
        contexts = [
            "Grade forgiveness is allowed three times within the undergraduate career."
        ]
        assert (
            corpus_fact_retrieved(contexts, "Grade forgiveness is allowed three times")
            is True
        )

    def test_false_when_absent(self) -> None:
        contexts = ["Parking permits are available at the cashier's office."]
        assert (
            corpus_fact_retrieved(contexts, "Grade forgiveness is allowed three times")
            is False
        )

    def test_none_when_no_distinctive_words(self) -> None:
        # All tokens are short/stopwords → nothing distinctive to match.
        assert corpus_fact_retrieved(["anything at all"], "is the a of") is None
