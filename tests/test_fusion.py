"""Tests for src/core/fusion.py — RRF and weighted score fusion."""

from __future__ import annotations

import math

import pytest

from ragbench.core.fusion import (
    max_score_dedup,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


class TestReciprocalRankFusion:
    def test_single_list_returns_descending(self) -> None:
        result = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        ids = [doc for doc, _ in result]
        assert ids == ["a", "b", "c"]
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_shared_doc_boosted_above_unique(self) -> None:
        # 'a' appears in both lists; 'b' and 'c' only in one each.
        result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
        ranked = [doc for doc, _ in result]
        assert ranked[0] == "a"
        assert set(ranked[1:]) == {"b", "c"}

    def test_score_formula_exact(self) -> None:
        # 'a' rank 1 in both lists: score = 1/61 + 1/61
        # 'b' rank 2 in list 0 only:  score = 1/62
        result = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=60))
        assert math.isclose(result["a"], 2.0 / 61.0)
        assert math.isclose(result["b"], 1.0 / 62.0)

    def test_empty_inner_list_ignored(self) -> None:
        result = reciprocal_rank_fusion([["a", "b"], []], k=60)
        ids = [doc for doc, _ in result]
        assert ids == ["a", "b"]

    def test_all_empty_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_disjoint_lists_combine_all(self) -> None:
        result = reciprocal_rank_fusion([["a"], ["b"], ["c"]], k=60)
        assert {doc for doc, _ in result} == {"a", "b", "c"}

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be > 0"):
            reciprocal_rank_fusion([["a"]], k=0)
        with pytest.raises(ValueError, match="k must be > 0"):
            reciprocal_rank_fusion([["a"]], k=-5)

    def test_different_k_changes_scores(self) -> None:
        small = dict(reciprocal_rank_fusion([["a"]], k=1))
        large = dict(reciprocal_rank_fusion([["a"]], k=1000))
        # Smaller k boosts top-rank scores more strongly.
        assert small["a"] > large["a"]


class TestWeightedScoreFusion:
    def test_single_list_with_normalization(self) -> None:
        result = weighted_score_fusion(
            [[("a", 10.0), ("b", 5.0), ("c", 0.0)]],
            weights=[1.0],
            normalize="min_max",
        )
        ids = [doc for doc, _ in result]
        assert ids == ["a", "b", "c"]
        scores = dict(result)
        # min_max maps to [0, 1]; weight normalization gives 1.0 share.
        assert math.isclose(scores["a"], 1.0)
        assert math.isclose(scores["c"], 0.0)

    def test_two_lists_equal_weight_min_max(self) -> None:
        # List 1: cosine-like; List 2: BM25-like (very different scales)
        result = dict(
            weighted_score_fusion(
                [
                    [("a", 0.95), ("b", 0.7), ("c", 0.5)],
                    [("a", 15.0), ("d", 8.0), ("c", 2.0)],
                ],
                weights=[0.5, 0.5],
                normalize="min_max",
            )
        )
        # 'a' is rank 1 in both: should be the top fused.
        assert max(result, key=lambda k: result[k]) == "a"

    def test_zero_weight_branch_ignored(self) -> None:
        result = dict(
            weighted_score_fusion(
                [
                    [("a", 1.0)],
                    [("b", 1.0)],
                ],
                weights=[1.0, 0.0],
                normalize="min_max",
            )
        )
        assert "a" in result
        assert "b" not in result

    def test_weights_normalized_internally(self) -> None:
        # Weights [2, 2] should behave like [0.5, 0.5] after internal norm.
        eq = dict(
            weighted_score_fusion(
                [[("a", 1.0)], [("a", 1.0)]],
                weights=[0.5, 0.5],
                normalize="min_max",
            )
        )
        scaled = dict(
            weighted_score_fusion(
                [[("a", 1.0)], [("a", 1.0)]],
                weights=[2.0, 2.0],
                normalize="min_max",
            )
        )
        assert math.isclose(eq["a"], scaled["a"])

    def test_all_equal_scores_collapse_to_one(self) -> None:
        # min_max with hi == lo: collapse to 1.0 so weights still apply.
        result = dict(
            weighted_score_fusion(
                [[("a", 5.0), ("b", 5.0)]],
                weights=[1.0],
                normalize="min_max",
            )
        )
        assert math.isclose(result["a"], 1.0)
        assert math.isclose(result["b"], 1.0)

    def test_normalize_none_uses_raw_scores(self) -> None:
        result = dict(
            weighted_score_fusion(
                [[("a", 100.0), ("b", 1.0)]],
                weights=[1.0],
                normalize="none",
            )
        )
        assert math.isclose(result["a"], 100.0)
        assert math.isclose(result["b"], 1.0)

    def test_empty_inputs(self) -> None:
        # Both branches empty but valid weights → empty fused list.
        assert weighted_score_fusion([[], []], weights=[0.5, 0.5]) == []

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            weighted_score_fusion([[("a", 1.0)]], weights=[0.5, 0.5])

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            weighted_score_fusion([[("a", 1.0)]], weights=[-0.1])

    def test_rejects_zero_sum_weights(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            weighted_score_fusion(
                [[("a", 1.0)], [("b", 1.0)]],
                weights=[0.0, 0.0],
            )

    def test_rejects_unknown_normalize(self) -> None:
        with pytest.raises(ValueError, match="normalize"):
            weighted_score_fusion([[("a", 1.0)]], weights=[1.0], normalize="zscore")


class TestMaxScoreDedup:
    def test_single_list_preserved_descending(self) -> None:
        result = max_score_dedup([[("a", 0.9), ("b", 0.5), ("c", 0.7)]])
        ids = [doc for doc, _ in result]
        assert ids == ["a", "c", "b"]

    def test_keeps_highest_across_lists(self) -> None:
        result = dict(
            max_score_dedup(
                [
                    [("a", 0.5), ("b", 0.8)],
                    [("a", 0.9), ("c", 0.6)],
                ]
            )
        )
        assert result["a"] == 0.9
        assert result["b"] == 0.8
        assert result["c"] == 0.6

    def test_empty_inputs(self) -> None:
        assert max_score_dedup([]) == []
        assert max_score_dedup([[], []]) == []

    def test_equal_scores_keep_one_entry(self) -> None:
        result = max_score_dedup([[("a", 0.5)], [("a", 0.5)]])
        assert result == [("a", 0.5)]

    def test_descending_sort(self) -> None:
        result = max_score_dedup([[("low", 0.1), ("hi", 0.9), ("mid", 0.5)]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
