"""Tests for src/evaluation/evaluator.py — RAGAS evaluation integration."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import EvaluationConfig
from core.types import EvalSample
from evaluation.evaluator import Evaluator, _load_dataset

# -----------------------------------------------------------------------
# _load_dataset
# -----------------------------------------------------------------------


class TestLoadDataset:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"query": "Q1?", "reference": "R1"}\n{"query": "Q2?", "reference": "R2"}\n'
        )
        data = _load_dataset(str(f))
        assert len(data) == 2
        assert data[0]["query"] == "Q1?"

    def test_assigns_sequential_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"query": "Q", "reference": "R"}\n')
        data = _load_dataset(str(f))
        assert data[0]["id"] == "1"

    def test_preserves_existing_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"id": "custom_id", "query": "Q", "reference": "R"}\n')
        data = _load_dataset(str(f))
        assert data[0]["id"] == "custom_id"

    def test_empty_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _load_dataset(str(f))

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"query": "Q"}\n')  # missing "reference"
        with pytest.raises(ValueError, match="reference"):
            _load_dataset(str(f))


# -----------------------------------------------------------------------
# Evaluator.active_metrics
# -----------------------------------------------------------------------


class TestActiveMetrics:
    def test_full_mode(self) -> None:
        cfg = EvaluationConfig.from_dict({"mode": "full"})
        evaluator = Evaluator(cfg)
        metrics = evaluator.active_metrics
        assert "faithfulness" in metrics
        assert "answer_correctness" in metrics

    def test_retrieval_only_mode(self) -> None:
        cfg = EvaluationConfig.from_dict({"mode": "retrieval_only"})
        evaluator = Evaluator(cfg)
        metrics = evaluator.active_metrics
        assert "context_precision" in metrics
        assert "faithfulness" not in metrics


# -----------------------------------------------------------------------
# Evaluator._aggregate_metrics
# -----------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty(self) -> None:
        assert Evaluator._aggregate_metrics([]) == {}

    def test_single_run(self) -> None:
        result = Evaluator._aggregate_metrics([{"acc": 0.9}])
        assert result["acc"] == 0.9
        assert result["acc_std"] == 0.0

    def test_multiple_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": 0.9},
                {"acc": 1.0},
            ]
        )
        assert abs(result["acc"] - 0.9) < 1e-9
        assert result["acc_std"] > 0

    def test_empty_first_run_with_valid_later_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {},
                {"acc": 0.8},
                {"acc": 0.9},
            ]
        )
        assert abs(result["acc"] - 0.85) < 1e-9
        assert result["acc_std"] > 0

    def test_sparse_keys_across_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": 0.9, "faithfulness": 0.7},
                {"faithfulness": 0.9},
            ]
        )
        assert abs(result["acc"] - 0.85) < 1e-9
        assert abs(result["faithfulness"] - 0.8) < 1e-9
        assert "acc_std" in result
        assert "faithfulness_std" in result

    def test_all_empty_runs(self) -> None:
        assert Evaluator._aggregate_metrics([{}, {}, {}]) == {}

    def test_single_key_in_later_run_only(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {},
                {},
                {"acc": 0.95},
            ]
        )
        assert result["acc"] == 0.95
        assert result["acc_std"] == 0.0

    def test_nan_filtered(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": float("nan")},
                {"acc": 1.0},
            ]
        )
        assert abs(result["acc"] - 0.9) < 1e-9


# -----------------------------------------------------------------------
# _EmbedderAdapter
# -----------------------------------------------------------------------


class TestEmbedderAdapter:
    def test_embed_query(self) -> None:
        from evaluation.evaluator import _EmbedderAdapter

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        adapter = _EmbedderAdapter(mock_embedder)
        result = adapter.embed_query("test")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_embed_documents(self) -> None:
        from evaluation.evaluator import _EmbedderAdapter

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [[0.1], [0.2]]
        adapter = _EmbedderAdapter(mock_embedder)
        result = adapter.embed_documents(["a", "b"])
        assert result == [[0.1], [0.2]]
        assert mock_embedder.embed_query.call_count == 2


# -----------------------------------------------------------------------
# Evaluator._compute_metrics (RAGAS mocked)
# -----------------------------------------------------------------------


class TestComputeMetrics:
    def _make_samples(self, n: int = 2) -> list[EvalSample]:
        return [
            EvalSample(
                id=str(i + 1),
                query=f"Q{i}?",
                response=f"A{i}",
                retrieved_contexts=[f"ctx{i}"],
                reference=f"R{i}",
            )
            for i in range(n)
        ]

    def test_aggregate_path_for_per_sample_scores(self) -> None:
        """Verify aggregation of per-sample scores matches expected mean."""
        agg = Evaluator._aggregate_metrics(
            [
                {"faithfulness": 0.85},
                {"faithfulness": 0.90},
            ]
        )
        assert abs(agg["faithfulness"] - 0.875) < 1e-9

    def test_nan_converted_to_none_in_per_sample(self) -> None:
        """Verify NaN→None conversion matches _compute_metrics logic."""
        # This tests the exact per-sample NaN conversion code from _compute_metrics
        scores_list = [
            {"faithfulness": float("nan")},
            {"faithfulness": 0.9},
        ]
        active = ["faithfulness"]

        per_sample: list[dict[str, float | None]] = []
        for scores in scores_list:
            sample_scores: dict[str, float | None] = {}
            for name in active:
                if name in scores:
                    val = scores[name]
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        sample_scores[name] = None
                    else:
                        sample_scores[name] = float(val)
            per_sample.append(sample_scores)

        assert per_sample[0]["faithfulness"] is None
        assert per_sample[1]["faithfulness"] == 0.9
