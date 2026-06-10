"""Tests for src/evaluation/comparison.py — experiment comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ragbench.evaluation.comparison import (
    _flatten_dict,
    classify_decoupling,
    compare_experiments,
    decoupling_pair,
    decoupling_report,
    diff_configs,
    format_comparison_table,
    load_config,
    load_per_sample_scores,
    load_summary,
    resolve_metric_name,
    sort_rows,
)


def _write_summary(path: Path, name: str, metrics: dict) -> None:
    """Helper: write a minimal summary.json."""
    path.mkdir(parents=True, exist_ok=True)
    summary = {"experiment_name": name, "metrics": metrics}
    (path / "summary.json").write_text(json.dumps(summary))


# -----------------------------------------------------------------------
# load_summary
# -----------------------------------------------------------------------


class TestLoadSummary:
    def test_valid(self, tmp_path: Path) -> None:
        _write_summary(tmp_path / "exp", "my_exp", {"acc": 0.9})
        s = load_summary(tmp_path / "exp")
        assert s["experiment_name"] == "my_exp"
        assert s["metrics"]["acc"] == 0.9

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_summary(tmp_path / "nonexistent")


# -----------------------------------------------------------------------
# compare_experiments
# -----------------------------------------------------------------------


class TestCompareExperiments:
    def test_two_experiments(self, tmp_path: Path) -> None:
        _write_summary(
            tmp_path / "a",
            "exp_a",
            {"faithfulness": 0.9, "faithfulness_std": 0.01},
        )
        _write_summary(
            tmp_path / "b",
            "exp_b",
            {"faithfulness": 0.7, "faithfulness_std": 0.02},
        )
        rows = compare_experiments(
            [tmp_path / "a", tmp_path / "b"],
            metrics=["faithfulness"],
        )
        assert len(rows) == 2
        assert rows[0]["experiment"] == "exp_a"
        assert rows[0]["faithfulness"] == 0.9

    def test_skips_missing(self, tmp_path: Path) -> None:
        _write_summary(tmp_path / "a", "exp_a", {"faithfulness": 0.9})
        rows = compare_experiments(
            [tmp_path / "a", tmp_path / "missing"],
            metrics=["faithfulness"],
        )
        assert len(rows) == 1

    def test_custom_metrics(self, tmp_path: Path) -> None:
        _write_summary(tmp_path / "a", "exp_a", {"x": 1.0, "y": 2.0})
        rows = compare_experiments([tmp_path / "a"], metrics=["x"])
        assert "x" in rows[0]
        assert "y" not in rows[0]


# -----------------------------------------------------------------------
# format_comparison_table
# -----------------------------------------------------------------------


class TestFormatComparisonTable:
    def test_markdown_output(self) -> None:
        rows = [{"experiment": "exp", "faithfulness": 0.85, "faithfulness_std": None}]
        table = format_comparison_table(rows, metrics=["faithfulness"])
        assert "exp" in table
        assert "0.850" in table
        assert "|" in table

    def test_with_std(self) -> None:
        rows = [{"experiment": "e", "faithfulness": 0.9, "faithfulness_std": 0.05}]
        table = format_comparison_table(rows, metrics=["faithfulness"])
        assert "0.900" in table
        assert "0.050" in table

    def test_missing_value_dash(self) -> None:
        rows = [{"experiment": "e", "faithfulness": None, "faithfulness_std": None}]
        table = format_comparison_table(rows, metrics=["faithfulness"])
        assert "-" in table


# -----------------------------------------------------------------------
# resolve_metric_name
# -----------------------------------------------------------------------


class TestResolveMetricName:
    def test_full_name(self) -> None:
        assert resolve_metric_name("faithfulness", ["faithfulness"]) == "faithfulness"

    def test_short_name(self) -> None:
        result = resolve_metric_name("Faith.", ["faithfulness"])
        assert result == "faithfulness"

    def test_unknown_returns_none(self) -> None:
        assert resolve_metric_name("bogus", ["faithfulness"]) is None


# -----------------------------------------------------------------------
# sort_rows
# -----------------------------------------------------------------------


class TestSortRows:
    def test_descending_default(self) -> None:
        rows = [
            {"experiment": "a", "acc": 0.5},
            {"experiment": "b", "acc": 0.9},
        ]
        sorted_rows = sort_rows(rows, "acc")
        assert sorted_rows[0]["experiment"] == "b"

    def test_ascending(self) -> None:
        rows = [
            {"experiment": "a", "acc": 0.9},
            {"experiment": "b", "acc": 0.5},
        ]
        sorted_rows = sort_rows(rows, "acc", ascending=True)
        assert sorted_rows[0]["experiment"] == "b"

    def test_missing_values_last(self) -> None:
        rows: list[dict[str, Any]] = [
            {"experiment": "a", "acc": None},
            {"experiment": "b", "acc": 0.9},
        ]
        sorted_rows = sort_rows(rows, "acc")
        assert sorted_rows[0]["experiment"] == "b"
        assert sorted_rows[1]["experiment"] == "a"


# -----------------------------------------------------------------------
# _flatten_dict
# -----------------------------------------------------------------------


class TestFlattenDict:
    def test_simple(self) -> None:
        assert _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_deep(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        assert _flatten_dict(d) == {"a.b.c": 42}

    def test_lists_as_leaves(self) -> None:
        d = {"a": [1, 2, 3]}
        assert _flatten_dict(d) == {"a": [1, 2, 3]}


# -----------------------------------------------------------------------
# load_config / diff_configs
# -----------------------------------------------------------------------


class TestConfigDiff:
    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        config = {"name": "test", "chunk_size": 512}
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        loaded = load_config(exp_dir)
        assert loaded["name"] == "test"

    def test_load_config_fallback_to_snapshot(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        summary = {
            "experiment_name": "old",
            "metrics": {},
            "config_snapshot": {"name": "old", "mode": "linear"},
        }
        (exp_dir / "summary.json").write_text(json.dumps(summary))
        loaded = load_config(exp_dir)
        assert loaded["name"] == "old"
        assert loaded["mode"] == "linear"

    def test_diff_configs_detects_changes(self, tmp_path: Path) -> None:
        for name, chunk_size in [("a", 512), ("b", 1000)]:
            d = tmp_path / name
            d.mkdir()
            with open(d / "config.yaml", "w") as f:
                yaml.safe_dump({"name": name, "chunk_size": chunk_size}, f)
        diffs = diff_configs(tmp_path / "a", tmp_path / "b")
        assert "name" in diffs
        assert "chunk_size" in diffs
        assert diffs["chunk_size"] == (512, 1000)

    def test_diff_configs_identical(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            with open(d / "config.yaml", "w") as f:
                yaml.safe_dump({"k": "v"}, f)
        assert diff_configs(tmp_path / "a", tmp_path / "b") == {}


# -----------------------------------------------------------------------
# load_per_sample_scores
# -----------------------------------------------------------------------


class TestLoadPerSampleScores:
    def test_loads_scores(self, tmp_path: Path) -> None:
        for exp_name, score in [("exp_a", 0.8), ("exp_b", 0.9)]:
            d = tmp_path / exp_name
            d.mkdir()
            _write_summary(d, exp_name, {})
            sample = {
                "id": "1",
                "input": {"query": "Q?", "reference": "R"},
                "output": {"response": "A", "retrieved_contexts": []},
                "scores": {"faithfulness": score},
            }
            (d / "run_1.jsonl").write_text(json.dumps(sample) + "\n")

        rows = load_per_sample_scores(
            [tmp_path / "exp_a", tmp_path / "exp_b"],
            metrics=["faithfulness"],
        )
        assert len(rows) == 2
        assert rows[0]["faithfulness"] == 0.8
        assert rows[1]["faithfulness"] == 0.9

    def test_missing_run_file(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        _write_summary(d, "exp", {})
        # No run_1.jsonl
        rows = load_per_sample_scores([d], metrics=["faithfulness"])
        assert rows == []

    def test_corrupted_jsonl_line_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        _write_summary(d, "exp", {})
        good = json.dumps(
            {
                "id": "1",
                "input": {"query": "Q?", "reference": "R"},
                "output": {"response": "A", "retrieved_contexts": []},
                "scores": {"faithfulness": 0.8},
            }
        )
        good2 = json.dumps(
            {
                "id": "2",
                "input": {"query": "Q2?", "reference": "R2"},
                "output": {"response": "A2", "retrieved_contexts": []},
                "scores": {"faithfulness": 0.9},
            }
        )
        (d / "run_1.jsonl").write_text(good + "\nNOT VALID JSON\n" + good2 + "\n")
        rows = load_per_sample_scores([d], metrics=["faithfulness"])
        assert len(rows) == 2


# -----------------------------------------------------------------------
# Decoupling analysis (Phase E, item 12)
# -----------------------------------------------------------------------


def _write_run(path: Path, name: str, samples: list[dict[str, Any]]) -> None:
    """Write summary.json + run_1.jsonl with per-sample scores.

    Each item in *samples* is ``{"id", "query"?, "scores"}``, serialized in the
    ``to_result_dict`` layout that ``load_per_sample_scores`` reads.
    """
    _write_summary(path, name, {})
    lines = [
        json.dumps(
            {
                "id": s["id"],
                "input": {"query": s.get("query", ""), "reference": "R"},
                "output": {"response": "A", "retrieved_contexts": []},
                "scores": s["scores"],
            }
        )
        for s in samples
    ]
    (path / "run_1.jsonl").write_text("\n".join(lines) + "\n")


class TestClassifyDecoupling:
    def test_both_good(self) -> None:
        assert classify_decoupling(0.9, 0.85) == "both_good"

    def test_both_poor(self) -> None:
        assert classify_decoupling(0.2, 0.1) == "both_poor"

    def test_retrieval_good_answer_poor(self) -> None:
        assert classify_decoupling(0.9, 0.2) == "retrieval_good_answer_poor"

    def test_retrieval_poor_answer_good(self) -> None:
        assert classify_decoupling(0.2, 0.9) == "retrieval_poor_answer_good"

    def test_dead_band_is_mixed(self) -> None:
        # 0.55 sits in the (0.4, 0.7) dead-band → not called decoupled.
        assert classify_decoupling(0.55, 0.9) == "mixed"

    def test_none_is_incomplete(self) -> None:
        assert classify_decoupling(None, 0.9) == "incomplete"
        assert classify_decoupling(0.9, None) == "incomplete"

    def test_nan_is_incomplete(self) -> None:
        assert classify_decoupling(float("nan"), 0.9) == "incomplete"


class TestDecouplingReport:
    def test_counts_quadrants(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        _write_run(
            d,
            "exp",
            [
                {
                    "id": "1",
                    "scores": {"context_precision": 0.9, "answer_correctness": 0.9},
                },
                {
                    "id": "2",
                    "scores": {"context_precision": 0.9, "answer_correctness": 0.1},
                },
                {
                    "id": "3",
                    "scores": {"context_precision": 0.1, "answer_correctness": 0.9},
                },
                {
                    "id": "4",
                    "scores": {"context_precision": 0.1, "answer_correctness": 0.1},
                },
            ],
        )
        report = decoupling_report(d)
        assert report["experiment"] == "exp"
        counts = report["counts"]
        assert counts["both_good"] == 1
        assert counts["retrieval_good_answer_poor"] == 1
        assert counts["retrieval_poor_answer_good"] == 1
        assert counts["both_poor"] == 1
        assert len(report["samples"]) == 4

    def test_incomplete_when_answer_metric_absent(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        _write_run(d, "exp", [{"id": "1", "scores": {"context_precision": 0.9}}])
        report = decoupling_report(d)
        assert report["counts"]["incomplete"] == 1


class TestDecouplingPair:
    def test_surfaces_retrieval_up_answer_down(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        # Sample 1: retrieval improves (0.4→0.9) but the answer drops (0.8→0.5) →
        # decoupled. Sample 2: both improve → coupled.
        _write_run(
            a,
            "a",
            [
                {
                    "id": "1",
                    "scores": {"context_precision": 0.4, "answer_correctness": 0.8},
                },
                {
                    "id": "2",
                    "scores": {"context_precision": 0.4, "answer_correctness": 0.4},
                },
            ],
        )
        _write_run(
            b,
            "b",
            [
                {
                    "id": "1",
                    "scores": {"context_precision": 0.9, "answer_correctness": 0.5},
                },
                {
                    "id": "2",
                    "scores": {"context_precision": 0.9, "answer_correctness": 0.9},
                },
            ],
        )
        rows = decoupling_pair(a, b)
        assert len(rows) == 2
        # Most-decoupled first.
        assert rows[0]["sample_id"] == "1"
        assert rows[0]["decoupled"] is True
        assert rows[0]["delta_retrieval"] == pytest.approx(0.5)
        assert rows[0]["delta_answer"] == pytest.approx(-0.3)
        assert rows[1]["decoupled"] is False

    def test_drops_unshared_samples(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        _write_run(
            a,
            "a",
            [
                {
                    "id": "1",
                    "scores": {"context_precision": 0.5, "answer_correctness": 0.5},
                }
            ],
        )
        _write_run(
            b,
            "b",
            [
                {
                    "id": "2",
                    "scores": {"context_precision": 0.5, "answer_correctness": 0.5},
                }
            ],
        )
        assert decoupling_pair(a, b) == []
