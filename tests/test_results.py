"""Tests for the results data layer."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_testing.results import RunResult, collect_runs, load_run, runs_to_dataframe

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_SAMPLE_CONFIG: dict[str, object] = {
    "git_sha": "abc1234",
    "models": {
        "llm": {"model": "gpt-4"},
        "embeddings": {"model": "text-embedding-3-small"},
    },
    "index": {"chunk_size": 1000, "chunk_overlap": 100},
    "eval": {
        "retriever_type": "dense",
        "reranker_type": "none",
        "reranker_model": "",
        "retrieval_k": 20,
        "top_k": 3,
        "mmr_lambda": 0.5,
        "mmr_fetch_k": 20,
        "qa_path": "data/queries/test.csv",
    },
}


def _make_scored_run(
    run_dir: Path,
    *,
    metrics: dict[str, float] | None = None,
    config: dict[str, object] | None = None,
    n_predictions: int | None = None,
) -> None:
    """Create a minimal scored run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics or {"faithfulness": 0.8}), encoding="utf-8"
    )
    if config is not None:
        (run_dir / "config_used.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
    if n_predictions is not None:
        lines = [json.dumps({"q": f"q{i}"}) for i in range(n_predictions)]
        (run_dir / "predictions.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------


class TestLoadRun:
    def test_returns_none_without_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        run_dir.mkdir()
        assert load_run(run_dir) is None

    def test_loads_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _make_scored_run(run_dir, metrics={"faithfulness": 0.85, "relevancy": 0.9})

        result = load_run(run_dir)

        assert result is not None
        assert result.name == "run_a"
        assert result.metrics["faithfulness"] == 0.85
        assert result.metrics["relevancy"] == 0.9

    def test_loads_config(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _make_scored_run(run_dir, config=_SAMPLE_CONFIG)

        result = load_run(run_dir)

        assert result is not None
        assert result.config["eval"]["retriever_type"] == "dense"
        assert result.config["models"]["llm"]["model"] == "gpt-4"
        assert result.config["git_sha"] == "abc1234"

    def test_empty_config_when_no_yaml(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _make_scored_run(run_dir)

        result = load_run(run_dir)

        assert result is not None
        assert result.config == {}

    def test_counts_predictions(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _make_scored_run(run_dir, n_predictions=5)

        result = load_run(run_dir)

        assert result is not None
        assert result.n_samples == 5

    def test_none_samples_when_no_predictions(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _make_scored_run(run_dir)

        result = load_run(run_dir)

        assert result is not None
        assert result.n_samples is None


# ---------------------------------------------------------------------------
# collect_runs
# ---------------------------------------------------------------------------


class TestCollectRuns:
    def test_empty_directory(self, tmp_path: Path) -> None:
        assert collect_runs(tmp_path) == []

    def test_skips_unscored_dirs(self, tmp_path: Path) -> None:
        unscored = tmp_path / "run_a"
        unscored.mkdir()
        (unscored / "predictions.jsonl").write_text("{}\n")

        assert collect_runs(tmp_path) == []

    def test_skips_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a run dir")
        assert collect_runs(tmp_path) == []

    def test_collects_scored_runs(self, tmp_path: Path) -> None:
        _make_scored_run(tmp_path / "run_a", metrics={"f": 0.7})
        _make_scored_run(tmp_path / "run_b", metrics={"f": 0.9})

        results = collect_runs(tmp_path)

        assert len(results) == 2
        assert results[0].name == "run_a"
        assert results[1].name == "run_b"

    def test_sorted_chronologically(self, tmp_path: Path) -> None:
        _make_scored_run(tmp_path / "20240102_run")
        _make_scored_run(tmp_path / "20240101_run")

        results = collect_runs(tmp_path)

        assert results[0].name == "20240101_run"
        assert results[1].name == "20240102_run"

    def test_excludes_unscored_from_mix(self, tmp_path: Path) -> None:
        _make_scored_run(tmp_path / "scored_run")
        unscored = tmp_path / "unscored_run"
        unscored.mkdir()

        results = collect_runs(tmp_path)

        assert len(results) == 1
        assert results[0].name == "scored_run"


# ---------------------------------------------------------------------------
# runs_to_dataframe
# ---------------------------------------------------------------------------


class TestRunsToDataframe:
    def test_empty_list(self) -> None:
        df = runs_to_dataframe([])
        assert len(df) == 0

    def test_extracts_config_fields(self) -> None:
        record = RunResult(
            name="run_a",
            config=_SAMPLE_CONFIG,
            metrics={"faithfulness": 0.8},
            n_samples=5,
        )

        df = runs_to_dataframe([record])

        assert df.index[0] == "run_a"
        assert df.loc["run_a", "git_sha"] == "abc1234"
        assert df.loc["run_a", "retriever_type"] == "dense"
        assert df.loc["run_a", "reranker_type"] == "none"
        assert df.loc["run_a", "retrieval_k"] == 20
        assert df.loc["run_a", "top_k"] == 3
        assert df.loc["run_a", "mmr_lambda"] == 0.5
        assert df.loc["run_a", "mmr_fetch_k"] == 20
        assert df.loc["run_a", "llm_model"] == "gpt-4"
        assert df.loc["run_a", "embedding_model"] == "text-embedding-3-small"
        assert df.loc["run_a", "chunk_size"] == 1000
        assert df.loc["run_a", "chunk_overlap"] == 100
        assert df.loc["run_a", "qa_path"] == "data/queries/test.csv"
        assert df.loc["run_a", "n_samples"] == 5
        assert df.loc["run_a", "faithfulness"] == 0.8

    def test_empty_config_omits_config_columns(self) -> None:
        record = RunResult(
            name="run_a",
            config={},
            metrics={"faithfulness": 0.8},
            n_samples=None,
        )

        df = runs_to_dataframe([record])

        assert df.loc["run_a", "faithfulness"] == 0.8
        assert df.loc["run_a", "n_samples"] is None
        assert "retriever_type" not in df.columns

    def test_multiple_runs(self) -> None:
        records = [
            RunResult("run_a", {}, {"f": 0.7}, n_samples=3),
            RunResult("run_b", {}, {"f": 0.9}, n_samples=5),
        ]

        df = runs_to_dataframe(records)

        assert len(df) == 2
        assert df.loc["run_a", "f"] == 0.7
        assert df.loc["run_b", "f"] == 0.9
