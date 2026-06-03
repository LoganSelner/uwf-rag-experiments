"""Tests for evaluation.results — experiment results writer."""

from __future__ import annotations

import json
from pathlib import Path

from ragbench.core.config import ExperimentConfig
from ragbench.core.types import ExperimentResult, ScoredSample
from ragbench.evaluation.results import save_experiment


def _make_config() -> ExperimentConfig:
    """Build a minimal real ExperimentConfig for save tests."""
    return ExperimentConfig.from_dict(
        {
            "name": "test_exp",
            "pipeline_mode": "linear",
            "indexing": {
                "sources": [{"name": "s", "path": "/f", "ingest": {"type": "pdf"}}],
                "chunking": {"type": "recursive"},
                "embedding": {"type": "huggingface"},
                "vectorstore": {"type": "faiss"},
            },
            "query": {
                "retrieval": {"type": "dense"},
                "generator": {"provider": "ollama", "model_name": "m"},
                "prompt": {"type": "chat"},
            },
            "evaluation": {"dataset": "d.jsonl"},
        }
    )


class TestSaveExperiment:
    def test_saves_summary_json(self, tmp_path: Path) -> None:
        config = _make_config()
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"acc": 0.9},
        )
        git_info = {"sha": "abc", "dirty": False}
        save_experiment(config, result, tmp_path, git_info=git_info)

        summary_path = tmp_path / "test_exp" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["experiment_name"] == "test_exp"
        assert summary["metrics"]["acc"] == 0.9
        assert summary["git_sha"] == "abc"
        assert summary["git_dirty"] is False
        assert "timestamp" in summary
        # config_snapshot is built from config, not from result
        assert summary["config_snapshot"]["name"] == "test_exp"

    def test_config_snapshot_fields(self, tmp_path: Path) -> None:
        config = _make_config()
        result = ExperimentResult(experiment_name="test_exp", metrics={})
        save_experiment(config, result, tmp_path)

        summary = json.loads((tmp_path / "test_exp" / "summary.json").read_text())
        snap = summary["config_snapshot"]

        assert snap["pipeline_mode"] == "linear"
        assert snap["chunking_type"] == "recursive"
        assert snap["embedding_type"] == "huggingface"
        assert snap["vectorstore_type"] == "faiss"
        assert snap["query_transform_type"] == "passthrough"
        assert snap["retrieval_type"] == "dense"
        assert snap["top_k_retrieve"] == 10
        assert snap["top_k_final"] == 5
        assert snap["reranking_type"] == "none"
        assert snap["generation_type"] == "ollama"
        assert snap["generation_model"] == "m"

    def test_saves_config_yaml(self, tmp_path: Path) -> None:
        config = _make_config()
        result = ExperimentResult(experiment_name="test_exp", metrics={})
        save_experiment(config, result, tmp_path)

        config_path = tmp_path / "test_exp" / "config.yaml"
        assert config_path.exists()

    def test_saves_run_jsonl(self, tmp_path: Path) -> None:
        config = _make_config()
        sample = ScoredSample(
            id="1",
            query="Q",
            response="A",
            retrieved_contexts=["ctx"],
            reference="R",
            scores={"f": 0.9},
        )
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"f": 0.9},
            per_run_samples=[[sample]],
        )
        save_experiment(config, result, tmp_path)

        run_path = tmp_path / "test_exp" / "run_1.jsonl"
        assert run_path.exists()
        line = json.loads(run_path.read_text().strip())
        assert line["id"] == "1"
        assert line["scores"]["f"] == 0.9

    def test_run_jsonl_persists_metadata(self, tmp_path: Path) -> None:
        """Per-query pipeline provenance (e.g. the agent loop trace) reaches the
        saved run_*.jsonl so agent runs are auditable from their artifacts."""
        config = _make_config()
        sample = ScoredSample(
            id="1",
            query="Q",
            response="A",
            retrieved_contexts=["ctx"],
            reference="R",
            scores={"f": 0.9},
            metadata={
                "mode": "agent",
                "iterations": 2,
                "num_tool_calls": 1,
                "steps": [{"action": "tool_call", "tool_name": "knowledge_base"}],
            },
        )
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"f": 0.9},
            per_run_samples=[[sample]],
        )
        save_experiment(config, result, tmp_path)

        line = json.loads((tmp_path / "test_exp" / "run_1.jsonl").read_text().strip())
        assert line["metadata"]["mode"] == "agent"
        assert line["metadata"]["iterations"] == 2
        assert line["metadata"]["steps"][0]["tool_name"] == "knowledge_base"

    def test_cleans_stale_run_files(self, tmp_path: Path) -> None:
        """When re-running with fewer runs, old run files are removed."""
        config = _make_config()
        exp_dir = tmp_path / "test_exp"
        exp_dir.mkdir()

        # Simulate a previous 3-run experiment
        (exp_dir / "run_1.jsonl").write_text("old")
        (exp_dir / "run_2.jsonl").write_text("old")
        (exp_dir / "run_3.jsonl").write_text("old")
        (exp_dir / "summary.json").write_text("{}")

        # Save a 1-run experiment
        sample = ScoredSample(
            id="1",
            query="Q",
            response="A",
            retrieved_contexts=["ctx"],
            reference="R",
            scores={"f": 0.8},
        )
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"f": 0.8},
            per_run_samples=[[sample]],
        )
        save_experiment(config, result, tmp_path)

        # run_1 should exist with new content
        assert (exp_dir / "run_1.jsonl").exists()
        assert "old" not in (exp_dir / "run_1.jsonl").read_text()

        # Stale files should be gone
        assert not (exp_dir / "run_2.jsonl").exists()
        assert not (exp_dir / "run_3.jsonl").exists()

    def test_returns_exp_dir_path(self, tmp_path: Path) -> None:
        config = _make_config()
        result = ExperimentResult(experiment_name="test_exp", metrics={})
        exp_dir = save_experiment(config, result, tmp_path)
        assert exp_dir == tmp_path / "test_exp"

    def test_git_info_defaults(self, tmp_path: Path) -> None:
        """Without git_info, summary uses safe defaults."""
        config = _make_config()
        result = ExperimentResult(experiment_name="test_exp", metrics={})
        save_experiment(config, result, tmp_path)

        summary = json.loads((tmp_path / "test_exp" / "summary.json").read_text())
        assert summary["git_sha"] == "unknown"
        assert summary["git_dirty"] is False
