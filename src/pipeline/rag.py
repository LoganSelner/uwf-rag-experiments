"""Top-level RAG pipeline dispatcher.

Reads ``pipeline_mode`` from config and constructs the appropriate
pipeline (linear QueryPipeline or AgentPipeline). Manages index
building/caching. Provides ``run_experiment()`` as the CLI entry point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
from typing import Any

# Trigger component registration
import components  # noqa: F401
from core.config import ExperimentConfig
from core.types import GenerationResult, IndexArtifact
from evaluation.evaluator import Evaluator
from pipeline.indexing import IndexingPipeline
from pipeline.query import QueryPipeline

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def _get_git_sha() -> str:
    """Return the short current git SHA, or ``'unknown'`` when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


class RAGPipeline:
    """Top-level dispatcher that manages index + query/agent pipeline.

    Usage:
        config = ExperimentConfig.from_yaml("configs/base.yaml")
        rag = RAGPipeline.from_config(config)
        result = rag.query("What is grade forgiveness?")
    """

    def __init__(
        self,
        config: ExperimentConfig,
        index_artifact: IndexArtifact,
        pipeline: QueryPipeline,
    ) -> None:
        self._config = config
        self._index_artifact = index_artifact
        self._pipeline = pipeline

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        no_cache: bool = False,
    ) -> RAGPipeline:
        """Build a RAGPipeline from config.

        Handles index building/caching and pipeline construction.
        """
        # Build or load index
        index_artifact = IndexingPipeline.run_or_load_cache(config, no_cache=no_cache)

        if config.pipeline_mode == "linear":
            retrieval_only = config.evaluation.mode == "retrieval_only"
            pipeline = QueryPipeline.from_config(
                config.query, index_artifact, retrieval_only=retrieval_only
            )
        elif config.pipeline_mode == "agent":
            # Import here to avoid circular import with stub
            from pipeline.agent import AgentPipeline

            raise NotImplementedError(
                "Agent pipeline is not yet implemented. "
                f"AgentPipeline class exists at {AgentPipeline.__module__} "
                "but requires implementation in a future branch."
            )
        else:
            raise ValueError(
                f"Unknown pipeline_mode: '{config.pipeline_mode}'. "
                "Expected 'linear' or 'agent'."
            )

        return cls(
            config=config,
            index_artifact=index_artifact,
            pipeline=pipeline,
        )

    @property
    def active_pipeline(self) -> QueryPipeline:
        """The currently active query pipeline."""
        return self._pipeline

    @property
    def index_artifact(self) -> IndexArtifact:
        """The index artifact (vectorstore + embedder)."""
        return self._index_artifact

    def query(self, question: str) -> GenerationResult:
        """Run a single query through the pipeline."""
        return self._pipeline.run(question)


def run_experiment(
    config_path: str,
    no_cache: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run a full experiment: build pipeline, evaluate, save results.

    This is the main CLI entry point.

    Args:
        config_path: Path to the experiment YAML config.
        no_cache: Force rebuild index even if cache exists.
        output_dir: Override the default results directory.

    Returns:
        Summary dict with aggregated metrics.
    """
    config = ExperimentConfig.from_yaml(config_path)
    logger.info("Running experiment: %s", config.name)

    # Build pipeline
    rag = RAGPipeline.from_config(config, no_cache=no_cache)

    # Run evaluation
    evaluator = Evaluator(config.evaluation)
    experiment_result = evaluator.evaluate(rag)

    # Save results
    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    _save_results(config, experiment_result, results_dir)

    logger.info(
        "Experiment '%s' complete. Results saved to %s",
        config.name,
        results_dir / config.name,
    )
    return experiment_result.metrics


def _save_results(
    config: ExperimentConfig,
    result: Any,
    base_dir: Path,
) -> None:
    """Save experiment results.

    results/<experiment_name>/
        summary.json    — aggregated metrics + config snapshot
        run_1.jsonl     — per-sample data for run 1
        run_2.jsonl
        ...
    """
    exp_dir = base_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = {
        "experiment_name": result.experiment_name,
        "metrics": result.metrics,
        "num_runs": result.num_runs,
        "config_snapshot": result.config_snapshot,
        "git_sha": _get_git_sha(),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Per-run results: JSONL with per-sample data
    for i, run_samples in enumerate(result.per_run_samples, 1):
        with open(exp_dir / f"run_{i}.jsonl", "w") as f:
            for sample in run_samples:
                f.write(json.dumps(sample.to_result_dict(), default=str) + "\n")
