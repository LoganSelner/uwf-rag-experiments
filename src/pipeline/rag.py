"""Top-level RAG pipeline dispatcher.

Reads ``pipeline_mode`` from config and constructs the appropriate
pipeline (linear QueryPipeline or AgentPipeline). Manages index
building/caching. Provides ``run_experiment()`` as the CLI entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Trigger component registration
import components  # noqa: F401
from core.config import ExperimentConfig
from core.git import get_git_dirty, get_git_sha
from core.types import GenerationResult, IndexArtifact
from evaluation.evaluator import Evaluator
from evaluation.results import save_experiment
from pipeline.indexing import IndexingPipeline
from pipeline.query import QueryPipeline

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


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

    # Capture git state early — before a long-running experiment
    # can change the working tree.
    git_info = {"sha": get_git_sha(), "dirty": get_git_dirty()}

    # Build pipeline
    rag = RAGPipeline.from_config(config, no_cache=no_cache)

    # Run evaluation
    evaluator = Evaluator(config.evaluation)
    experiment_result = evaluator.evaluate(rag, experiment_name=config.name)

    # Save results
    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    exp_dir = save_experiment(config, experiment_result, results_dir, git_info=git_info)

    logger.info("Experiment '%s' complete. Results saved to %s", config.name, exp_dir)
    return experiment_result.metrics
