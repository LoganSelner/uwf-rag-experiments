"""Top-level RAG pipeline dispatcher.

Reads ``pipeline_mode`` from config and constructs the appropriate
pipeline (linear QueryPipeline or AgentPipeline). Manages index
building/caching.
"""

from __future__ import annotations

import logging

from core.config import ExperimentConfig
from core.types import GenerationResult
from pipeline.indexing import IndexArtifact, IndexingPipeline
from pipeline.query import QueryPipeline

logger = logging.getLogger(__name__)


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
