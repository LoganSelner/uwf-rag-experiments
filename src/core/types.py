"""Shared data types for the argobot-bench framework.

These are the contracts between pipeline stages. Every type is a
dataclass with well-defined fields. No business logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Indexing-side types
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Raw document produced by an ingestor."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document, produced by a chunker."""

    content: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector, produced by an embedder."""

    chunk: Chunk
    embedding: list[float]


# ---------------------------------------------------------------------------
# Query-side types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk retrieved from a vectorstore, with a relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str = ""


@dataclass
class GenerationResult:
    """The output of a pipeline run (linear or agent).

    This is the common type consumed by the evaluator. Both
    QueryPipeline and AgentPipeline produce this.
    """

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent-side types
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result from a tool execution in an agent pipeline."""

    tool_name: str
    content: str
    success: bool
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """One step in an agent's reasoning loop, for logging/debugging."""

    step_number: int
    action: str
    agent_name: str = ""
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """One evaluation sample in the format RAGAS expects."""

    query: str
    response: str
    retrieved_contexts: list[str]
    reference: str


@dataclass
class ExperimentResult:
    """Aggregated results from a full experiment run."""

    experiment_name: str
    metrics: dict[str, float]
    per_run_metrics: list[dict[str, float]] = field(default_factory=list)
    num_runs: int = 1
    config_snapshot: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline artifacts
# ---------------------------------------------------------------------------


@dataclass
class IndexArtifact:
    """Handoff between the indexing pipeline and the query pipeline.

    Carries the populated vectorstore and embedder (needed at query
    time to embed incoming queries). Also carries stats about what
    was indexed.
    """

    vectorstore: Any  # BaseVectorStore — typed as Any to avoid circular import
    embedder: Any  # BaseEmbedder — typed as Any to avoid circular import
    stats: dict[str, Any] = field(default_factory=dict)
