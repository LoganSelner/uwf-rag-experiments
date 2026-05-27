"""Shared data types for the argobot-bench framework.

These are the contracts between pipeline stages. Every type is a
dataclass with well-defined fields. No business logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class TransformedQuery:
    """A query string produced by a BaseQueryTransformer, with an optional
    branch hint for per-retriever routing inside a HybridRetriever.

    ``branch=None`` is the broadcast sentinel: the query is consumed by
    every child retriever. A non-None ``branch`` matches a child's
    ``name`` inside the hybrid — at runtime, HybridRetriever raises
    ``ValueError`` for branch hints that reference no child. Non-hybrid
    retrievers ignore the hint entirely.

    Used by HyDE (hypothetical doc tagged ``branch="dense"``; original,
    when included, tagged ``branch="bm25"``) and multi-query (all
    ``branch=None`` — broadcast).
    """

    text: str
    branch: str | None = None


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


@runtime_checkable
class Queryable(Protocol):
    """Anything that can answer a question.

    Satisfied by RAGPipeline, AgentPipeline, and test mocks.
    The evaluator depends on this protocol, not on concrete
    pipeline classes.
    """

    def query(self, question: str) -> GenerationResult: ...


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

    id: str
    query: str
    response: str
    retrieved_contexts: list[str]
    reference: str


@dataclass
class ScoredSample:
    """One evaluated sample with its per-metric scores.

    This is the unit of data written to per-run JSONL result files.
    """

    id: str
    query: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    scores: dict[str, float | None]

    def to_result_dict(self) -> dict[str, Any]:
        """Serialize to nested dict for JSONL output.

        Follows the input/output/scores convention used by
        AWS Bedrock and OpenAI Evals.
        """
        return {
            "id": self.id,
            "input": {"query": self.query, "reference": self.reference},
            "output": {
                "response": self.response,
                "retrieved_contexts": self.retrieved_contexts,
            },
            "scores": self.scores,
        }


@dataclass
class ExperimentResult:
    """Aggregated results from a full experiment run."""

    experiment_name: str
    metrics: dict[str, float]
    per_run_metrics: list[dict[str, float]] = field(default_factory=list)
    per_run_samples: list[list[ScoredSample]] = field(default_factory=list)
    num_runs: int = 1
    config_snapshot: dict[str, Any] = field(default_factory=dict)
