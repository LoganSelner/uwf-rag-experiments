"""Shared data types for the argobot-bench framework.

These are the contracts between pipeline stages. Every type is a
dataclass with well-defined fields. No business logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, Protocol, Required, TypedDict, runtime_checkable

# ---------------------------------------------------------------------------
# Shared aliases
# ---------------------------------------------------------------------------


class Message(TypedDict, total=False):
    """A single chat message in the neutral, OpenAI-style schema shared by
    every generator.

    Only ``role`` is required. ``content`` is a string (or ``None`` on an
    assistant turn that only carries tool calls); a tool-calling assistant turn
    additionally carries ``tool_calls`` and a tool-result turn carries
    ``tool_call_id``. It is a ``TypedDict`` (a plain ``dict`` at runtime), so the
    OpenAI generator still forwards the list verbatim while the other three
    translators get static shape-checking — see ``components/_tool_protocol``.
    """

    role: Required[str]
    content: NotRequired[str | None]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]


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
    """A piece of a document, produced by a chunker.

    ``content`` is the canonical text: it is stored, retrieved, shown to the
    generator, and scored by the evaluator. ``index_text`` is an optional,
    build-time-only override for the text that gets **embedded and
    lexically indexed** — set by a chunk enricher (e.g. contextual
    retrieval) to prepend situating context without polluting what the
    generator/evaluator sees. It is never persisted by the vector store, so
    retrieved chunks always carry ``index_text=None``.
    """

    content: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index_text: str | None = None

    @property
    def text_for_index(self) -> str:
        """Text used for embedding and lexical indexing.

        Falls back to ``content`` when no enricher has set ``index_text``.
        """
        return self.index_text if self.index_text is not None else self.content


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

    ``tool_calls`` and ``finish_reason`` are populated only by a
    tool-calling generator (``generate(prompt, tools=...)``); they are empty
    on the linear path, so adding them leaves linear behavior unchanged. When
    a model chooses to call a tool instead of answering, ``answer`` may be
    ``""`` and ``tool_calls`` is non-empty — the agent loop branches on that.
    """

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""


@runtime_checkable
class Queryable(Protocol):
    """Anything that can answer a question.

    Satisfied by RAGPipeline, AgentPipeline, and test mocks.
    The evaluator depends on this protocol, not on concrete
    pipeline classes.

    ``history`` is an optional, additive parameter for multi-turn evaluation:
    the evaluator owns conversation state and threads prior turns in as a
    ``list[Message]``. Single-turn callers omit it (``None``), so behavior is
    unchanged. The pipeline is otherwise a pure function of ``(question,
    history)`` — no internal session state — which keeps runs reproducible.
    """

    def query(
        self, question: str, history: list[Message] | None = None
    ) -> GenerationResult: ...


# ---------------------------------------------------------------------------
# Agent-side types
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A tool advertised to a tool-calling LLM.

    ``parameters`` is a JSON-Schema object describing the tool's arguments
    (e.g. ``{"type": "object", "properties": {"query": {"type": "string"}},
    "required": ["query"]}``). Each generator translates this into its
    provider's tool/function-declaration format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A single tool invocation requested by a tool-calling LLM.

    ``arguments`` is the **parsed** argument dict (not a JSON string) — each
    generator handles the string↔dict conversion at its own provider boundary,
    so the agent loop and tests work with an ergonomic dict. ``id`` correlates
    the call with its result turn; providers that omit ids (Ollama, Gemini)
    get a synthesized one.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
    """One evaluation sample in the format RAGAS expects.

    ``metadata`` carries the producing pipeline's ``GenerationResult.metadata``
    (e.g. the agent's iteration/tool-call/step trace, or a generator's
    model/token counts) through to the saved per-sample output. RAGAS ignores
    it — it exists so runs stay auditable from their artifacts.
    """

    id: str
    query: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_result_dict(self) -> dict[str, Any]:
        """Serialize to nested dict for JSONL output.

        Follows the input/output/scores convention used by AWS Bedrock and
        OpenAI Evals, plus a top-level ``metadata`` carrying the pipeline's
        per-query provenance (e.g. the agent's iteration/tool-call/step trace)
        so runs stay auditable from their artifacts.
        """
        return {
            "id": self.id,
            "input": {"query": self.query, "reference": self.reference},
            "output": {
                "response": self.response,
                "retrieved_contexts": self.retrieved_contexts,
            },
            "scores": self.scores,
            "metadata": self.metadata,
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
