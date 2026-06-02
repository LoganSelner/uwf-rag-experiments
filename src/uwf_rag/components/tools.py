"""Tool implementations for agent pipelines.

Tools are the actions an agent pipeline can take. Each wraps a capability behind
``BaseTool.execute(query) -> ToolResult`` and is advertised to the tool-calling
LLM via a uniform single-``query``-string schema (see :func:`tool_to_spec`).

Implemented:
- ``rag`` — searches the indexed knowledge base, returning the raw retrieved
  chunks as the observation so the agent's own LLM synthesizes the final answer.

Deferred (end of Phase D1): a ``web_search`` tool over an external API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uwf_rag.components.base import BaseTool
from uwf_rag.core.config import (
    AgentDefinitionConfig,
    ExperimentConfig,
    QueryTransformConfig,
)
from uwf_rag.core.registry import registry
from uwf_rag.core.types import RetrievedChunk, ToolResult, ToolSpec

if TYPE_CHECKING:
    from uwf_rag.pipeline.indexing import IndexArtifact

logger = logging.getLogger(__name__)

_DEFAULT_RAG_DESCRIPTION = (
    "Search the knowledge base of indexed source documents for passages "
    "relevant to a question. Use this whenever you need factual information "
    "from the documents to answer. Input: a natural-language search query."
)


def tool_to_spec(tool: BaseTool, description: str | None = None) -> ToolSpec:
    """Advertise a tool to a tool-calling LLM with a single ``query`` arg.

    Every tool in this harness takes one natural-language query string
    (``execute(query)``), so the JSON schema is uniform; only the name and
    description vary. Keeps JSON-schema authoring out of ``BaseTool``.
    """
    return ToolSpec(
        name=tool.name,
        description=tool.description if description is None else description,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language search query.",
                }
            },
            "required": ["query"],
        },
    )


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered observation for the agent."""
    if not chunks:
        return "No relevant passages were found in the knowledge base."
    return "\n\n".join(f"[{i}] {rc.chunk.content}" for i, rc in enumerate(chunks, 1))


@registry.register("tool", "rag")
class RAGSearchTool(BaseTool):
    """Searches the indexed knowledge base via a retrieval-only QueryPipeline.

    Wraps the experiment's query retrieval + rerank stack so the agent searches
    the *same* index and stack as the linear baseline — keeping linear-vs-agent
    comparable. Returns the top chunks verbatim as the observation and populates
    ``ToolResult.retrieved_chunks`` so the evaluator can still score retrieval in
    agent mode.

    The query transform is forced to ``passthrough``: the agent has already
    decided what to search for, so re-running the linear stack's transform
    (contextualizer/HyDE/...) inside the tool would double-transform and add a
    hidden LLM call. The wrapped pipeline is injected (duck-typed: anything with
    ``run(query) -> GenerationResult``), so this component never imports the
    pipeline layer at its interface.
    """

    def __init__(
        self,
        pipeline: Any,
        name: str = "knowledge_base",
        description: str = _DEFAULT_RAG_DESCRIPTION,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._name = name
        self._description = description

    @classmethod
    def from_config(
        cls,
        entry: AgentDefinitionConfig,
        experiment_config: ExperimentConfig,
        index_artifact: IndexArtifact,
    ) -> RAGSearchTool:
        """Build a retrieval-only QueryPipeline from the experiment's query stack.

        ``QueryPipeline`` is imported lazily here so this component module never
        depends on the pipeline layer at import time (keeping the dependency
        graph pointing inward).
        """
        from uwf_rag.pipeline.query import QueryPipeline

        query_config = experiment_config.query.model_copy(
            update={"query_transform": QueryTransformConfig(type="passthrough")},
        )
        pipeline = QueryPipeline.from_config(
            query_config, index_artifact, retrieval_only=True
        )
        return cls(
            pipeline=pipeline,
            name=entry.name or "knowledge_base",
            description=entry.description or _DEFAULT_RAG_DESCRIPTION,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, query: str) -> ToolResult:
        try:
            result = self._pipeline.run(query)
        except Exception as e:
            # A tool failure must become a recoverable observation, not a crash:
            # the agent can re-plan rather than the whole query dying.
            logger.warning(
                "RAG tool '%s' failed for query %r: %s", self._name, query, e
            )
            return ToolResult(
                tool_name=self._name,
                content=f"Knowledge base search failed: {e}",
                success=False,
                retrieved_chunks=[],
            )
        chunks = result.retrieved_chunks
        return ToolResult(
            tool_name=self._name,
            content=_format_chunks(chunks),
            success=True,
            retrieved_chunks=chunks,
            metadata={"num_chunks": len(chunks)},
        )
