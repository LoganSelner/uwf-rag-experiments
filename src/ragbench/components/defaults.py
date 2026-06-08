"""Default (passthrough / no-op) component implementations.

These allow the pipeline to run with minimal config by providing
sensible defaults for optional stages. For example, if no reranker
is configured, the NoOpReranker passes chunks through unchanged.
"""

from __future__ import annotations

from typing import Any

from ragbench.components.base import (
    BaseChunkEnricher,
    BaseMemory,
    BaseQueryTransformer,
    BaseReranker,
    ComponentParams,
)
from ragbench.core.registry import registry
from ragbench.core.types import (
    Chunk,
    Document,
    Message,
    RetrievedChunk,
    TransformedQuery,
)


@registry.register("query_transform", "passthrough")
class PassthroughQueryTransformer(BaseQueryTransformer):
    """Returns the original query unchanged."""

    def transform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[TransformedQuery]:
        return [TransformedQuery(text=query)]


@registry.register("reranking", "none")
class NoOpReranker(BaseReranker):
    """Passes chunks through without reranking, truncated to top_k."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


@registry.register("chunk_enricher", "none")
class NoOpChunkEnricher(BaseChunkEnricher):
    """Passes chunks through unchanged (no ``index_text`` set)."""

    def enrich(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> list[Chunk]:
        return chunks


@registry.register("memory", "none")
class NoMemory(BaseMemory):
    """No-op memory — always returns empty history."""

    def add_turn(self, role: str, content: str) -> None:
        pass

    def get_history(self) -> list[Message]:
        return []

    def clear(self) -> None:
        pass


@registry.register("memory", "buffer_window")
class BufferWindowMemory(BaseMemory):
    """Keeps the last N turns of conversation history.

    Registered and functional, but **not used by the evaluation loop**:
    multi-turn evaluation threads conversation history through
    ``Queryable.query(question, history)`` (the evaluator owns it as the single
    source of truth), so this remains an honest stub for a future stateful
    serving surface rather than a parallel state path inside the pipeline.
    """

    class Params(ComponentParams):
        window_size: int = 5

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._window_size = self.p.window_size
        self._history: list[Message] = []

    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[Message]:
        return self._history[-self._window_size :]

    def clear(self) -> None:
        self._history.clear()
