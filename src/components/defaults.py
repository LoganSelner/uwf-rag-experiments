"""Default (passthrough / no-op) component implementations.

These allow the pipeline to run with minimal config by providing
sensible defaults for optional stages. For example, if no reranker
is configured, the NoOpReranker passes chunks through unchanged.
"""

from __future__ import annotations

from typing import Any

from components.base import BaseMemory, BaseQueryTransformer, BaseReranker
from core.registry import registry
from core.types import RetrievedChunk


@registry.register("query_transform", "passthrough")
class PassthroughQueryTransformer(BaseQueryTransformer):
    """Returns the original query unchanged."""

    def transform(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        return [query]


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


@registry.register("memory", "none")
class NoMemory(BaseMemory):
    """No-op memory — always returns empty history."""

    def add_turn(self, role: str, content: str) -> None:
        pass

    def get_history(self) -> list[dict[str, str]]:
        return []

    def clear(self) -> None:
        pass


@registry.register("memory", "buffer_window")
class BufferWindowMemory(BaseMemory):
    """Keeps the last N turns of conversation history."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._window_size = self.config.get("window_size", 5)
        self._history: list[dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        return self._history[-self._window_size :]

    def clear(self) -> None:
        self._history.clear()
