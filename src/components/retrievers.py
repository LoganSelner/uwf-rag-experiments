"""Retriever implementations.

Each retriever uses an embedder and vectorstore (injected after
construction) to find relevant chunks for a query.
"""

from __future__ import annotations

from typing import Any

from components.base import BaseRetriever
from core.registry import registry
from core.types import RetrievedChunk


@registry.register("retrieval", "dense")
class DenseRetriever(BaseRetriever):
    """Dense (embedding-based) retriever.

    Embeds the query, searches the vectorstore, returns top_k results.
    Supports metadata filtering via default_filters (from config) merged
    with per-call filters.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._embedder is None:
            raise RuntimeError("Embedder not set. Call set_embedder() first.")
        if self._vectorstore is None:
            raise RuntimeError("VectorStore not set. Call set_vectorstore() first.")

        merged_filters = {**self._default_filters, **(filters or {})}
        query_vector = self._embedder.embed_query(query)
        return self._vectorstore.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=merged_filters if merged_filters else None,
        )
