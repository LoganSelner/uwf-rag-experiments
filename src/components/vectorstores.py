"""VectorStore implementations.

Each vectorstore stores embedded chunks and provides similarity search.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import faiss
import numpy as np

from components.base import BaseVectorStore
from core.registry import registry
from core.types import Chunk, EmbeddedChunk, RetrievedChunk


@registry.register("vectorstore", "faiss")
class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store with flat cosine similarity search.

    Stores chunk metadata alongside the FAISS index. Supports
    save/load for index caching and post-retrieval filtering
    (FAISS doesn't support native metadata filters).

    Config params:
        metric: Distance metric — "cosine" (default) or "l2"
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._metric: str = self.config.get("metric", "cosine")
        self._index: faiss.IndexFlatIP | faiss.IndexFlatL2 | None = None
        self._chunks: list[Chunk] = []
        self._dim: int | None = None

    def _ensure_index(self, dim: int) -> None:
        """Lazily create the FAISS index on first add()."""
        if self._index is not None:
            return
        self._dim = dim
        if self._metric == "cosine":
            # Inner product on L2-normalized vectors = cosine similarity
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        dim = len(chunks[0].embedding)
        self._ensure_index(dim)
        if self._index is None:
            raise RuntimeError("FAISS index was not created by _ensure_index()")

        vectors = np.array([ec.embedding for ec in chunks], dtype=np.float32)
        self._index.add(vectors)
        self._chunks.extend(ec.chunk for ec in chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._index is None or self._index.ntotal == 0:
            return []

        if filters:
            return self._filtered_search(query_vector, top_k, filters)

        vec = np.array([query_vector], dtype=np.float32)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(idx)],
                    score=-float(score) if self._metric == "l2" else float(score),
                    retrieval_method="dense",
                )
            )
        return results

    def _filtered_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """Over-retrieve, apply metadata filters, return top_k.

        FAISS has no native filtering. We retrieve extra candidates
        (4x top_k, minimum 50), filter by metadata, then truncate.
        """
        if self._index is None:
            raise RuntimeError("FAISS index is not initialized — call add() first")
        fetch_k = min(max(top_k * 4, 50), self._index.ntotal)
        vec = np.array([query_vector], dtype=np.float32)
        scores, indices = self._index.search(vec, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            chunk = self._chunks[int(idx)]
            if self._matches_filters(chunk, filters):
                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=-float(score) if self._metric == "l2" else float(score),
                        retrieval_method="dense",
                    )
                )
                if len(results) >= top_k:
                    break
        return results

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check if a chunk's metadata matches all filter criteria."""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            raise RuntimeError("Cannot save — FAISS index is not initialized")
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump({"metric": self._metric, "dim": self._dim}, f)

    def load(self, directory: str) -> None:
        path = Path(directory)

        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        with open(path / "meta.pkl", "rb") as f:
            meta: dict[str, Any] = pickle.load(f)
        self._metric = meta["metric"]
        self._dim = meta["dim"]
