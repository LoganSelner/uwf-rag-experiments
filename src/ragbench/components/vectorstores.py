"""VectorStore implementations.

Each vectorstore stores embedded chunks and provides similarity search.
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
import pickle
import typing
from typing import Any

import faiss
import numpy as np

from ragbench.components.base import BaseVectorStore, ComponentParams
from ragbench.core.filtering import metadata_matches
from ragbench.core.registry import registry
from ragbench.core.types import Chunk, EmbeddedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


@registry.register("vectorstore", "faiss")
class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store with flat cosine similarity search.

    Stores chunk metadata alongside the FAISS index. Supports
    save/load for index caching and post-retrieval filtering
    (FAISS doesn't support native metadata filters).

    Config params:
        metric: Distance metric — "cosine" (default) or "l2"
    """

    class Params(ComponentParams):
        metric: str = "cosine"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._metric = self.p.metric
        _valid_metrics = {"cosine", "l2"}
        if self._metric not in _valid_metrics:
            raise ValueError(
                f"FAISSVectorStore metric must be one of {_valid_metrics}, "
                f"got '{self._metric}'"
            )
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

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize for cosine so inner product == cosine; no-op otherwise.

        ``IndexFlatIP`` computes a raw inner product, which equals cosine
        similarity only on unit-norm vectors. Some embedders (notably several
        cloud providers) don't return unit vectors, so the store normalizes here
        rather than trusting the upstream embedder — making ``metric="cosine"``
        correct for any embedder. A no-op for already-normalized inputs.
        """
        if self._metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        dim = len(chunks[0].embedding)
        self._ensure_index(dim)
        if self._index is None:
            raise RuntimeError("FAISS index was not created by _ensure_index()")

        vectors = self._prepare(
            np.array([ec.embedding for ec in chunks], dtype=np.float32)
        )
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

        vec = self._prepare(np.array([query_vector], dtype=np.float32))
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
        vec = self._prepare(np.array([query_vector], dtype=np.float32))
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
        """Check if a chunk's metadata matches all filter criteria.

        Delegates to the shared :func:`metadata_matches` so dense (FAISS) and
        sparse (BM25) post-filtering interpret a filter dict identically —
        scalar = equality, list/tuple/set = ``$in`` membership.
        """
        return metadata_matches(chunk.metadata, filters)

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


@registry.register("vectorstore", "chroma")
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store with native metadata filtering.

    Uses an in-memory ``EphemeralClient`` during the build phase.
    On ``save()``, data is copied to a ``PersistentClient`` at the
    target directory.  On ``load()``, a ``PersistentClient`` reopens
    the persisted collection.

    Config params:
        metric: Distance metric — "cosine" (default), "l2", or "ip".
        collection_name: ChromaDB collection name (default: "chunks").
    """

    _VALID_METRICS: typing.ClassVar[set[str]] = {"cosine", "l2", "ip"}

    class Params(ComponentParams):
        metric: str = "cosine"
        collection_name: str = "chunks"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._metric = self.p.metric
        if self._metric not in self._VALID_METRICS:
            raise ValueError(
                f"ChromaVectorStore metric must be one of "
                f"{self._VALID_METRICS}, got '{self._metric}'"
            )
        self._collection_name = self.p.collection_name

        import chromadb

        self._client: Any = chromadb.EphemeralClient()
        self._collection: Any = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric},
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Ensure all metadata values are ChromaDB-compatible primitives.

        ChromaDB accepts str, int, float, and bool.  Lists and dicts
        are serialised to JSON strings.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = _json.dumps(value)
        return sanitized

    @staticmethod
    def _build_where(filters: dict[str, Any]) -> dict[str, Any] | None:
        """Translate a ``{key: value}`` filter dict to ChromaDB ``where`` syntax.

        A scalar value becomes an equality match; a list/tuple/set value becomes
        an ``$in`` membership match (scoping to a group of values — e.g. several
        sources). Returns ``None`` when *filters* is empty.
        """
        if not filters:
            return None

        def clause(value: Any) -> dict[str, Any]:
            if isinstance(value, (list, tuple, set)):
                return {"$in": list(value)}
            return {"$eq": value}

        clauses: list[dict[str, Any]] = [{k: clause(v)} for k, v in filters.items()]
        if len(clauses) == 1:
            return clauses[0]
        conjunction: dict[str, Any] = {"$and": clauses}
        return conjunction

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        ids = [ec.chunk.chunk_id for ec in chunks]
        embeddings = [ec.embedding for ec in chunks]
        documents = [ec.chunk.content for ec in chunks]
        metadatas = [self._sanitize_metadata(ec.chunk.metadata) for ec in chunks]

        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        count = self._collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        where = self._build_where(filters) if filters else None

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for cid, dist, doc, meta in zip(
            ids, distances, documents, metadatas, strict=True
        ):
            score = self._distance_to_score(dist)
            chunk = Chunk(content=doc, chunk_id=cid, metadata=meta or {})
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=score, retrieval_method="dense")
            )

        return retrieved

    def _distance_to_score(self, distance: float) -> float:
        """Convert a ChromaDB distance to a score (higher = better)."""
        if self._metric == "cosine":
            return 1.0 - distance
        # l2 and ip: negate so higher is better
        return -distance

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        import chromadb

        # Copy all data from in-memory collection to a persistent one
        data = self._collection.get(
            include=["embeddings", "documents", "metadatas"],
        )

        persistent = chromadb.PersistentClient(path=str(path))
        # Delete any pre-existing collection so stale chunks from a prior
        # save do not persist alongside the current data.
        try:
            persistent.delete_collection(name=self._collection_name)
        except chromadb.errors.NotFoundError:
            pass
        pcol = persistent.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric},
        )
        if data["ids"]:
            batch_size = persistent.get_max_batch_size()
            for start in range(0, len(data["ids"]), batch_size):
                end = start + batch_size
                pcol.upsert(
                    ids=data["ids"][start:end],
                    embeddings=data["embeddings"][start:end],
                    documents=data["documents"][start:end],
                    metadatas=data["metadatas"][start:end],
                )
        logger.info(
            "Saved ChromaDB collection to %s (%d vectors)",
            path,
            len(data["ids"]),
        )

    def load(self, directory: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=directory)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric},
        )
