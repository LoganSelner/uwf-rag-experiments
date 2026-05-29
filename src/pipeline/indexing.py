"""Indexing pipeline.

Builds the vector index from source documents:
  sources → ingest → chunk → embed → vectorstore

Supports caching via index_fingerprint: if a cached index exists
for the same indexing config, it's loaded from disk instead of rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

from components.base import (
    BaseChunkEnricher,
    BaseChunker,
    BaseEmbedder,
    BaseIngestor,
    BaseLexicalIndex,
    BaseVectorStore,
)
from core.config import ExperimentConfig
from core.registry import registry
from core.types import Chunk, Document

logger = logging.getLogger(__name__)

INDEX_CACHE_DIR = Path("data/indexes")

# Key under which the BM25 lexical index lives inside
# ``IndexArtifact.auxiliary_stores``. Names the implementation (per
# ROADMAP §7.1) so a future SPLADE store can coexist under its own key.
BM25_AUX_KEY = "bm25"


@dataclass
class IndexArtifact:
    """Handoff between the indexing pipeline and the query pipeline.

    Carries the populated vectorstore and embedder (needed at query
    time to embed incoming queries). Also carries optional auxiliary
    indexes — BM25 for now, room for SPLADE / contextual-embedding
    secondary stores later — and stats about what was indexed.
    """

    vectorstore: BaseVectorStore
    embedder: BaseEmbedder
    auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore] = field(
        default_factory=dict
    )
    stats: dict[str, Any] = field(default_factory=dict)


class IndexingPipeline:
    """Orchestrates document ingestion, chunking, embedding, and storage."""

    def __init__(
        self,
        ingestors: list[tuple[str, str, BaseIngestor]],
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        sparse_index: BaseLexicalIndex | None = None,
        chunk_enricher: BaseChunkEnricher | None = None,
    ) -> None:
        self._ingestors = ingestors  # [(source_name, path, ingestor), ...]
        self._chunker = chunker
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._sparse_index = sparse_index
        self._chunk_enricher = chunk_enricher

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> IndexingPipeline:
        """Build an IndexingPipeline from an ExperimentConfig."""
        idx = config.indexing

        # Build per-source ingestors
        ingestors: list[tuple[str, str, BaseIngestor]] = []
        for source in idx.sources:
            ingestor_cls = registry.get("ingest", source.ingest.type)
            ingestor: BaseIngestor = ingestor_cls(config=source.ingest.params)
            ingestors.append((source.name, source.path, ingestor))

        # Build shared components
        chunker_cls = registry.get("chunking", idx.chunking.type)
        chunker: BaseChunker = chunker_cls(config=idx.chunking.params)

        embedder_cls = registry.get("embedding", idx.embedding.type)
        embedder: BaseEmbedder = embedder_cls(config=idx.embedding.params)

        vectorstore_cls = registry.get("vectorstore", idx.vectorstore.type)
        vectorstore: BaseVectorStore = vectorstore_cls(config=idx.vectorstore.params)

        sparse_index: BaseLexicalIndex | None = None
        if idx.sparse_index.type:
            sparse_cls = registry.get("sparse_index", idx.sparse_index.type)
            sparse_index = sparse_cls(config=idx.sparse_index.params)

        chunk_enricher: BaseChunkEnricher | None = None
        if idx.chunk_enricher.type:
            enricher_cls = registry.get("chunk_enricher", idx.chunk_enricher.type)
            chunk_enricher = enricher_cls(config=idx.chunk_enricher.params)

        return cls(
            ingestors=ingestors,
            chunker=chunker,
            embedder=embedder,
            vectorstore=vectorstore,
            sparse_index=sparse_index,
            chunk_enricher=chunk_enricher,
        )

    def build(self) -> IndexArtifact:
        """Run the full indexing pipeline.

        Ordering: ingest → chunk → enrich (if configured) → sparse index
        (if configured) → embed → vector store. The enricher may set each
        chunk's ``index_text`` (e.g. contextual retrieval), which both the
        sparse index and the embedder consume via ``Chunk.text_for_index``,
        so it must run before either. BM25 indexing still happens *before*
        embedding so a tokenizer/stemmer misconfiguration fails fast,
        before spending embedding API budget.
        """
        all_documents: list[Document] = []
        source_counts: dict[str, int] = {}

        # Ingest from each source
        for source_name, path, ingestor in self._ingestors:
            logger.info("Ingesting source '%s' from %s", source_name, path)
            docs = ingestor.ingest(path)
            for doc in docs:
                doc.metadata["source_name"] = source_name
            all_documents.extend(docs)
            source_counts[source_name] = len(docs)

        logger.info(
            "Ingested %d documents from %d sources",
            len(all_documents),
            len(self._ingestors),
        )

        # Chunk
        chunks: list[Chunk] = self._chunker.chunk(all_documents)
        logger.info("Produced %d chunks", len(chunks))

        # Enrich (e.g. contextual retrieval) — may set chunk.index_text,
        # which the sparse index and embedder consume below. Runs before
        # both so the enriched text is what gets indexed/embedded.
        if self._chunk_enricher is not None:
            chunks = self._chunk_enricher.enrich(all_documents, chunks)
            logger.info("Enriched %d chunks", len(chunks))

        # Sparse index (BM25 etc.) — built before embedding so a bad
        # tokenizer/stemmer config doesn't waste embedding API calls.
        if self._sparse_index is not None:
            self._sparse_index.add(chunks)
            logger.info("Added %d chunks to sparse index", len(chunks))

        # Embed
        embedded = self._embedder.embed_chunks(chunks)
        logger.info("Embedded %d chunks", len(embedded))

        # Store
        self._vectorstore.add(embedded)
        logger.info("Added %d vectors to store", len(embedded))

        stats: dict[str, Any] = {
            "total_documents": len(all_documents),
            "total_chunks": len(chunks),
            "source_counts": source_counts,
            "has_sparse_index": self._sparse_index is not None,
            "has_chunk_enricher": self._chunk_enricher is not None,
        }

        auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore] = {}
        if self._sparse_index is not None:
            auxiliary_stores[BM25_AUX_KEY] = self._sparse_index

        return IndexArtifact(
            vectorstore=self._vectorstore,
            embedder=self._embedder,
            auxiliary_stores=auxiliary_stores,
            stats=stats,
        )

    def save_index(self, directory: str) -> None:
        """Save the vector + sparse indexes to disk for caching."""
        self._vectorstore.save(directory)
        if self._sparse_index is not None:
            self._sparse_index.save(str(Path(directory) / BM25_AUX_KEY))

    @classmethod
    def run_or_load_cache(
        cls,
        config: ExperimentConfig,
        cache_dir: Path | None = None,
        no_cache: bool = False,
    ) -> IndexArtifact:
        """Build index or load from cache if fingerprint matches.

        Args:
            config: The experiment configuration.
            cache_dir: Override the default cache directory.
            no_cache: Force rebuild even if cache exists.

        Returns:
            IndexArtifact with populated vectorstore and embedder.
        """
        base_dir = cache_dir or INDEX_CACHE_DIR
        fingerprint = config.index_fingerprint()
        index_dir = base_dir / fingerprint

        if not no_cache and index_dir.exists():
            logger.info(
                "Loading cached index from %s (fingerprint: %s)",
                index_dir,
                fingerprint,
            )
            # Build pipeline to get embedder, vectorstore, and sparse
            # index instances; populate them from disk.
            pipeline = cls.from_config(config)
            pipeline._vectorstore.load(str(index_dir))

            auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore] = {}
            if pipeline._sparse_index is not None:
                pipeline._sparse_index.load(str(index_dir / BM25_AUX_KEY))
                auxiliary_stores[BM25_AUX_KEY] = pipeline._sparse_index

            return IndexArtifact(
                vectorstore=pipeline._vectorstore,
                embedder=pipeline._embedder,
                auxiliary_stores=auxiliary_stores,
                stats={"cached": True, "fingerprint": fingerprint},
            )

        logger.info("Building fresh index (fingerprint: %s)", fingerprint)
        pipeline = cls.from_config(config)
        artifact = pipeline.build()

        # Cache for future use
        pipeline.save_index(str(index_dir))
        logger.info("Cached index to %s", index_dir)

        artifact.stats["fingerprint"] = fingerprint
        return artifact
