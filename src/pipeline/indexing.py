"""Indexing pipeline.

Builds the vector index from source documents:
  sources → ingest → chunk → embed → vectorstore

Supports caching via index_fingerprint: if a cached index exists
for the same indexing config, it's loaded from disk instead of rebuilt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from components.base import BaseChunker, BaseEmbedder, BaseIngestor, BaseVectorStore
from core.config import ExperimentConfig
from core.registry import registry
from core.types import Chunk, Document, IndexArtifact

logger = logging.getLogger(__name__)

INDEX_CACHE_DIR = Path("data/indexes")


class IndexingPipeline:
    """Orchestrates document ingestion, chunking, embedding, and storage."""

    def __init__(
        self,
        ingestors: list[tuple[str, str, BaseIngestor]],
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
    ) -> None:
        self._ingestors = ingestors  # [(source_name, path, ingestor), ...]
        self._chunker = chunker
        self._embedder = embedder
        self._vectorstore = vectorstore

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

        return cls(
            ingestors=ingestors,
            chunker=chunker,
            embedder=embedder,
            vectorstore=vectorstore,
        )

    def build(self) -> IndexArtifact:
        """Run the full indexing pipeline: ingest → chunk → embed → store.

        Returns an IndexArtifact carrying the populated vectorstore
        and embedder (needed at query time).
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
        }

        return IndexArtifact(
            vectorstore=self._vectorstore,
            embedder=self._embedder,
            stats=stats,
        )

    def save_index(self, directory: str) -> None:
        """Save the vectorstore to disk for caching."""
        self._vectorstore.save(directory)

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
            # Build pipeline to get embedder and vectorstore instances
            pipeline = cls.from_config(config)
            pipeline._vectorstore.load(str(index_dir))
            return IndexArtifact(
                vectorstore=pipeline._vectorstore,
                embedder=pipeline._embedder,
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
