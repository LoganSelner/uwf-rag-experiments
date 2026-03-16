"""Integration tests for the indexing and query pipelines.

Tests the full flow: PDF → ingest → chunk → embed → store → retrieve.
Uses a small test PDF and a lightweight embedding model to keep tests fast.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Trigger component registration
import components  # noqa: F401
from core.config import ExperimentConfig
from core.types import (
    Chunk,
    GenerationResult,
    IndexArtifact,
    RetrievedChunk,
)
from pipeline.indexing import IndexingPipeline
from pipeline.query import QueryPipeline

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample.pdf"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_test_config(sources_path: str | None = None) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    path = sources_path or str(SAMPLE_PDF)
    return ExperimentConfig.from_dict(
        {
            "name": "test",
            "pipeline_mode": "linear",
            "indexing": {
                "sources": [
                    {
                        "name": "test_doc",
                        "path": path,
                        "ingest": {"type": "pdf"},
                    }
                ],
                "chunking": {
                    "type": "recursive",
                    "params": {"chunk_size": 200, "chunk_overlap": 20},
                },
                "embedding": {
                    "type": "huggingface",
                    "params": {"model_name": "BAAI/bge-small-en-v1.5"},
                },
                "vectorstore": {
                    "type": "faiss",
                    "params": {"metric": "cosine"},
                },
            },
            "query": {
                "query_transform": {"type": "passthrough"},
                "retrieval": {
                    "type": "dense",
                    "top_k_retrieve": 5,
                    "top_k_final": 3,
                },
                "reranking": {"type": "none"},
            },
        }
    )


def _build_mock_index_artifact() -> IndexArtifact:
    """Build a mock IndexArtifact for query pipeline tests."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    chunks = [
        Chunk(content=f"chunk {i}", chunk_id=f"c{i}", metadata={"source_name": "test"})
        for i in range(5)
    ]
    mock_store = MagicMock()
    mock_store.search.return_value = [
        RetrievedChunk(chunk=chunks[i], score=0.9 - i * 0.1) for i in range(3)
    ]

    return IndexArtifact(
        vectorstore=mock_store,
        embedder=mock_embedder,
        stats={"total_chunks": 5},
    )


# -----------------------------------------------------------------------
# IndexingPipeline
# -----------------------------------------------------------------------


class TestIndexingPipeline:
    def test_from_config_constructs(self) -> None:
        """Pipeline constructs from config without error."""
        config = _make_test_config()
        pipeline = IndexingPipeline.from_config(config)
        assert pipeline is not None
        assert len(pipeline._ingestors) == 1
        assert pipeline._ingestors[0][0] == "test_doc"

    @pytest.mark.slow
    def test_build_produces_artifact(self) -> None:
        """Full build: ingest → chunk → embed → store → artifact."""
        config = _make_test_config()
        pipeline = IndexingPipeline.from_config(config)
        artifact = pipeline.build()

        assert isinstance(artifact, IndexArtifact)
        assert artifact.vectorstore is not None
        assert artifact.embedder is not None
        assert artifact.stats["total_documents"] == 2  # 2 PDF pages
        assert artifact.stats["total_chunks"] > 0
        assert artifact.stats["source_counts"]["test_doc"] == 2

    @pytest.mark.slow
    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        """Index can be saved and loaded from disk."""
        config = _make_test_config()
        pipeline = IndexingPipeline.from_config(config)
        artifact = pipeline.build()

        # Save
        cache_dir = str(tmp_path / "cache")
        pipeline.save_index(cache_dir)

        # Load into a fresh pipeline
        pipeline2 = IndexingPipeline.from_config(config)
        pipeline2._vectorstore.load(cache_dir)

        # Verify loaded store works
        query_vec = artifact.embedder.embed_query("grade forgiveness")
        results = pipeline2._vectorstore.search(query_vec, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    @pytest.mark.slow
    def test_run_or_load_cache(self, tmp_path: Path) -> None:
        """run_or_load_cache builds fresh, then loads from cache."""
        config = _make_test_config()

        # First call: builds fresh
        artifact1 = IndexingPipeline.run_or_load_cache(config, cache_dir=tmp_path)
        assert artifact1.stats.get("cached") is not True

        # Second call: loads from cache
        artifact2 = IndexingPipeline.run_or_load_cache(config, cache_dir=tmp_path)
        assert artifact2.stats.get("cached") is True

    @pytest.mark.slow
    def test_source_name_in_chunk_metadata(self) -> None:
        """Each chunk should carry the source_name from config."""
        config = _make_test_config()
        pipeline = IndexingPipeline.from_config(config)
        artifact = pipeline.build()

        # Search to get chunks back
        query_vec = artifact.embedder.embed_query("university policy")
        results = artifact.vectorstore.search(query_vec, top_k=5)
        for r in results:
            assert r.chunk.metadata["source_name"] == "test_doc"


# -----------------------------------------------------------------------
# QueryPipeline
# -----------------------------------------------------------------------


class TestQueryPipeline:
    def test_retrieval_only_mode(self) -> None:
        """Retrieval-only mode returns chunks with empty answer."""
        artifact = _build_mock_index_artifact()
        config = _make_test_config()

        pipeline = QueryPipeline.from_config(
            config.query, artifact, retrieval_only=True
        )
        result = pipeline.run("What is grade forgiveness?")

        assert isinstance(result, GenerationResult)
        assert result.answer == ""
        assert result.query == "What is grade forgiveness?"
        assert len(result.retrieved_chunks) > 0
        assert result.metadata.get("retrieval_only") is True

    def test_deduplication_across_queries(self) -> None:
        """When multiple queries return the same chunk, keep highest score."""
        artifact = _build_mock_index_artifact()
        config = _make_test_config()

        pipeline = QueryPipeline.from_config(
            config.query, artifact, retrieval_only=True
        )

        # Mock transformer to return multiple queries
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = ["query1", "query2"]
        pipeline._query_transformer = mock_transformer

        result = pipeline.run("test")
        # Chunks should be deduplicated
        chunk_ids = [r.chunk.chunk_id for r in result.retrieved_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.slow
    def test_end_to_end_retrieval_only(self) -> None:
        """Full integration: build index then query in retrieval-only mode."""
        config = _make_test_config()

        # Build index
        pipeline = IndexingPipeline.from_config(config)
        artifact = pipeline.build()

        # Query
        query_pipeline = QueryPipeline.from_config(
            config.query, artifact, retrieval_only=True
        )
        result = query_pipeline.run("What is the grade forgiveness policy?")

        assert isinstance(result, GenerationResult)
        assert result.answer == ""
        assert len(result.retrieved_chunks) > 0
        # At least one chunk should mention grade forgiveness
        contents = [r.chunk.content for r in result.retrieved_chunks]
        assert any("forgiveness" in c.lower() or "grade" in c.lower() for c in contents)

    def test_generation_mode_requires_generator(self) -> None:
        """Non-retrieval-only mode fails without generator configured."""
        from components.defaults import NoOpReranker, PassthroughQueryTransformer
        from components.retrievers import DenseRetriever

        artifact = _build_mock_index_artifact()
        retriever = DenseRetriever()
        retriever.set_embedder(artifact.embedder)
        retriever.set_vectorstore(artifact.vectorstore)

        # Construct directly with no generator/prompt — retrieval_only=False
        pipeline = QueryPipeline(
            query_transformer=PassthroughQueryTransformer(),
            retriever=retriever,
            reranker=NoOpReranker(),
            top_k_retrieve=5,
            top_k_final=3,
            generator=None,
            prompt_template=None,
            retrieval_only=False,
        )
        with pytest.raises(RuntimeError, match="Generator and prompt template"):
            pipeline.run("test")
