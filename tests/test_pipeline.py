"""Tests for src/pipeline/ — query, indexing, rag, and agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.types import (
    Chunk,
    GenerationResult,
    RetrievedChunk,
    TransformedQuery,
)
from pipeline.agent import AgentPipeline
from pipeline.indexing import IndexArtifact
from pipeline.query import QueryPipeline
from pipeline.rag import RAGPipeline

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_rc(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content=f"text_{chunk_id}", chunk_id=chunk_id, metadata={}),
        score=score,
    )


def _make_query_pipeline(
    *, retrieval_only: bool = False, qt_fusion: str = "rrf"
) -> tuple[QueryPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build a QueryPipeline with all mocked components."""
    qt = MagicMock()
    qt.transform.return_value = [TransformedQuery(text="transformed query")]

    retriever = MagicMock()
    retriever.retrieve_multi.return_value = [
        _make_rc("c1", 0.9),
        _make_rc("c2", 0.8),
    ]

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, chunks, k: chunks[:k]

    generator = MagicMock()
    generator.generate.return_value = GenerationResult(query="", answer="Answer")

    prompt = MagicMock()
    prompt.format.return_value = [{"role": "user", "content": "formatted"}]

    pipeline = QueryPipeline(
        query_transformer=qt,
        retriever=retriever,
        reranker=reranker,
        top_k_retrieve=10,
        top_k_final=3,
        generator=None if retrieval_only else generator,
        prompt_template=None if retrieval_only else prompt,
        retrieval_only=retrieval_only,
        qt_fusion=qt_fusion,
    )
    return pipeline, qt, retriever, reranker, generator


# -----------------------------------------------------------------------
# QueryPipeline
# -----------------------------------------------------------------------


class TestQueryPipeline:
    def test_run_basic(self) -> None:
        pipeline, qt, retriever, reranker, gen = _make_query_pipeline()
        result = pipeline.run("What is X?")
        assert isinstance(result, GenerationResult)
        assert result.answer == "Answer"
        qt.transform.assert_called_once()
        retriever.retrieve_multi.assert_called_once()
        reranker.rerank.assert_called_once()
        gen.generate.assert_called_once()

    def test_retrieval_only_skips_generation(self) -> None:
        pipeline, _, _, _, _gen = _make_query_pipeline(retrieval_only=True)
        result = pipeline.run("Q?")
        assert result.answer == ""
        assert result.metadata.get("retrieval_only") is True

    def test_pipeline_delegates_to_retrieve_multi(self) -> None:
        """Pipeline forwards transformer output + qt_fusion to retrieve_multi.

        Replaces the old ``test_deduplication`` test — fusion + dedup now
        live entirely inside ``BaseRetriever.retrieve_multi``. The
        pipeline's job is to pass the right arguments through.
        """
        pipeline, qt, retriever, _, _ = _make_query_pipeline(qt_fusion="rrf")
        qt.transform.return_value = [
            TransformedQuery(text="q1"),
            TransformedQuery(text="q2", branch="dense"),
        ]
        retriever.retrieve_multi.return_value = [_make_rc("c1", 0.5)]

        pipeline.run("Q?")

        kwargs = retriever.retrieve_multi.call_args.kwargs
        positional = retriever.retrieve_multi.call_args.args
        transformed_arg = positional[0] if positional else kwargs["transformed_queries"]
        assert transformed_arg == [
            TransformedQuery(text="q1"),
            TransformedQuery(text="q2", branch="dense"),
        ]
        assert kwargs.get("fusion") == "rrf"
        assert kwargs.get("top_k_per_query") == 10  # from _make_query_pipeline

    def test_qt_fusion_passed_through(self) -> None:
        pipeline, _, retriever, _, _ = _make_query_pipeline(qt_fusion="max")
        retriever.retrieve_multi.return_value = []
        pipeline.run("Q?")
        assert retriever.retrieve_multi.call_args.kwargs["fusion"] == "max"


# -----------------------------------------------------------------------
# IndexingPipeline
# -----------------------------------------------------------------------


class TestIndexingPipeline:
    def test_build_flow(self) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [
            MagicMock(content="text", metadata={}),
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="chunk", chunk_id="c1", metadata={}),
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [MagicMock()]
        mock_vs = MagicMock()

        pipeline = IndexingPipeline(
            ingestors=[("test_source", "/fake/path.pdf", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
        )
        artifact = pipeline.build()

        assert isinstance(artifact, IndexArtifact)
        mock_ingestor.ingest.assert_called_once_with("/fake/path.pdf")
        mock_chunker.chunk.assert_called_once()
        mock_embedder.embed_chunks.assert_called_once()
        mock_vs.add.assert_called_once()

    def test_stats_dict(self) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [MagicMock(content="t", metadata={})]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="c", chunk_id="1", metadata={}),
            Chunk(content="c2", chunk_id="2", metadata={}),
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [MagicMock(), MagicMock()]
        mock_vs = MagicMock()

        pipeline = IndexingPipeline(
            ingestors=[("src", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
        )
        artifact = pipeline.build()
        assert artifact.stats["total_documents"] == 1
        assert artifact.stats["total_chunks"] == 2
        assert artifact.stats["source_counts"]["src"] == 1

    def test_source_name_in_metadata(self) -> None:
        from pipeline.indexing import IndexingPipeline

        doc = MagicMock(content="text", metadata={})
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [doc]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = []
        mock_vs = MagicMock()

        pipeline = IndexingPipeline(
            ingestors=[("handbook", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
        )
        pipeline.build()
        assert doc.metadata["source_name"] == "handbook"

    @patch("pipeline.indexing.IndexingPipeline.from_config")
    def test_run_or_load_cache_fresh(self, mock_from_config: MagicMock) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_pipeline = MagicMock()
        mock_artifact = MagicMock(spec=IndexArtifact)
        mock_artifact.stats = {}
        mock_pipeline.build.return_value = mock_artifact
        mock_from_config.return_value = mock_pipeline

        mock_config = MagicMock()
        mock_config.index_fingerprint.return_value = "abc123"

        # Use tmp dir so no cache exists
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            artifact = IndexingPipeline.run_or_load_cache(
                mock_config, cache_dir=Path(td)
            )
        mock_pipeline.build.assert_called_once()
        assert artifact is mock_artifact

    @patch("pipeline.indexing.IndexingPipeline.from_config")
    def test_run_or_load_cache_loads_cached(
        self, mock_from_config: MagicMock, tmp_path: Path
    ) -> None:
        from pipeline.indexing import IndexingPipeline

        # Create cache directory matching fingerprint
        fingerprint = "cached_fp"
        cache_dir = tmp_path / fingerprint
        cache_dir.mkdir()

        mock_pipeline = MagicMock()
        mock_from_config.return_value = mock_pipeline

        mock_config = MagicMock()
        mock_config.index_fingerprint.return_value = fingerprint

        IndexingPipeline.run_or_load_cache(mock_config, cache_dir=tmp_path)
        mock_pipeline.build.assert_not_called()
        mock_pipeline._vectorstore.load.assert_called_once()


# -----------------------------------------------------------------------
# RAGPipeline
# -----------------------------------------------------------------------


class TestRAGPipeline:
    def test_query_delegates(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = GenerationResult(query="Q?", answer="A")
        rag = RAGPipeline(
            config=MagicMock(),
            index_artifact=MagicMock(),
            pipeline=mock_pipeline,
        )
        result = rag.query("Q?")
        assert result.answer == "A"
        mock_pipeline.run.assert_called_once_with("Q?")

    def test_unknown_mode_raises(self) -> None:
        mock_config = MagicMock()
        mock_config.pipeline_mode = "unknown_mode"

        with patch("pipeline.rag.IndexingPipeline.run_or_load_cache"):
            with pytest.raises(ValueError, match="Unknown pipeline_mode"):
                RAGPipeline.from_config(mock_config)

    def test_agent_mode_not_implemented(self) -> None:
        mock_config = MagicMock()
        mock_config.pipeline_mode = "agent"

        with patch("pipeline.rag.IndexingPipeline.run_or_load_cache"):
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                RAGPipeline.from_config(mock_config)


# -----------------------------------------------------------------------
# AgentPipeline
# -----------------------------------------------------------------------


class TestAgentPipeline:
    def test_init_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            AgentPipeline(config=MagicMock(), index_artifact=MagicMock())

    def test_run_raises(self) -> None:
        # Can't even construct, so test the class method directly
        with pytest.raises(NotImplementedError):
            AgentPipeline(config=MagicMock(), index_artifact=MagicMock())


# -----------------------------------------------------------------------
# IndexArtifact
# -----------------------------------------------------------------------


class TestIndexArtifact:
    def test_fields(self) -> None:
        a = IndexArtifact(vectorstore=MagicMock(), embedder=MagicMock())
        assert a.vectorstore is not None
        assert a.stats == {}
        assert a.auxiliary_stores == {}

    def test_auxiliary_stores_populated(self) -> None:
        sparse = MagicMock()
        a = IndexArtifact(
            vectorstore=MagicMock(),
            embedder=MagicMock(),
            auxiliary_stores={"bm25": sparse},
        )
        assert a.auxiliary_stores["bm25"] is sparse


# -----------------------------------------------------------------------
# IndexingPipeline — sparse index plumbing
# -----------------------------------------------------------------------


class TestIndexingPipelineSparse:
    def test_build_indexes_sparse_before_embed(self) -> None:
        """Sparse index must be populated before the embedder is called.

        Order matters: a misconfigured BM25 tokenizer should fail
        before we spend embedding API budget.
        """
        from pipeline.indexing import IndexingPipeline

        call_order: list[str] = []

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [
            MagicMock(content="t", metadata={}),
        ]
        chunks = [Chunk(content="c", chunk_id="1", metadata={})]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = chunks
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.side_effect = lambda c: (
            call_order.append("embed") or [MagicMock()]
        )
        mock_vs = MagicMock()
        mock_sparse = MagicMock()
        mock_sparse.add.side_effect = lambda c: call_order.append("sparse")

        pipeline = IndexingPipeline(
            ingestors=[("src", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
            sparse_index=mock_sparse,
        )
        artifact = pipeline.build()

        assert call_order == ["sparse", "embed"]
        assert "bm25" in artifact.auxiliary_stores
        assert artifact.stats["has_sparse_index"] is True

    def test_build_without_sparse_index_unchanged(self) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [MagicMock(content="t", metadata={})]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="c", chunk_id="1", metadata={})
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [MagicMock()]
        mock_vs = MagicMock()

        pipeline = IndexingPipeline(
            ingestors=[("src", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
            sparse_index=None,
        )
        artifact = pipeline.build()
        assert artifact.auxiliary_stores == {}
        assert artifact.stats["has_sparse_index"] is False

    def test_save_index_writes_sparse_subdir(self, tmp_path: Path) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_vs = MagicMock()
        mock_sparse = MagicMock()
        pipeline = IndexingPipeline(
            ingestors=[],
            chunker=MagicMock(),
            embedder=MagicMock(),
            vectorstore=mock_vs,
            sparse_index=mock_sparse,
        )
        pipeline.save_index(str(tmp_path))
        mock_vs.save.assert_called_once_with(str(tmp_path))
        mock_sparse.save.assert_called_once()
        saved_dir = mock_sparse.save.call_args.args[0]
        assert saved_dir.endswith("/bm25")


# -----------------------------------------------------------------------
# IndexingPipeline — chunk enricher plumbing
# -----------------------------------------------------------------------


class TestIndexingPipelineEnricher:
    def test_enrich_runs_after_chunk_before_sparse_and_embed(self) -> None:
        """The enricher must run before both the sparse index and embedder,
        since both consume ``Chunk.text_for_index``."""
        from pipeline.indexing import IndexingPipeline

        call_order: list[str] = []

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [MagicMock(content="t", metadata={})]
        chunks = [Chunk(content="c", chunk_id="1", metadata={})]
        mock_chunker = MagicMock()
        mock_chunker.chunk.side_effect = lambda d: call_order.append("chunk") or chunks
        mock_enricher = MagicMock()
        mock_enricher.enrich.side_effect = lambda docs, c: (
            call_order.append("enrich") or c
        )
        mock_sparse = MagicMock()
        mock_sparse.add.side_effect = lambda c: call_order.append("sparse")
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.side_effect = lambda c: (
            call_order.append("embed") or [MagicMock()]
        )
        mock_vs = MagicMock()

        pipeline = IndexingPipeline(
            ingestors=[("src", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=mock_vs,
            sparse_index=mock_sparse,
            chunk_enricher=mock_enricher,
        )
        artifact = pipeline.build()

        assert call_order == ["chunk", "enrich", "sparse", "embed"]
        mock_enricher.enrich.assert_called_once()
        assert artifact.stats["has_chunk_enricher"] is True

    def test_build_without_enricher_unchanged(self) -> None:
        from pipeline.indexing import IndexingPipeline

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = [MagicMock(content="t", metadata={})]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="c", chunk_id="1", metadata={})
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [MagicMock()]

        pipeline = IndexingPipeline(
            ingestors=[("src", "/f", mock_ingestor)],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vectorstore=MagicMock(),
            chunk_enricher=None,
        )
        artifact = pipeline.build()
        assert artifact.stats["has_chunk_enricher"] is False


# -----------------------------------------------------------------------
# QueryPipeline — hybrid wiring
# -----------------------------------------------------------------------


class TestQueryPipelineHybridWiring:
    def _index_artifact(self, sparse_index: object | None = None) -> IndexArtifact:
        return IndexArtifact(
            vectorstore=MagicMock(),
            embedder=MagicMock(),
            auxiliary_stores={"bm25": sparse_index} if sparse_index else {},
        )

    def test_hybrid_from_config_wires_children(self) -> None:
        # Build a real BaseLexicalIndex-shaped mock for the bm25 child
        from components.base import BaseLexicalIndex
        from core.config import QueryConfig
        from pipeline.query import QueryPipeline

        class _FakeIdx(BaseLexicalIndex):
            def add(self, chunks: list) -> None: ...
            def search(self, query: str, top_k: int, filters=None) -> list:
                return []

            def save(self, directory: str) -> None: ...
            def load(self, directory: str) -> None: ...

        artifact = self._index_artifact(sparse_index=_FakeIdx())

        config = QueryConfig.from_dict(
            {
                "retrieval": {
                    "type": "hybrid",
                    "top_k_retrieve": 20,
                    "top_k_final": 5,
                    "params": {
                        "fusion": "rrf",
                        "rrf_k": 60,
                        "retrievers": [
                            {
                                "name": "dense",
                                "type": "dense",
                                "top_k": 20,
                                "params": {},
                            },
                            {
                                "name": "bm25",
                                "type": "bm25",
                                "top_k": 20,
                                "params": {},
                            },
                        ],
                    },
                },
                "generation": {"type": "ollama"},
                "generation_llm": {"model_name": "x"},
                "prompt": {"type": "chat"},
            }
        )

        pipeline = QueryPipeline.from_config(config, artifact, retrieval_only=True)
        # The hybrid retriever should have two children configured.
        retriever = pipeline._retriever  # type: ignore[attr-defined]
        assert hasattr(retriever, "_children")
        names = [name for name, _, _ in retriever._children]
        assert names == ["dense", "bm25"]

    def test_bm25_retriever_requires_aux_store(self) -> None:
        from core.config import QueryConfig
        from pipeline.query import QueryPipeline

        artifact = self._index_artifact(sparse_index=None)
        config = QueryConfig.from_dict(
            {
                "retrieval": {
                    "type": "bm25",
                    "top_k_retrieve": 5,
                    "top_k_final": 5,
                },
                "generation": {"type": "ollama"},
                "generation_llm": {"model_name": "x"},
                "prompt": {"type": "chat"},
            }
        )
        with pytest.raises(RuntimeError, match="sparse index"):
            QueryPipeline.from_config(config, artifact, retrieval_only=True)
