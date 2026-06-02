"""Tests for src/pipeline/ — query, indexing, rag, and agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from uwf_rag.core.types import (
    Chunk,
    GenerationResult,
    RetrievedChunk,
    ToolCall,
    ToolResult,
    ToolSpec,
    TransformedQuery,
)
from uwf_rag.pipeline.agent import AgentPipeline
from uwf_rag.pipeline.indexing import IndexArtifact
from uwf_rag.pipeline.query import QueryPipeline
from uwf_rag.pipeline.rag import RAGPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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

    @patch("uwf_rag.pipeline.indexing.IndexingPipeline.from_config")
    def test_run_or_load_cache_fresh(self, mock_from_config: MagicMock) -> None:
        from uwf_rag.pipeline.indexing import IndexingPipeline

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

    @patch("uwf_rag.pipeline.indexing.IndexingPipeline.from_config")
    def test_run_or_load_cache_loads_cached(
        self, mock_from_config: MagicMock, tmp_path: Path
    ) -> None:
        from uwf_rag.pipeline.indexing import IndexingPipeline

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

        with patch("uwf_rag.pipeline.rag.IndexingPipeline.run_or_load_cache"):
            with pytest.raises(ValueError, match="Unknown pipeline_mode"):
                RAGPipeline.from_config(mock_config)

    def test_agent_mode_builds_agent_pipeline(self) -> None:
        mock_config = MagicMock()
        mock_config.pipeline_mode = "agent"
        agent = MagicMock()

        with patch("uwf_rag.pipeline.rag.IndexingPipeline.run_or_load_cache"):
            with patch(
                "uwf_rag.pipeline.rag.AgentPipeline.from_config", return_value=agent
            ) as mock_from:
                rag = RAGPipeline.from_config(mock_config)

        mock_from.assert_called_once()
        assert rag.active_pipeline is agent


# -----------------------------------------------------------------------
# AgentPipeline — single-agent ReAct loop
# -----------------------------------------------------------------------


def _fake_tool(name: str = "knowledge_base") -> MagicMock:
    """A BaseTool-shaped mock with a real ``.name`` string."""
    tool = MagicMock()
    tool.name = name  # assign after construction (MagicMock(name=...) is special)
    return tool


def _agent(
    generator: MagicMock,
    tools: list[MagicMock],
    *,
    max_iterations: int = 5,
    top_k_final: int = 3,
) -> AgentPipeline:
    specs = [ToolSpec(name=t.name, description="d") for t in tools]
    return AgentPipeline(
        reasoning_generator=generator,
        tools=tools,  # type: ignore[arg-type]
        tool_specs=specs,
        max_iterations=max_iterations,
        top_k_final=top_k_final,
        system_prompt="You are an advising agent.",
    )


class TestAgentPipeline:
    def test_reason_act_observe_finalize(self) -> None:
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                query="",
                answer="",
                tool_calls=[ToolCall("c1", "knowledge_base", {"query": "grades"})],
            ),
            GenerationResult(query="", answer="final answer", tool_calls=[]),
        ]
        tool = _fake_tool()
        tool.execute.return_value = ToolResult(
            tool_name="knowledge_base",
            content="ctx",
            success=True,
            retrieved_chunks=[_make_rc("c1", 0.9)],
        )

        result = _agent(gen, [tool]).run("How does grade forgiveness work?")

        assert gen.generate.call_count == 2
        tool.execute.assert_called_once_with("grades")
        assert result.answer == "final answer"
        assert [rc.chunk.chunk_id for rc in result.retrieved_chunks] == ["c1"]
        assert result.metadata["iterations"] == 2
        assert result.metadata["num_tool_calls"] == 1
        assert result.metadata["forced_final"] is False
        # First call advertised tools, and a tool-result turn was fed back.
        assert gen.generate.call_args_list[0].kwargs.get("tools") is not None

    def test_budget_exhaustion_forces_final_answer(self) -> None:
        # The model keeps calling tools; after max_iterations a final answer
        # is forced with tools=None (guaranteeing termination).
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c1", "knowledge_base", {})]
            ),
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c2", "knowledge_base", {})]
            ),
            GenerationResult(query="", answer="forced", tool_calls=[]),
        ]
        tool = _fake_tool()
        tool.execute.return_value = ToolResult(
            tool_name="knowledge_base", content="ctx", success=True
        )

        result = _agent(gen, [tool], max_iterations=2).run("Q?")

        assert gen.generate.call_count == 3
        assert gen.generate.call_args_list[-1].kwargs.get("tools") is None
        assert result.answer == "forced"
        assert result.metadata["forced_final"] is True
        assert result.metadata["iterations"] == 2

    def test_chunks_deduped_and_capped(self) -> None:
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c1", "knowledge_base", {})]
            ),
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c2", "knowledge_base", {})]
            ),
            GenerationResult(query="", answer="done", tool_calls=[]),
        ]
        tool = _fake_tool()
        tool.execute.side_effect = [
            ToolResult(
                tool_name="knowledge_base",
                content="ctx1",
                success=True,
                retrieved_chunks=[_make_rc("c1", 0.9), _make_rc("c2", 0.5)],
            ),
            ToolResult(
                tool_name="knowledge_base",
                content="ctx2",
                success=True,
                retrieved_chunks=[_make_rc("c1", 0.95), _make_rc("c3", 0.7)],
            ),
        ]

        result = _agent(gen, [tool], top_k_final=2).run("Q?")

        # dedup by id keeping max score (c1→0.95), sort desc, cap at 2.
        assert [rc.chunk.chunk_id for rc in result.retrieved_chunks] == ["c1", "c3"]
        assert result.retrieved_chunks[0].score == 0.95
        assert result.metadata["retrieved_chunk_count"] == 2

    def test_tool_failure_is_recoverable(self) -> None:
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c1", "knowledge_base", {})]
            ),
            GenerationResult(query="", answer="recovered", tool_calls=[]),
        ]
        tool = _fake_tool()
        tool.execute.return_value = ToolResult(
            tool_name="knowledge_base",
            content="search failed",
            success=False,
            retrieved_chunks=[],
        )

        result = _agent(gen, [tool]).run("Q?")  # must not raise

        assert result.answer == "recovered"
        assert result.retrieved_chunks == []

    def test_unknown_tool_is_recoverable(self) -> None:
        gen = MagicMock()
        gen.generate.side_effect = [
            GenerationResult(
                query="", answer="", tool_calls=[ToolCall("c1", "ghost_tool", {})]
            ),
            GenerationResult(query="", answer="done", tool_calls=[]),
        ]
        tool = _fake_tool()

        result = _agent(gen, [tool]).run("Q?")

        tool.execute.assert_not_called()
        assert result.answer == "done"
        # The observation fed back names the unknown tool.
        observation = gen.generate.call_args_list[1].args[0][-1]
        assert observation["role"] == "tool"
        assert "ghost_tool" in observation["content"]

    @patch("uwf_rag.pipeline.agent.build_generator")
    @patch("uwf_rag.pipeline.agent.tool_to_spec")
    @patch("uwf_rag.pipeline.agent.registry")
    def test_from_config_wires_generator_tools_and_budget(
        self,
        mock_registry: MagicMock,
        mock_tool_to_spec: MagicMock,
        mock_build_generator: MagicMock,
    ) -> None:
        from uwf_rag.core.config import ExperimentConfig

        tool_cls = MagicMock()
        mem_cls = MagicMock()
        mock_registry.get.side_effect = lambda category, name: {
            "tool": tool_cls,
            "memory": mem_cls,
        }[category]

        config = ExperimentConfig.from_dict(
            {
                "pipeline_mode": "agent",
                "agent": {
                    "mode": "single",
                    "max_iterations": 7,
                    "llm": {"provider": "ollama", "model_name": "qwen2.5"},
                    "tools": [{"name": "knowledge_base", "type": "rag"}],
                    "memory": {"type": "none"},
                },
                "query": {"retrieval": {"top_k_retrieve": 10, "top_k_final": 4}},
            }
        )

        pipeline = AgentPipeline.from_config(config, MagicMock())

        assert pipeline._max_iterations == 7  # type: ignore[attr-defined]
        assert pipeline._top_k_final == 4  # type: ignore[attr-defined]
        mock_build_generator.assert_called_once()  # reasoning generator built
        tool_cls.build.assert_called_once()  # tool built from its entry


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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.pipeline.indexing import IndexingPipeline

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
        from uwf_rag.components.base import BaseLexicalIndex
        from uwf_rag.core.config import QueryConfig
        from uwf_rag.pipeline.query import QueryPipeline

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
                "generator": {"provider": "ollama", "model_name": "x"},
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
        from uwf_rag.core.config import QueryConfig
        from uwf_rag.pipeline.query import QueryPipeline

        artifact = self._index_artifact(sparse_index=None)
        config = QueryConfig.from_dict(
            {
                "retrieval": {
                    "type": "bm25",
                    "top_k_retrieve": 5,
                    "top_k_final": 5,
                },
                "generator": {"provider": "ollama", "model_name": "x"},
                "prompt": {"type": "chat"},
            }
        )
        with pytest.raises(RuntimeError, match="sparse index"):
            QueryPipeline.from_config(config, artifact, retrieval_only=True)
