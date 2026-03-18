"""Tests for src/pipeline/ — query, indexing, rag, and agent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.types import (
    Chunk,
    ExperimentResult,
    GenerationResult,
    IndexArtifact,
    RetrievedChunk,
    ScoredSample,
)
from pipeline.agent import AgentPipeline
from pipeline.query import QueryPipeline
from pipeline.rag import RAGPipeline, _get_git_dirty, _get_git_sha, _save_results

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_rc(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content=f"text_{chunk_id}", chunk_id=chunk_id, metadata={}),
        score=score,
    )


def _make_query_pipeline(
    *, retrieval_only: bool = False
) -> tuple[QueryPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build a QueryPipeline with all mocked components."""
    qt = MagicMock()
    qt.transform.return_value = ["transformed query"]

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_rc("c1", 0.9), _make_rc("c2", 0.8)]

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
        retriever.retrieve.assert_called_once()
        reranker.rerank.assert_called_once()
        gen.generate.assert_called_once()

    def test_retrieval_only_skips_generation(self) -> None:
        pipeline, _, _, _, _gen = _make_query_pipeline(retrieval_only=True)
        result = pipeline.run("Q?")
        assert result.answer == ""
        assert result.metadata.get("retrieval_only") is True

    def test_deduplication(self) -> None:
        pipeline, qt, retriever, _, _ = _make_query_pipeline()
        # Two queries returning overlapping chunks
        qt.transform.return_value = ["q1", "q2"]
        retriever.retrieve.side_effect = [
            [_make_rc("c1", 0.9), _make_rc("c2", 0.8)],
            [_make_rc("c1", 0.7), _make_rc("c3", 0.6)],
        ]
        result = pipeline.run("Q?")
        chunk_ids = [rc.chunk.chunk_id for rc in result.retrieved_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_dedup_keeps_best_score(self) -> None:
        pipeline, qt, retriever, reranker, _ = _make_query_pipeline()
        qt.transform.return_value = ["q1", "q2"]
        retriever.retrieve.side_effect = [
            [_make_rc("c1", 0.5)],
            [_make_rc("c1", 0.9)],
        ]
        reranker.rerank.side_effect = lambda q, chunks, k: chunks[:k]
        result = pipeline.run("Q?")
        c1_chunks = [rc for rc in result.retrieved_chunks if rc.chunk.chunk_id == "c1"]
        assert c1_chunks[0].score == 0.9


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
# Git helpers (subprocess mocked)
# -----------------------------------------------------------------------


class TestGitHelpers:
    @patch("pipeline.rag.subprocess.check_output")
    def test_get_git_sha(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = "abc1234\n"
        assert _get_git_sha() == "abc1234"

    @patch("pipeline.rag.subprocess.check_output", side_effect=FileNotFoundError)
    def test_get_git_sha_fallback(self, mock_sub: MagicMock) -> None:
        assert _get_git_sha() == "unknown"

    @patch("pipeline.rag.subprocess.check_output")
    def test_get_git_dirty_clean(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = b""
        assert _get_git_dirty() is False

    @patch("pipeline.rag.subprocess.check_output")
    def test_get_git_dirty_dirty(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = b" M src/foo.py\n"
        assert _get_git_dirty() is True

    @patch(
        "pipeline.rag.subprocess.check_output",
        side_effect=FileNotFoundError,
    )
    def test_get_git_dirty_no_git(self, mock_sub: MagicMock) -> None:
        assert _get_git_dirty() is False


# -----------------------------------------------------------------------
# _save_results
# -----------------------------------------------------------------------


class TestSaveResults:
    def _make_config(self, minimal_config_dict: dict | None = None):
        """Build a real ExperimentConfig for _save_results."""
        from core.config import ExperimentConfig

        if minimal_config_dict:
            return ExperimentConfig.from_dict(minimal_config_dict)
        # Inline minimal config
        return ExperimentConfig.from_dict(
            {
                "name": "test_exp",
                "pipeline_mode": "linear",
                "indexing": {
                    "sources": [{"name": "s", "path": "/f", "ingest": {"type": "pdf"}}],
                    "chunking": {"type": "recursive"},
                    "embedding": {"type": "huggingface"},
                    "vectorstore": {"type": "faiss"},
                },
                "query": {
                    "retrieval": {"type": "dense"},
                    "generation": {"type": "ollama"},
                    "generation_llm": {"model_name": "m"},
                    "prompt": {"type": "chat"},
                },
                "evaluation": {"dataset": "d.jsonl"},
            }
        )

    def test_saves_summary_json(self, tmp_path: Path) -> None:
        config = self._make_config()
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"acc": 0.9},
            config_snapshot={"key": "value"},
        )
        with patch("pipeline.rag._get_git_sha", return_value="abc"):
            with patch("pipeline.rag._get_git_dirty", return_value=False):
                _save_results(config, result, tmp_path)

        summary_path = tmp_path / "test_exp" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["experiment_name"] == "test_exp"
        assert summary["metrics"]["acc"] == 0.9
        assert summary["git_sha"] == "abc"
        assert summary["git_dirty"] is False
        assert "timestamp" in summary

    def test_saves_config_yaml(self, tmp_path: Path) -> None:
        config = self._make_config()
        result = ExperimentResult(experiment_name="test_exp", metrics={})
        with patch("pipeline.rag._get_git_sha", return_value="x"):
            with patch("pipeline.rag._get_git_dirty", return_value=False):
                _save_results(config, result, tmp_path)

        config_path = tmp_path / "test_exp" / "config.yaml"
        assert config_path.exists()

    def test_saves_run_jsonl(self, tmp_path: Path) -> None:
        config = self._make_config()
        sample = ScoredSample(
            id="1",
            query="Q",
            response="A",
            retrieved_contexts=["ctx"],
            reference="R",
            scores={"f": 0.9},
        )
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"f": 0.9},
            per_run_samples=[[sample]],
        )
        with patch("pipeline.rag._get_git_sha", return_value="x"):
            with patch("pipeline.rag._get_git_dirty", return_value=False):
                _save_results(config, result, tmp_path)

        run_path = tmp_path / "test_exp" / "run_1.jsonl"
        assert run_path.exists()
        line = json.loads(run_path.read_text().strip())
        assert line["id"] == "1"
        assert line["scores"]["f"] == 0.9
