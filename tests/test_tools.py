"""Tests for src/components/tools.py — agent tools (RAG search)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ragbench.components.build import BuildContext
from ragbench.components.tools import RAGSearchTool, tool_to_spec
from ragbench.core.config import ExperimentConfig, ToolConfig
from ragbench.core.registry import registry
from ragbench.core.types import Chunk, GenerationResult, RetrievedChunk, ToolResult


def _rc(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content=text, chunk_id=chunk_id, metadata={}), score=score
    )


class TestToolToSpec:
    def test_uses_tool_name_and_description(self) -> None:
        tool = RAGSearchTool(pipeline=None, name="kb", description="Search the KB.")
        spec = tool_to_spec(tool)
        assert spec.name == "kb"
        assert spec.description == "Search the KB."
        assert spec.parameters["required"] == ["query"]
        assert spec.parameters["properties"]["query"]["type"] == "string"

    def test_description_override(self) -> None:
        tool = RAGSearchTool(pipeline=None, name="kb", description="default")
        assert tool_to_spec(tool, description="override").description == "override"


class TestRAGSearchTool:
    def test_execute_returns_chunks_as_observation(self) -> None:
        pipeline = MagicMock()
        chunks = [_rc("c1", "grade forgiveness info"), _rc("c2", "more info")]
        pipeline.run.return_value = GenerationResult(
            query="q", answer="", retrieved_chunks=chunks
        )
        tool = RAGSearchTool(pipeline=pipeline, name="knowledge_base")
        result = tool.execute("grade forgiveness")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "knowledge_base"
        assert result.retrieved_chunks == chunks
        assert "grade forgiveness info" in result.content
        assert "[1]" in result.content and "[2]" in result.content
        assert result.metadata["num_chunks"] == 2
        pipeline.run.assert_called_once_with("grade forgiveness")

    def test_execute_empty_results(self) -> None:
        pipeline = MagicMock()
        pipeline.run.return_value = GenerationResult(
            query="q", answer="", retrieved_chunks=[]
        )
        result = RAGSearchTool(pipeline=pipeline).execute("q")
        assert result.success is True
        assert result.retrieved_chunks == []
        assert "No relevant passages" in result.content

    def test_execute_failure_is_recoverable(self) -> None:
        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("vectorstore down")
        result = RAGSearchTool(pipeline=pipeline, name="kb").execute("q")
        assert result.success is False
        assert result.retrieved_chunks == []
        assert "failed" in result.content.lower()

    @patch("ragbench.pipeline.query.QueryPipeline.from_config")
    def test_build_forces_passthrough_transform(
        self, mock_from_config: MagicMock
    ) -> None:
        mock_from_config.return_value = MagicMock()
        experiment_config = ExperimentConfig.from_dict(
            {
                "query": {
                    "query_transform": {"type": "contextualizer"},
                    "retrieval": {
                        "type": "dense",
                        "top_k_retrieve": 10,
                        "top_k_final": 5,
                    },
                    "reranking": {"type": "cross_encoder"},
                }
            }
        )
        entry = ToolConfig(name="knowledge_base", type="rag")
        ctx = BuildContext(
            registry=registry, index=MagicMock(), query=experiment_config.query
        )
        tool = RAGSearchTool.build(entry, ctx)

        assert tool.name == "knowledge_base"
        passed_query_config = mock_from_config.call_args[0][0]
        # The agent already reformulates, so the tool retrieves with passthrough...
        assert passed_query_config.query_transform.type == "passthrough"
        # ...but reuses the experiment's retrieval + rerank stack, in
        # retrieval-only mode.
        assert passed_query_config.retrieval.type == "dense"
        assert passed_query_config.reranking.type == "cross_encoder"
        assert mock_from_config.call_args.kwargs.get("retrieval_only") is True

    @patch(
        "ragbench.pipeline.query.QueryPipeline.from_config", return_value=MagicMock()
    )
    def test_build_applies_param_retrieval_scope(
        self, mock_from_config: MagicMock
    ) -> None:
        """cfg.params filters/top_k scope the tool's retrieval (multitool rung)."""
        experiment_config = ExperimentConfig.from_dict(
            {
                "query": {
                    "retrieval": {
                        "type": "dense",
                        "top_k_retrieve": 10,
                        "top_k_final": 5,
                        "filters": {"lang": "en"},
                    }
                }
            }
        )
        entry = ToolConfig(
            name="handbook",
            type="rag",
            params={
                "filters": {"source_name": "student_handbook"},
                "top_k_final": 3,
            },
        )
        ctx = BuildContext(
            registry=registry, index=MagicMock(), query=experiment_config.query
        )
        RAGSearchTool.build(entry, ctx)

        passed = mock_from_config.call_args[0][0]
        # Per-tool filters merge over the stack's defaults; top_k overrides.
        assert passed.retrieval.filters == {
            "lang": "en",
            "source_name": "student_handbook",
        }
        assert passed.retrieval.top_k_final == 3
        assert passed.retrieval.top_k_retrieve == 10  # untouched

    @patch(
        "ragbench.pipeline.query.QueryPipeline.from_config", return_value=MagicMock()
    )
    def test_build_default_name_and_description(
        self, _mock_from_config: MagicMock
    ) -> None:
        entry = ToolConfig(type="rag")  # no name / description
        ctx = BuildContext(
            registry=registry,
            index=MagicMock(),
            query=ExperimentConfig.from_dict({}).query,
        )
        tool = RAGSearchTool.build(entry, ctx)
        assert tool.name == "knowledge_base"
        assert "knowledge base" in tool.description.lower()
