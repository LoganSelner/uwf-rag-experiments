"""Tests for src/core/types.py — dataclass definitions."""

from __future__ import annotations

from core.types import (
    AgentStep,
    Chunk,
    Document,
    EmbeddedChunk,
    EvalSample,
    ExperimentResult,
    GenerationResult,
    RetrievedChunk,
    ScoredSample,
    ToolResult,
)


class TestDocument:
    def test_defaults(self) -> None:
        doc = Document(content="hello")
        assert doc.content == "hello"
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        doc = Document(content="x", metadata={"page": 1})
        assert doc.metadata["page"] == 1


class TestChunk:
    def test_fields(self) -> None:
        c = Chunk(content="text", chunk_id="abc", metadata={"k": "v"})
        assert c.content == "text"
        assert c.chunk_id == "abc"
        assert c.metadata == {"k": "v"}


class TestEmbeddedChunk:
    def test_carries_chunk(self) -> None:
        c = Chunk(content="x", chunk_id="1", metadata={})
        ec = EmbeddedChunk(chunk=c, embedding=[0.1, 0.2])
        assert ec.chunk.chunk_id == "1"
        assert len(ec.embedding) == 2


class TestRetrievedChunk:
    def test_defaults(self) -> None:
        c = Chunk(content="x", chunk_id="1", metadata={})
        rc = RetrievedChunk(chunk=c, score=0.9)
        assert rc.retrieval_method == ""


class TestGenerationResult:
    def test_defaults(self) -> None:
        r = GenerationResult(query="q", answer="a")
        assert r.retrieved_chunks == []
        assert r.metadata == {}


class TestToolResult:
    def test_fields(self) -> None:
        r = ToolResult(tool_name="search", content="found", success=True)
        assert r.tool_name == "search"
        assert r.success is True


class TestAgentStep:
    def test_defaults(self) -> None:
        s = AgentStep(step_number=1, action="think")
        assert s.agent_name == ""
        assert s.tool_name == ""


class TestEvalSample:
    def test_fields(self) -> None:
        s = EvalSample(
            id="1",
            query="q",
            response="r",
            retrieved_contexts=["c"],
            reference="ref",
        )
        assert s.id == "1"
        assert s.reference == "ref"


class TestScoredSample:
    def test_to_result_dict(self) -> None:
        s = ScoredSample(
            id="1",
            query="q",
            response="r",
            retrieved_contexts=["c1", "c2"],
            reference="ref",
            scores={"faithfulness": 0.9, "accuracy": 0.8},
        )
        d = s.to_result_dict()
        assert d["id"] == "1"
        assert d["input"]["query"] == "q"
        assert d["input"]["reference"] == "ref"
        assert d["output"]["response"] == "r"
        assert d["output"]["retrieved_contexts"] == ["c1", "c2"]
        assert d["scores"]["faithfulness"] == 0.9

    def test_to_result_dict_with_none_scores(self) -> None:
        s = ScoredSample(
            id="1",
            query="q",
            response="r",
            retrieved_contexts=[],
            reference="ref",
            scores={"metric": None},
        )
        d = s.to_result_dict()
        assert d["scores"]["metric"] is None


class TestExperimentResult:
    def test_defaults(self) -> None:
        r = ExperimentResult(
            experiment_name="test",
            metrics={"acc": 0.9},
        )
        assert r.per_run_metrics == []
        assert r.per_run_samples == []
        assert r.num_runs == 1
        assert r.config_snapshot == {}

    def test_mutable_defaults_independent(self) -> None:
        r1 = ExperimentResult(experiment_name="a", metrics={})
        r2 = ExperimentResult(experiment_name="b", metrics={})
        r1.per_run_metrics.append({"x": 1.0})
        assert r2.per_run_metrics == []
