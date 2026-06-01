"""Tests for src/core/types.py — dataclass definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

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
    ToolCall,
    ToolResult,
    ToolSpec,
    TransformedQuery,
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

    def test_index_text_defaults_none_and_falls_back_to_content(self) -> None:
        c = Chunk(content="raw chunk", chunk_id="abc")
        assert c.index_text is None
        assert c.text_for_index == "raw chunk"

    def test_index_text_overrides_for_indexing_only(self) -> None:
        c = Chunk(content="raw chunk", chunk_id="abc")
        c.index_text = "context.\n\nraw chunk"
        assert c.text_for_index == "context.\n\nraw chunk"
        # content (stored / generated / scored) is untouched.
        assert c.content == "raw chunk"


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


class TestTransformedQuery:
    def test_defaults(self) -> None:
        tq = TransformedQuery(text="What is X?")
        assert tq.text == "What is X?"
        assert tq.branch is None

    def test_with_branch(self) -> None:
        tq = TransformedQuery(text="hypothetical doc", branch="dense")
        assert tq.branch == "dense"

    def test_frozen(self) -> None:
        tq = TransformedQuery(text="q")
        with pytest.raises(FrozenInstanceError):
            tq.text = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TransformedQuery(text="q", branch=None)
        b = TransformedQuery(text="q", branch=None)
        assert a == b


class TestGenerationResult:
    def test_defaults(self) -> None:
        # Two-arg positional construction must keep working after adding the
        # tool-calling fields — the linear pipeline relies on it.
        r = GenerationResult(query="q", answer="a")
        assert r.retrieved_chunks == []
        assert r.metadata == {}
        assert r.tool_calls == []
        assert r.finish_reason == ""

    def test_tool_calling_fields(self) -> None:
        r = GenerationResult(
            query="q",
            answer="",
            tool_calls=[ToolCall(id="c1", name="kb", arguments={"query": "x"})],
            finish_reason="tool_calls",
        )
        assert r.finish_reason == "tool_calls"
        assert r.tool_calls[0].name == "kb"

    def test_mutable_defaults_independent(self) -> None:
        a = GenerationResult(query="q", answer="a")
        b = GenerationResult(query="q", answer="a")
        a.tool_calls.append(ToolCall(id="c", name="kb"))
        assert b.tool_calls == []


class TestToolSpec:
    def test_fields(self) -> None:
        spec = ToolSpec(
            name="knowledge_base",
            description="Search the KB.",
            parameters={"type": "object", "properties": {}},
        )
        assert spec.name == "knowledge_base"
        assert spec.parameters["type"] == "object"

    def test_parameters_default_empty(self) -> None:
        assert ToolSpec(name="t", description="d").parameters == {}


class TestToolCall:
    def test_fields(self) -> None:
        tc = ToolCall(id="c1", name="kb", arguments={"query": "x"})
        assert tc.id == "c1"
        assert tc.name == "kb"
        assert tc.arguments == {"query": "x"}

    def test_arguments_default_empty(self) -> None:
        assert ToolCall(id="c1", name="kb").arguments == {}


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
        assert s.metadata == {}


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
        # Metadata defaults to an empty dict and is always present.
        assert d["metadata"] == {}

    def test_to_result_dict_persists_metadata(self) -> None:
        # Agent provenance (iterations, tool calls, step trace) must reach the
        # saved per-sample output so runs are auditable.
        s = ScoredSample(
            id="1",
            query="q",
            response="r",
            retrieved_contexts=["c"],
            reference="ref",
            scores={"f": 0.9},
            metadata={"mode": "agent", "iterations": 2, "num_tool_calls": 1},
        )
        d = s.to_result_dict()
        assert d["metadata"]["mode"] == "agent"
        assert d["metadata"]["iterations"] == 2
        assert d["metadata"]["num_tool_calls"] == 1

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
