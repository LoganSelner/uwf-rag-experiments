"""Tests for the foundation layer: types, registry, config system."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Ensure src/ is on the path so bare imports like `from core.types import ...` work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.config import (
    ExperimentConfig,
    _deep_merge,
    load_yaml_with_inheritance,
)
from core.registry import ComponentRegistry, registry
from core.types import (
    AgentStep,
    Chunk,
    Document,
    EmbeddedChunk,
    EvalSample,
    ExperimentResult,
    GenerationResult,
    IndexArtifact,
    RetrievedChunk,
    ScoredSample,
    ToolResult,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------


class TestTypes:
    def test_document_defaults(self) -> None:
        doc = Document(content="hello")
        assert doc.content == "hello"
        assert doc.metadata == {}

    def test_chunk_carries_metadata(self) -> None:
        chunk = Chunk(
            content="text",
            chunk_id="c1",
            metadata={"source_name": "handbook"},
        )
        assert chunk.metadata["source_name"] == "handbook"

    def test_embedded_chunk(self) -> None:
        chunk = Chunk(content="text", chunk_id="c1")
        ec = EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3])
        assert len(ec.embedding) == 3
        assert ec.chunk is chunk

    def test_retrieved_chunk(self) -> None:
        chunk = Chunk(content="text", chunk_id="c1")
        rc = RetrievedChunk(chunk=chunk, score=0.95, retrieval_method="dense")
        assert rc.score == 0.95

    def test_generation_result(self) -> None:
        result = GenerationResult(query="q", answer="a")
        assert result.retrieved_chunks == []
        assert result.metadata == {}

    def test_tool_result(self) -> None:
        tr = ToolResult(tool_name="web", content="result", success=True)
        assert tr.success is True

    def test_agent_step(self) -> None:
        step = AgentStep(step_number=1, action="call_tool", tool_name="web")
        assert step.step_number == 1

    def test_eval_sample(self) -> None:
        sample = EvalSample(
            id="1",
            query="q",
            response="a",
            retrieved_contexts=["ctx1"],
            reference="ref",
        )
        assert sample.id == "1"
        assert len(sample.retrieved_contexts) == 1

    def test_scored_sample(self) -> None:
        sample = ScoredSample(
            id="1",
            query="q",
            response="a",
            retrieved_contexts=["ctx1"],
            reference="ref",
            scores={"faithfulness": 0.9, "answer_correctness": None},
        )
        assert sample.scores["faithfulness"] == 0.9
        assert sample.scores["answer_correctness"] is None

    def test_scored_sample_to_result_dict(self) -> None:
        sample = ScoredSample(
            id="42",
            query="What is X?",
            response="X is Y.",
            retrieved_contexts=["ctx"],
            reference="ref",
            scores={"faithfulness": 0.9},
        )
        d = sample.to_result_dict()
        assert d["id"] == "42"
        assert d["input"]["query"] == "What is X?"
        assert d["input"]["reference"] == "ref"
        assert d["output"]["response"] == "X is Y."
        assert d["output"]["retrieved_contexts"] == ["ctx"]
        assert d["scores"]["faithfulness"] == 0.9

    def test_experiment_result(self) -> None:
        result = ExperimentResult(
            experiment_name="test",
            metrics={"answer_correctness": 0.85},
        )
        assert result.num_runs == 1
        assert result.per_run_samples == []

    def test_index_artifact(self) -> None:
        artifact = IndexArtifact(vectorstore=None, embedder=None)
        assert artifact.stats == {}


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------


class TestRegistry:
    def setup_method(self) -> None:
        """Use a fresh registry for each test."""
        self.reg = ComponentRegistry()

    def test_register_and_get(self) -> None:
        @self.reg.register("chunking", "test_chunker")
        class TestChunker:
            pass

        assert self.reg.get("chunking", "test_chunker") is TestChunker

    def test_duplicate_raises(self) -> None:
        @self.reg.register("chunking", "dup")
        class First:
            pass

        with pytest.raises(ValueError, match="Duplicate"):

            @self.reg.register("chunking", "dup")
            class Second:
                pass

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="No component"):
            self.reg.get("chunking", "nonexistent")

    def test_list_category(self) -> None:
        @self.reg.register("embedding", "a")
        class A:
            pass

        @self.reg.register("embedding", "b")
        class B:
            pass

        @self.reg.register("chunking", "c")
        class C:
            pass

        names = self.reg.list_category("embedding")
        assert sorted(names) == ["a", "b"]

    def test_is_registered(self) -> None:
        @self.reg.register("chunking", "x")
        class X:
            pass

        assert self.reg.is_registered("chunking", "x")
        assert not self.reg.is_registered("chunking", "y")

    def test_clear(self) -> None:
        @self.reg.register("chunking", "z")
        class Z:
            pass

        self.reg.clear()
        assert not self.reg.is_registered("chunking", "z")

    def test_global_registry_has_defaults(self) -> None:
        """The singleton registry should have defaults from defaults.py."""
        # Import components to trigger registration
        import components  # noqa: F401

        assert registry.is_registered("query_transform", "passthrough")
        assert registry.is_registered("reranking", "none")
        assert registry.is_registered("memory", "none")
        assert registry.is_registered("memory", "buffer_window")


# -----------------------------------------------------------------------
# Config: deep merge
# -----------------------------------------------------------------------


class TestDeepMerge:
    def test_scalar_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_dict_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3}}
        assert _deep_merge(base, override) == {"x": {"a": 1, "b": 3}}

    def test_list_replacement(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        assert _deep_merge(base, override) == {"items": [4, 5]}

    def test_new_keys_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_deeply_nested(self) -> None:
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 3}}}}
        result = _deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 3}}}}

    def test_base_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}


# -----------------------------------------------------------------------
# Config: YAML loading with inheritance
# -----------------------------------------------------------------------


class TestYAMLInheritance:
    def test_load_base(self) -> None:
        data = load_yaml_with_inheritance(FIXTURES / "base.yaml")
        assert data["name"] == "baseline"
        assert data["indexing"]["chunking"]["params"]["chunk_size"] == 512

    def test_load_child_inherits(self) -> None:
        data = load_yaml_with_inheritance(FIXTURES / "child_experiment.yaml")
        # Overridden values
        assert data["name"] == "chunking_1000"
        assert data["indexing"]["chunking"]["params"]["chunk_size"] == 1000
        assert data["indexing"]["chunking"]["params"]["chunk_overlap"] == 100
        # Inherited values
        assert data["indexing"]["chunking"]["type"] == "recursive"
        assert data["indexing"]["embedding"]["type"] == "huggingface"
        assert data["pipeline_mode"] == "linear"
        # Partially overridden nested dict
        assert data["query"]["retrieval"]["top_k_final"] == 3
        assert data["query"]["retrieval"]["top_k_retrieve"] == 10  # inherited


# -----------------------------------------------------------------------
# Config: ExperimentConfig parsing
# -----------------------------------------------------------------------


class TestExperimentConfig:
    def test_from_yaml(self) -> None:
        config = ExperimentConfig.from_yaml(FIXTURES / "base.yaml")
        assert config.name == "baseline"
        assert config.pipeline_mode == "linear"
        assert config.indexing.chunking.type == "recursive"
        assert config.indexing.chunking.params["chunk_size"] == 512
        assert config.indexing.embedding.params["model_name"] == "BAAI/bge-m3"
        assert config.query.retrieval.top_k_retrieve == 10
        assert config.query.retrieval.top_k_final == 5
        assert config.query.generation_llm.provider == "ollama"
        assert config.query.prompt.system_template == "You are an academic advisor."
        assert config.evaluation.num_runs == 3

    def test_child_inherits_and_overrides(self) -> None:
        config = ExperimentConfig.from_yaml(FIXTURES / "child_experiment.yaml")
        assert config.name == "chunking_1000"
        assert config.indexing.chunking.params["chunk_size"] == 1000
        # Inherited
        assert config.indexing.chunking.type == "recursive"
        assert config.indexing.embedding.type == "huggingface"

    def test_empty_dict_produces_defaults(self) -> None:
        config = ExperimentConfig.from_dict({})
        assert config.pipeline_mode == "linear"
        assert config.query.retrieval.top_k_retrieve == 10
        assert config.evaluation.num_runs == 3


# -----------------------------------------------------------------------
# Config: Index fingerprinting
# -----------------------------------------------------------------------


class TestIndexFingerprint:
    def test_same_indexing_same_fingerprint(self) -> None:
        c1 = ExperimentConfig.from_yaml(FIXTURES / "base.yaml")
        c2 = ExperimentConfig.from_yaml(FIXTURES / "base.yaml")
        assert c1.index_fingerprint() == c2.index_fingerprint()

    def test_different_query_same_fingerprint(self) -> None:
        """Query-side changes should NOT affect the index fingerprint."""
        indexing = {
            "sources": [{"name": "s", "path": "p", "ingest": {"type": "pdf"}}],
            "chunking": {"type": "recursive", "params": {"chunk_size": 512}},
            "embedding": {"type": "hf", "params": {"model_name": "bge"}},
            "vectorstore": {"type": "faiss"},
        }
        c1 = ExperimentConfig.from_dict(
            {
                "indexing": indexing,
                "query": {"retrieval": {"top_k_final": 5}},
            }
        )
        c2 = ExperimentConfig.from_dict(
            {
                "indexing": indexing,
                "query": {"retrieval": {"top_k_final": 3}},
            }
        )
        assert c1.index_fingerprint() == c2.index_fingerprint()

    def test_fingerprint_format(self) -> None:
        """Fingerprint should be 12 hex chars."""
        c1 = ExperimentConfig.from_yaml(FIXTURES / "base.yaml")
        fp = c1.index_fingerprint()
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_chunking_different_fingerprint(self) -> None:
        c1 = ExperimentConfig.from_yaml(FIXTURES / "base.yaml")
        c2 = ExperimentConfig.from_yaml(FIXTURES / "child_experiment.yaml")
        # child overrides chunk_size from 512 to 1000
        assert c1.index_fingerprint() != c2.index_fingerprint()

    def test_pipeline_mode_excluded(self) -> None:
        """pipeline_mode should NOT affect fingerprint."""
        c1 = ExperimentConfig.from_dict(
            {
                "pipeline_mode": "linear",
                "indexing": {
                    "sources": [{"name": "s", "path": "p", "ingest": {"type": "pdf"}}],
                    "chunking": {"type": "recursive", "params": {"chunk_size": 512}},
                    "embedding": {"type": "hf", "params": {"model_name": "bge"}},
                    "vectorstore": {"type": "faiss"},
                },
            }
        )
        c2 = ExperimentConfig.from_dict(
            {
                "pipeline_mode": "agent",
                "indexing": {
                    "sources": [{"name": "s", "path": "p", "ingest": {"type": "pdf"}}],
                    "chunking": {"type": "recursive", "params": {"chunk_size": 512}},
                    "embedding": {"type": "hf", "params": {"model_name": "bge"}},
                    "vectorstore": {"type": "faiss"},
                },
            }
        )
        assert c1.index_fingerprint() == c2.index_fingerprint()
