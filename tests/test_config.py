"""Tests for src/core/config.py — YAML config system with inheritance."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    AgentConfig,
    ComponentConfig,
    ConfigValidationError,
    EvaluationConfig,
    ExperimentConfig,
    IndexingConfig,
    LLMConfig,
    PromptConfig,
    RetrievalConfig,
    _deep_merge,
    load_yaml_with_inheritance,
    validate_config,
)
from core.registry import registry

# -----------------------------------------------------------------------
# _deep_merge
# -----------------------------------------------------------------------


class TestDeepMerge:
    def test_flat_override(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_dicts_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3}}
        assert _deep_merge(base, override) == {"x": {"a": 1, "b": 3}}

    def test_list_replacement(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        assert _deep_merge(base, override) == {"items": [4, 5]}

    def test_new_key_added(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_empty_override(self) -> None:
        base = {"a": 1, "b": {"c": 3}}
        assert _deep_merge(base, {}) == base


# -----------------------------------------------------------------------
# load_yaml_with_inheritance
# -----------------------------------------------------------------------


class TestLoadYamlWithInheritance:
    def test_no_extends(self, fixtures_dir: Path) -> None:
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        assert raw["name"] == "baseline"
        assert "indexing" in raw

    def test_with_extends(self, fixtures_dir: Path) -> None:
        raw = load_yaml_with_inheritance(fixtures_dir / "child_experiment.yaml")
        # Child overrides
        assert raw["name"] == "chunking_1000"
        # Inherited from parent
        assert "indexing" in raw

    def test_extends_overrides_applied(self, fixtures_dir: Path) -> None:
        raw = load_yaml_with_inheritance(fixtures_dir / "child_experiment.yaml")
        assert raw["indexing"]["chunking"]["params"]["chunk_size"] == 1000

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_with_inheritance(tmp_path / "nonexistent.yaml")


# -----------------------------------------------------------------------
# ComponentConfig
# -----------------------------------------------------------------------


class TestComponentConfig:
    def test_from_dict(self) -> None:
        cfg = ComponentConfig.from_dict({"type": "recursive", "params": {"k": 5}})
        assert cfg.type == "recursive"
        assert cfg.params == {"k": 5}

    def test_from_none(self) -> None:
        cfg = ComponentConfig.from_dict(None)
        assert cfg.type == ""
        assert cfg.params == {}

    def test_from_empty(self) -> None:
        cfg = ComponentConfig.from_dict({})
        assert cfg.type == ""


# -----------------------------------------------------------------------
# ExperimentConfig
# -----------------------------------------------------------------------


class TestExperimentConfig:
    def test_from_dict_full(self, minimal_config_dict: dict) -> None:
        cfg = ExperimentConfig.from_dict(minimal_config_dict)
        assert cfg.name == "test_baseline"
        assert cfg.pipeline_mode == "linear"
        assert cfg.indexing.chunking.type == "recursive"
        assert cfg.query.retrieval.top_k_final == 3

    def test_defaults(self) -> None:
        cfg = ExperimentConfig.from_dict({})
        assert cfg.name == ""
        assert cfg.pipeline_mode == "linear"
        assert cfg.evaluation.num_runs == 3
        assert len(cfg.evaluation.metrics) == 5

    def test_from_yaml(self, fixtures_dir: Path) -> None:
        cfg = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        assert cfg.name == "baseline"
        assert cfg.indexing.chunking.type == "recursive"


# -----------------------------------------------------------------------
# Specific config types
# -----------------------------------------------------------------------


class TestSpecificConfigs:
    def test_llm_config(self) -> None:
        cfg = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "model_name": "qwen2.5:14b",
                "temperature": 0.1,
                "max_tokens": 512,
            }
        )
        assert cfg.provider == "ollama"
        assert cfg.model_name == "qwen2.5:14b"
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 512

    def test_retrieval_config_defaults(self) -> None:
        cfg = RetrievalConfig.from_dict(None)
        assert cfg.type == "dense"
        assert cfg.top_k_retrieve == 10
        assert cfg.top_k_final == 5

    def test_prompt_config(self) -> None:
        cfg = PromptConfig.from_dict(
            {
                "type": "chat",
                "use_chain_of_thought": True,
                "citation_style": "inline",
            }
        )
        assert cfg.use_chain_of_thought is True
        assert cfg.citation_style == "inline"

    def test_evaluation_config_defaults(self) -> None:
        cfg = EvaluationConfig.from_dict(None)
        assert "faithfulness" in cfg.metrics
        assert cfg.num_runs == 3

    def test_indexing_config_sources(self) -> None:
        cfg = IndexingConfig.from_dict(
            {
                "sources": [
                    {"name": "doc1", "path": "/a.pdf", "ingest": {"type": "pdf"}}
                ],
            }
        )
        assert len(cfg.sources) == 1
        assert cfg.sources[0].name == "doc1"

    def test_agent_config_nested(self) -> None:
        cfg = AgentConfig.from_dict(
            {
                "mode": "multi",
                "agents": [{"name": "rag_agent", "type": "rag"}],
                "tools": [{"name": "search", "type": "rag", "tool": "search"}],
            }
        )
        assert cfg.mode == "multi"
        assert len(cfg.agents) == 1
        assert len(cfg.tools) == 1


# -----------------------------------------------------------------------
# index_fingerprint
# -----------------------------------------------------------------------


class TestIndexFingerprint:
    def test_deterministic(self, fixtures_dir: Path) -> None:
        cfg = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        fp1 = cfg.index_fingerprint()
        fp2 = cfg.index_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 12

    def test_changes_with_config(self, fixtures_dir: Path) -> None:
        cfg1 = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw["indexing"]["chunking"]["params"]["chunk_size"] = 9999
        cfg2 = ExperimentConfig.from_dict(raw)
        assert cfg1.index_fingerprint() != cfg2.index_fingerprint()


# -----------------------------------------------------------------------
# validate_config
# -----------------------------------------------------------------------


class TestValidateConfig:
    """Tests for validate_config() against the real component registry."""

    def _make_valid_config(self, **overrides: object) -> ExperimentConfig:
        """Build a minimal valid linear config with optional overrides."""
        data: dict = {
            "name": "test",
            "pipeline_mode": "linear",
            "indexing": {
                "sources": [{"name": "s", "path": "x.pdf", "ingest": {"type": "pdf"}}],
                "chunking": {"type": "recursive"},
                "embedding": {"type": "huggingface"},
                "vectorstore": {"type": "faiss"},
            },
            "query": {
                "retrieval": {"type": "dense", "top_k_retrieve": 10, "top_k_final": 5},
                "generation": {"type": "ollama"},
                "prompt": {"type": "chat"},
            },
            "evaluation": {"dataset": "", "mode": "full"},
        }
        for key, value in overrides.items():
            parts = key.split(".")
            d = data
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
        return ExperimentConfig.from_dict(data)

    def test_valid_config_passes(self) -> None:
        cfg = self._make_valid_config()
        validate_config(cfg, registry)  # should not raise

    def test_unregistered_component_type(self) -> None:
        cfg = self._make_valid_config()
        cfg.indexing.chunking.type = "bogus_chunker"
        with pytest.raises(ConfigValidationError, match="bogus_chunker"):
            validate_config(cfg, registry)

    def test_error_lists_available_types(self) -> None:
        cfg = self._make_valid_config()
        cfg.indexing.chunking.type = "bogus"
        with pytest.raises(ConfigValidationError, match="recursive"):
            validate_config(cfg, registry)

    def test_top_k_constraint(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.retrieval.top_k_final = 20
        cfg.query.retrieval.top_k_retrieve = 5
        with pytest.raises(ConfigValidationError, match="top_k_final"):
            validate_config(cfg, registry)

    def test_missing_dataset(self, tmp_path: Path) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.dataset = str(tmp_path / "nonexistent.jsonl")
        with pytest.raises(ConfigValidationError, match="not found"):
            validate_config(cfg, registry)

    def test_existing_dataset_passes(self, tmp_path: Path) -> None:
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "Q", "reference": "R"}\n')
        cfg = self._make_valid_config()
        cfg.evaluation.dataset = str(ds)
        validate_config(cfg, registry)  # should not raise

    def test_multiple_errors_collected(self) -> None:
        cfg = self._make_valid_config()
        cfg.indexing.chunking.type = "bad1"
        cfg.indexing.embedding.type = "bad2"
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg, registry)
        assert len(exc_info.value.errors) >= 2
        assert any("bad1" in e for e in exc_info.value.errors)
        assert any("bad2" in e for e in exc_info.value.errors)

    def test_agent_mode_no_agents_or_tools(self) -> None:
        cfg = self._make_valid_config()
        cfg.pipeline_mode = "agent"
        cfg.agent.agents = []
        cfg.agent.tools = []
        with pytest.raises(ConfigValidationError, match="no agents or tools"):
            validate_config(cfg, registry)

    def test_empty_query_transform_defaults_to_passthrough(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.query_transform.type = ""
        validate_config(cfg, registry)  # should not raise

    def test_empty_reranking_defaults_to_none(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.reranking.type = ""
        validate_config(cfg, registry)  # should not raise

    def test_retrieval_only_skips_generation_check(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.generation.type = ""
        cfg.evaluation.mode = "retrieval_only"
        validate_config(cfg, registry)  # should not raise

    def test_full_mode_requires_generation(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.generation.type = ""
        cfg.evaluation.mode = "full"
        with pytest.raises(ConfigValidationError, match="generator is required"):
            validate_config(cfg, registry)

    def test_unknown_pipeline_mode(self) -> None:
        cfg = self._make_valid_config()
        cfg.pipeline_mode = "unknown"
        with pytest.raises(ConfigValidationError, match="not recognized"):
            validate_config(cfg, registry)
