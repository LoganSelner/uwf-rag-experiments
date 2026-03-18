"""Tests for src/core/config.py — YAML config system with inheritance."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    AgentConfig,
    ComponentConfig,
    EvaluationConfig,
    ExperimentConfig,
    IndexingConfig,
    LLMConfig,
    PromptConfig,
    RetrievalConfig,
    _deep_merge,
    load_yaml_with_inheritance,
)

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
