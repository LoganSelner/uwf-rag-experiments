"""Tests for src/core/config.py — YAML config system with inheritance."""

from __future__ import annotations

from pathlib import Path

import pytest

from uwf_rag.core.config import (
    AgentConfig,
    AgentDefinitionConfig,
    ComponentConfig,
    ConfigValidationError,
    EvaluationConfig,
    ExperimentConfig,
    IndexingConfig,
    LLMConfig,
    PromptConfig,
    QueryTransformConfig,
    RetrievalConfig,
    _deep_merge,
    load_yaml_with_inheritance,
    validate_config,
)
from uwf_rag.core.registry import registry

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

    def test_circular_extends_raises(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text('extends: "b.yaml"\nname: "a"\n')
        b.write_text('extends: "a.yaml"\nname: "b"\n')
        with pytest.raises(ConfigValidationError, match="Circular"):
            load_yaml_with_inheritance(a)


# -----------------------------------------------------------------------
# ComponentConfig
# -----------------------------------------------------------------------


class TestComponentConfig:
    def test_from_dict(self) -> None:
        cfg = ComponentConfig.from_dict(
            {"type": "recursive_langchain", "params": {"k": 5}}
        )
        assert cfg.type == "recursive_langchain"
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
        assert cfg.indexing.chunking.type == "recursive_langchain"
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
        assert cfg.indexing.chunking.type == "recursive_langchain"


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

    def test_query_transform_config_defaults(self) -> None:
        cfg = QueryTransformConfig.from_dict(None)
        assert cfg.type == "passthrough"
        assert cfg.fusion == "rrf"
        assert cfg.params == {}

    def test_query_transform_config_from_dict(self) -> None:
        cfg = QueryTransformConfig.from_dict(
            {
                "type": "multi_query",
                "fusion": "max",
                "params": {"num_queries": 3},
            }
        )
        assert cfg.type == "multi_query"
        assert cfg.fusion == "max"
        assert cfg.params == {"num_queries": 3}

    def test_query_transform_config_fusion_default_when_missing(self) -> None:
        # Existing YAML configs predate the fusion field; ensure they
        # still resolve to the default.
        cfg = QueryTransformConfig.from_dict({"type": "hyde", "params": {}})
        assert cfg.fusion == "rrf"

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
        assert cfg.run_config.timeout == 600
        assert cfg.run_config.max_retries == 2
        assert cfg.run_config.max_wait == 60
        assert cfg.run_config.max_workers == 2

    def test_evaluation_config_run_config_override(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {
                "run_config": {
                    "timeout": 900,
                    "max_retries": 3,
                    "max_wait": 45,
                    "max_workers": 2,
                }
            }
        )
        assert cfg.run_config.timeout == 900
        assert cfg.run_config.max_retries == 3
        assert cfg.run_config.max_wait == 45
        assert cfg.run_config.max_workers == 2

    def test_evaluation_config_run_config_partial_override_preserves_defaults(
        self,
    ) -> None:
        cfg = EvaluationConfig.from_dict({"run_config": {"timeout": 900}})
        assert cfg.run_config.timeout == 900
        assert cfg.run_config.max_retries == 2
        assert cfg.run_config.max_wait == 60
        assert cfg.run_config.max_workers == 2

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

    def test_evaluation_config_evaluator_embedding(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {
                "evaluator_embedding": {
                    "type": "huggingface",
                    "params": {"model_name": "BAAI/bge-m3", "normalize": True},
                }
            }
        )
        assert cfg.evaluator_embedding.type == "huggingface"
        assert cfg.evaluator_embedding.params["model_name"] == "BAAI/bge-m3"

    def test_evaluation_config_evaluator_embedding_defaults_empty(self) -> None:
        cfg = EvaluationConfig.from_dict(None)
        assert cfg.evaluator_embedding.type == ""
        assert cfg.evaluator_embedding.params == {}

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

    def test_agent_config_defaults(self) -> None:
        cfg = AgentConfig.from_dict(None)
        assert cfg.mode == "single"
        assert cfg.max_iterations == 5
        assert cfg.system_prompt == ""

    def test_agent_config_max_iterations_and_system_prompt(self) -> None:
        cfg = AgentConfig.from_dict(
            {"max_iterations": 8, "system_prompt": "You are an advising agent."}
        )
        assert cfg.max_iterations == 8
        assert cfg.system_prompt == "You are an advising agent."

    def test_agent_definition_description_round_trip(self) -> None:
        cfg = AgentConfig.from_dict(
            {"tools": [{"name": "kb", "type": "rag", "description": "Search docs."}]}
        )
        assert cfg.tools[0].description == "Search docs."


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
                "sources": [{"name": "s", "path": "", "ingest": {"type": "pdf"}}],
                "chunking": {"type": "recursive_langchain"},
                "embedding": {"type": "huggingface"},
                "vectorstore": {"type": "faiss"},
            },
            "query": {
                "retrieval": {"type": "dense", "top_k_retrieve": 10, "top_k_final": 5},
                "generation": {"type": "ollama"},
                "generation_llm": {"model_name": "test-model"},
                "prompt": {"type": "chat"},
            },
            "evaluation": {
                "dataset": "",
                "mode": "full",
                "evaluator_embedding": {"type": "huggingface"},
            },
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
        with pytest.raises(ConfigValidationError, match="recursive_langchain"):
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

    def test_default_qt_fusion_passes(self) -> None:
        cfg = self._make_valid_config()
        assert cfg.query.query_transform.fusion == "rrf"
        validate_config(cfg, registry)  # should not raise

    def test_unknown_qt_fusion_rejected(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.query_transform.fusion = "bogus"
        with pytest.raises(ConfigValidationError, match=r"query_transform\.fusion"):
            validate_config(cfg, registry)

    def test_qt_fusion_max_accepted(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.query_transform.fusion = "max"
        validate_config(cfg, registry)  # should not raise

    def test_qt_fusion_none_rejected(self) -> None:
        # "none" was removed — it silently dropped queries 2..N for
        # multi-emit transformers. Only "rrf" and "max" are valid.
        cfg = self._make_valid_config()
        cfg.query.query_transform.fusion = "none"
        with pytest.raises(ConfigValidationError, match=r"query_transform\.fusion"):
            validate_config(cfg, registry)

    def _make_hybrid_config(self, qt_params: dict) -> ExperimentConfig:
        """Build a minimal valid hybrid+HyDE-style config for branch tests."""
        return self._make_valid_config(
            **{
                "indexing.sparse_index": {"type": "bm25"},
                "query.query_transform": {
                    "type": "hyde",
                    "fusion": "rrf",
                    "params": qt_params,
                },
                "query.retrieval": {
                    "type": "hybrid",
                    "top_k_retrieve": 10,
                    "top_k_final": 5,
                    "params": {
                        "fusion": "rrf",
                        "retrievers": [
                            {"name": "dense", "type": "dense", "top_k": 10},
                            {"name": "bm25", "type": "bm25", "top_k": 10},
                        ],
                    },
                },
            }
        )

    def test_qt_branch_matches_hybrid_child(self) -> None:
        cfg = self._make_hybrid_config(
            {
                "generator_type": "edenai",
                "branch": "dense",
                "original_branch": "bm25",
                "include_original": True,
                "llm": {"model_name": "m"},
            }
        )
        validate_config(cfg, registry)  # should not raise

    def test_qt_branch_unknown_rejected(self) -> None:
        cfg = self._make_hybrid_config(
            {
                "generator_type": "edenai",
                "branch": "splade",  # not a hybrid child
                "llm": {"model_name": "m"},
            }
        )
        with pytest.raises(ConfigValidationError, match="not match any hybrid child"):
            validate_config(cfg, registry)

    def test_qt_original_branch_unknown_rejected(self) -> None:
        cfg = self._make_hybrid_config(
            {
                "generator_type": "edenai",
                "branch": "dense",
                "include_original": True,
                "original_branch": "ghost",  # not a hybrid child
                "llm": {"model_name": "m"},
            }
        )
        with pytest.raises(ConfigValidationError, match="not match any hybrid child"):
            validate_config(cfg, registry)

    def test_qt_branch_ignored_on_non_hybrid(self) -> None:
        # Branch hints are silently ignored when retrieval is not hybrid —
        # the same HyDE config should work with dense retrieval.
        cfg = self._make_valid_config()
        cfg.query.query_transform.type = "hyde"
        cfg.query.query_transform.params = {
            "generator_type": "edenai",
            "branch": "anything",
            "llm": {"model_name": "m"},
        }
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

    def test_none_mode_skips_embedder_requirement(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.mode = "none"
        cfg.evaluation.evaluator_embedding.type = ""
        validate_config(cfg, registry)  # should not raise

    def test_none_mode_still_requires_generation(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.mode = "none"
        cfg.query.generation.type = ""
        with pytest.raises(ConfigValidationError, match="generator is required"):
            validate_config(cfg, registry)

    def test_unknown_eval_mode(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.mode = "bogus"
        with pytest.raises(ConfigValidationError, match=r"evaluation\.mode"):
            validate_config(cfg, registry)

    def test_unknown_pipeline_mode(self) -> None:
        cfg = self._make_valid_config()
        cfg.pipeline_mode = "unknown"
        with pytest.raises(ConfigValidationError, match="not recognized"):
            validate_config(cfg, registry)

    def test_empty_sources_list(self) -> None:
        cfg = self._make_valid_config()
        cfg.indexing.sources = []
        with pytest.raises(ConfigValidationError, match="sources is empty"):
            validate_config(cfg, registry)

    def test_generation_llm_model_name_empty(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.generation_llm.model_name = ""
        with pytest.raises(ConfigValidationError, match="model_name is empty"):
            validate_config(cfg, registry)

    def test_top_k_retrieve_zero(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.retrieval.top_k_retrieve = 0
        with pytest.raises(ConfigValidationError, match="top_k_retrieve must be > 0"):
            validate_config(cfg, registry)

    def test_top_k_final_negative(self) -> None:
        cfg = self._make_valid_config()
        cfg.query.retrieval.top_k_final = -1
        with pytest.raises(ConfigValidationError, match="top_k_final must be > 0"):
            validate_config(cfg, registry)

    def test_evaluation_run_config_timeout_must_be_positive(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.run_config.timeout = 0
        with pytest.raises(
            ConfigValidationError, match=r"evaluation\.run_config\.timeout"
        ):
            validate_config(cfg, registry)

    def test_evaluation_run_config_max_workers_must_be_positive(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.run_config.max_workers = 0
        with pytest.raises(
            ConfigValidationError, match=r"evaluation\.run_config\.max_workers"
        ):
            validate_config(cfg, registry)

    def test_unsupported_evaluator_llm_provider(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.evaluator_llm.provider = "anthropic"
        with pytest.raises(ConfigValidationError, match="not supported"):
            validate_config(cfg, registry)

    def test_supported_evaluator_llm_providers_pass(self) -> None:
        for provider in ("ollama", "edenai", "google", "openai"):
            cfg = self._make_valid_config()
            cfg.evaluation.evaluator_llm.provider = provider
            validate_config(cfg, registry)  # should not raise

    def test_empty_evaluator_llm_provider_passes(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.evaluator_llm.provider = ""
        validate_config(cfg, registry)  # should not raise

    def test_evaluator_embedding_type_empty_raises(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.evaluator_embedding.type = ""
        with pytest.raises(
            ConfigValidationError,
            match=r"evaluation\.evaluator_embedding\.type is empty",
        ):
            validate_config(cfg, registry)

    def test_evaluator_embedding_type_unregistered_raises(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.evaluator_embedding.type = "bogus_embedder"
        with pytest.raises(ConfigValidationError, match="bogus_embedder"):
            validate_config(cfg, registry)

    def test_evaluator_embedding_type_registered_passes(self) -> None:
        cfg = self._make_valid_config()
        cfg.evaluation.evaluator_embedding.type = "huggingface"
        validate_config(cfg, registry)  # should not raise


# -----------------------------------------------------------------------
# Phase A — sparse_index fingerprint + validator
# -----------------------------------------------------------------------


class TestSparseIndexFingerprint:
    def test_empty_sparse_index_does_not_change_fingerprint(
        self, fixtures_dir: Path
    ) -> None:
        # An empty sparse_index ComponentConfig must canonicalize to
        # absence — same fingerprint as a config that doesn't mention it.
        cfg_no_key = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw["indexing"]["sparse_index"] = {"type": "", "params": {}}
        cfg_explicit_empty = ExperimentConfig.from_dict(raw)
        assert cfg_no_key.index_fingerprint() == cfg_explicit_empty.index_fingerprint()

    def test_sparse_index_set_changes_fingerprint(self, fixtures_dir: Path) -> None:
        cfg_no = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw["indexing"]["sparse_index"] = {"type": "bm25", "params": {"k1": 1.5}}
        cfg_yes = ExperimentConfig.from_dict(raw)
        assert cfg_no.index_fingerprint() != cfg_yes.index_fingerprint()

    def test_sparse_index_params_change_fingerprint(self, fixtures_dir: Path) -> None:
        raw1 = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw1["indexing"]["sparse_index"] = {"type": "bm25", "params": {"k1": 1.2}}
        raw2 = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw2["indexing"]["sparse_index"] = {"type": "bm25", "params": {"k1": 1.5}}
        assert (
            ExperimentConfig.from_dict(raw1).index_fingerprint()
            != ExperimentConfig.from_dict(raw2).index_fingerprint()
        )


class TestChunkEnricherFingerprint:
    """Phase C Part 2 — chunk_enricher fingerprint canonicalization."""

    def test_empty_chunk_enricher_does_not_change_fingerprint(
        self, fixtures_dir: Path
    ) -> None:
        # An empty chunk_enricher must canonicalize to absence — adding the
        # field must not invalidate existing (enricher-less) cached indexes.
        cfg_no_key = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw["indexing"]["chunk_enricher"] = {"type": "", "params": {}}
        cfg_explicit_empty = ExperimentConfig.from_dict(raw)
        assert cfg_no_key.index_fingerprint() == cfg_explicit_empty.index_fingerprint()

    def test_chunk_enricher_set_changes_fingerprint(self, fixtures_dir: Path) -> None:
        cfg_no = ExperimentConfig.from_yaml(fixtures_dir / "base.yaml")
        raw = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw["indexing"]["chunk_enricher"] = {
            "type": "contextual",
            "params": {"context_scope": "window"},
        }
        cfg_yes = ExperimentConfig.from_dict(raw)
        assert cfg_no.index_fingerprint() != cfg_yes.index_fingerprint()

    def test_chunk_enricher_params_change_fingerprint(self, fixtures_dir: Path) -> None:
        raw1 = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw1["indexing"]["chunk_enricher"] = {
            "type": "contextual",
            "params": {"context_scope": "page"},
        }
        raw2 = load_yaml_with_inheritance(fixtures_dir / "base.yaml")
        raw2["indexing"]["chunk_enricher"] = {
            "type": "contextual",
            "params": {"context_scope": "window"},
        }
        assert (
            ExperimentConfig.from_dict(raw1).index_fingerprint()
            != ExperimentConfig.from_dict(raw2).index_fingerprint()
        )


class TestValidateConfigSparseAndHybrid:
    """Validator coverage for Phase A retrieval rules."""

    def _make_valid_dense_config(self) -> ExperimentConfig:
        return ExperimentConfig.from_dict(
            {
                "name": "test",
                "pipeline_mode": "linear",
                "indexing": {
                    "sources": [{"name": "s", "path": "", "ingest": {"type": "pdf"}}],
                    "chunking": {"type": "recursive_langchain"},
                    "embedding": {"type": "huggingface"},
                    "vectorstore": {"type": "faiss"},
                },
                "query": {
                    "retrieval": {
                        "type": "dense",
                        "top_k_retrieve": 10,
                        "top_k_final": 5,
                    },
                    "generation": {"type": "ollama"},
                    "generation_llm": {"model_name": "x"},
                    "prompt": {"type": "chat"},
                },
                "evaluation": {
                    "mode": "full",
                    "evaluator_embedding": {"type": "huggingface"},
                },
            }
        )

    def _add_hybrid(
        self,
        cfg: ExperimentConfig,
        *,
        children: list[dict],
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
    ) -> None:
        cfg.indexing.sparse_index.type = "bm25"
        cfg.indexing.sparse_index.params = {}
        cfg.query.retrieval.type = "hybrid"
        cfg.query.retrieval.top_k_retrieve = top_k_retrieve
        cfg.query.retrieval.top_k_final = top_k_final
        params: dict = {
            "fusion": fusion,
            "rrf_k": rrf_k,
            "retrievers": children,
        }
        if weights is not None:
            params["weights"] = weights
        cfg.query.retrieval.params = params

    def test_sparse_index_unregistered_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        cfg.indexing.sparse_index.type = "bogus"
        with pytest.raises(ConfigValidationError, match="bogus"):
            validate_config(cfg, registry)

    def test_chunk_enricher_unregistered_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        cfg.indexing.chunk_enricher.type = "bogus_enricher"
        with pytest.raises(ConfigValidationError, match="bogus_enricher"):
            validate_config(cfg, registry)

    def test_chunk_enricher_registered_passes(self) -> None:
        cfg = self._make_valid_dense_config()
        cfg.indexing.chunk_enricher.type = "contextual"
        validate_config(cfg, registry)  # validation checks registration only

    def test_bm25_retriever_without_sparse_index_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        cfg.query.retrieval.type = "bm25"
        with pytest.raises(ConfigValidationError, match=r"indexing\.sparse_index"):
            validate_config(cfg, registry)

    def test_bm25_retriever_with_sparse_index_passes(self) -> None:
        cfg = self._make_valid_dense_config()
        cfg.indexing.sparse_index.type = "bm25"
        cfg.query.retrieval.type = "bm25"
        validate_config(cfg, registry)

    def test_hybrid_valid_rrf_passes(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        validate_config(cfg, registry)

    def test_hybrid_valid_weighted_passes(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": 0.6, "bm25": 0.4},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        validate_config(cfg, registry)

    def test_hybrid_fewer_than_two_children_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[{"name": "dense", "type": "dense", "top_k": 20, "params": {}}],
        )
        with pytest.raises(ConfigValidationError, match="at least 2"):
            validate_config(cfg, registry)

    def test_hybrid_nested_disallowed(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"name": "inner", "type": "hybrid", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="nested 'hybrid'"):
            validate_config(cfg, registry)

    def test_hybrid_duplicate_names_fail(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"name": "x", "type": "dense", "top_k": 20, "params": {}},
                {"name": "x", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="duplicated"):
            validate_config(cfg, registry)

    def test_hybrid_duplicate_types_without_explicit_names_fail(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"type": "dense", "top_k": 20, "params": {}},
                {"type": "dense", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match=r"duplicated|unique 'name'"):
            validate_config(cfg, registry)

    def test_hybrid_child_top_k_must_be_positive(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"name": "dense", "type": "dense", "top_k": 0, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="top_k must be a positive"):
            validate_config(cfg, registry)

    def test_hybrid_sum_child_top_k_below_top_k_retrieve_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            top_k_retrieve=50,
            children=[
                {"name": "dense", "type": "dense", "top_k": 10, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 10, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="cannot fill the slate"):
            validate_config(cfg, registry)

    def test_hybrid_bm25_child_requires_sparse_index_bm25(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        cfg.indexing.sparse_index.type = ""  # break the dependency
        with pytest.raises(ConfigValidationError):
            validate_config(cfg, registry)

    def test_hybrid_unknown_fusion_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="bogus",
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="not supported"):
            validate_config(cfg, registry)

    def test_hybrid_weighted_keys_must_match_children(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": 0.5, "wrong_name": 0.5},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="must match"):
            validate_config(cfg, registry)

    def test_hybrid_weighted_negative_weight_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": -0.1, "bm25": 0.5},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="non-negative"):
            validate_config(cfg, registry)

    def test_hybrid_weighted_zero_sum_fails(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": 0.0, "bm25": 0.0},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="sum to > 0"):
            validate_config(cfg, registry)

    def test_hybrid_rrf_k_must_be_positive(self) -> None:
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            rrf_k=0,
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        with pytest.raises(ConfigValidationError, match="rrf_k"):
            validate_config(cfg, registry)

    def test_hybrid_weighted_unknown_normalize_fails(self) -> None:
        # Catches typos at config-validation time rather than letting
        # them blow up inside core.fusion on the first query.
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": 0.5, "bm25": 0.5},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        cfg.query.retrieval.params["normalize"] = "zscore"
        with pytest.raises(ConfigValidationError, match=r"normalize.*not supported"):
            validate_config(cfg, registry)

    def test_hybrid_weighted_default_normalize_passes(self) -> None:
        # Unset normalize is fine — defaults to min_max in fusion.
        cfg = self._make_valid_dense_config()
        self._add_hybrid(
            cfg,
            fusion="weighted",
            weights={"dense": 0.5, "bm25": 0.5},
            children=[
                {"name": "dense", "type": "dense", "top_k": 20, "params": {}},
                {"name": "bm25", "type": "bm25", "top_k": 20, "params": {}},
            ],
        )
        validate_config(cfg, registry)


# -----------------------------------------------------------------------
# Phase D1 — agent pipeline validation
# -----------------------------------------------------------------------


class TestValidateAgentConfig:
    """Validator coverage for the single-agent (ReAct) pipeline."""

    def _make_valid_agent_config(self, **overrides: object) -> ExperimentConfig:
        data: dict = {
            "name": "agent_test",
            "pipeline_mode": "agent",
            "indexing": {
                "sources": [{"name": "s", "path": "", "ingest": {"type": "pdf"}}],
                "chunking": {"type": "recursive_langchain"},
                "embedding": {"type": "huggingface"},
                "vectorstore": {"type": "faiss"},
            },
            "query": {
                "retrieval": {"type": "dense", "top_k_retrieve": 10, "top_k_final": 5},
                "reranking": {"type": "none"},
            },
            "agent": {
                "mode": "single",
                "max_iterations": 5,
                "llm": {"provider": "ollama", "model_name": "qwen2.5"},
                "memory": {"type": "none"},
                "tools": [{"name": "knowledge_base", "type": "rag"}],
            },
            "evaluation": {
                "dataset": "",
                "mode": "full",
                "evaluator_embedding": {"type": "huggingface"},
            },
        }
        for key, value in overrides.items():
            parts = key.split(".")
            d = data
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
        return ExperimentConfig.from_dict(data)

    def test_valid_agent_config_passes(self) -> None:
        validate_config(self._make_valid_agent_config(), registry)  # should not raise

    def test_max_iterations_must_be_positive(self) -> None:
        cfg = self._make_valid_agent_config(**{"agent.max_iterations": 0})
        with pytest.raises(ConfigValidationError, match="max_iterations must be > 0"):
            validate_config(cfg, registry)

    def test_mode_multi_not_implemented(self) -> None:
        cfg = self._make_valid_agent_config(**{"agent.mode": "multi"})
        with pytest.raises(ConfigValidationError, match="not implemented yet"):
            validate_config(cfg, registry)

    def test_unknown_mode_rejected(self) -> None:
        cfg = self._make_valid_agent_config(**{"agent.mode": "bogus"})
        with pytest.raises(ConfigValidationError, match=r"agent\.mode"):
            validate_config(cfg, registry)

    def test_llm_provider_must_be_registered_generator(self) -> None:
        cfg = self._make_valid_agent_config(
            **{"agent.llm": {"provider": "anthropic", "model_name": "x"}}
        )
        with pytest.raises(ConfigValidationError, match=r"agent\.llm\.provider"):
            validate_config(cfg, registry)

    def test_llm_model_name_required(self) -> None:
        cfg = self._make_valid_agent_config(
            **{"agent.llm": {"provider": "ollama", "model_name": ""}}
        )
        with pytest.raises(
            ConfigValidationError, match=r"agent\.llm\.model_name is empty"
        ):
            validate_config(cfg, registry)

    def test_tool_type_must_be_registered(self) -> None:
        cfg = self._make_valid_agent_config(
            **{"agent.tools": [{"name": "kb", "type": "bogus_tool"}]}
        )
        with pytest.raises(ConfigValidationError, match="bogus_tool"):
            validate_config(cfg, registry)

    def test_single_mode_requires_tools(self) -> None:
        # A single agent with agents-but-no-tools is degenerate (agents are
        # for D2's supervisor): the single loop needs at least one tool.
        cfg = self._make_valid_agent_config()
        cfg.agent.tools = []
        cfg.agent.agents = [AgentDefinitionConfig(name="a", type="rag")]
        with pytest.raises(
            ConfigValidationError, match=r"single.*needs at least one tool"
        ):
            validate_config(cfg, registry)

    def test_no_tools_or_agents_rejected(self) -> None:
        cfg = self._make_valid_agent_config()
        cfg.agent.tools = []
        cfg.agent.agents = []
        with pytest.raises(ConfigValidationError, match="no agents or tools"):
            validate_config(cfg, registry)

    def test_memory_type_must_be_registered(self) -> None:
        cfg = self._make_valid_agent_config(**{"agent.memory": {"type": "bogus_mem"}})
        with pytest.raises(ConfigValidationError, match="bogus_mem"):
            validate_config(cfg, registry)

    def test_retrieval_only_incompatible_with_agent(self) -> None:
        cfg = self._make_valid_agent_config(**{"evaluation.mode": "retrieval_only"})
        with pytest.raises(
            ConfigValidationError, match=r"retrieval_only.*not compatible"
        ):
            validate_config(cfg, registry)

    def test_agent_reuses_query_retrieval_validation(self) -> None:
        # The RAG tool depends on query.retrieval, so its top_k budget is
        # validated in agent mode too.
        cfg = self._make_valid_agent_config()
        cfg.query.retrieval.top_k_final = 99
        cfg.query.retrieval.top_k_retrieve = 5
        with pytest.raises(ConfigValidationError, match="top_k_final"):
            validate_config(cfg, registry)

    def test_agent_hybrid_rag_tool_requires_sparse_index(self) -> None:
        # Agent RAG tool can use hybrid retrieval; its bm25 dependency is
        # validated through the shared retrieval-core check.
        cfg = self._make_valid_agent_config()
        cfg.query.retrieval.type = "bm25"
        with pytest.raises(ConfigValidationError, match=r"indexing\.sparse_index"):
            validate_config(cfg, registry)

    def test_none_eval_mode_passes(self) -> None:
        cfg = self._make_valid_agent_config(**{"evaluation.mode": "none"})
        cfg.evaluation.evaluator_embedding.type = ""
        validate_config(cfg, registry)  # should not raise
