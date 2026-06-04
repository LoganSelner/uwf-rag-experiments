"""Config system for the RAG experimentation harness.

All experiment configuration is a tree of Pydantic models (:mod:`models`),
loaded from YAML with ``extends:`` inheritance (:mod:`loading`) and checked by
``validate_config`` (:mod:`validation`) before pipeline construction.

This package re-exports the full public surface, so existing imports
(``from ragbench.core.config import ExperimentConfig, validate_config, ...``)
keep working unchanged after the split from a single module.

Usage:

    config = ExperimentConfig.from_yaml("configs/experiments/chunking_1000.yaml")
    fingerprint = config.index_fingerprint()
"""

from __future__ import annotations

from ragbench.core.config.errors import ConfigValidationError
from ragbench.core.config.loading import (
    _deep_merge,
    _selects_different_impl,
    load_yaml_with_inheritance,
)
from ragbench.core.config.models import (
    AgentConfig,
    AgentSpecConfig,
    ComponentConfig,
    ConfigModel,
    EvalRunConfig,
    EvaluationConfig,
    ExperimentConfig,
    IndexingConfig,
    LLMConfig,
    MemoryConfig,
    PromptConfig,
    QueryConfig,
    QueryTransformConfig,
    RetrievalConfig,
    SourceConfig,
    ToolConfig,
    ValidateContext,
)
from ragbench.core.config.validation import (
    SUPPORTED_EVALUATOR_LLM_PROVIDERS,
    validate_config,
)

__all__ = [
    "SUPPORTED_EVALUATOR_LLM_PROVIDERS",
    "AgentConfig",
    "AgentSpecConfig",
    "ComponentConfig",
    "ConfigModel",
    "ConfigValidationError",
    "EvalRunConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "IndexingConfig",
    "LLMConfig",
    "MemoryConfig",
    "PromptConfig",
    "QueryConfig",
    "QueryTransformConfig",
    "RetrievalConfig",
    "SourceConfig",
    "ToolConfig",
    "ValidateContext",
    "_deep_merge",
    "_selects_different_impl",
    "load_yaml_with_inheritance",
    "validate_config",
]
