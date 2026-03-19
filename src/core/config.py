"""Config system for argobot-bench.

All experiment configuration is defined as dataclasses here, loaded
from YAML files. Supports inheritance via ``extends: base.yaml`` with
deep-merge semantics (dicts merge recursively, lists replace, scalars
replace).

Usage:

    config = ExperimentConfig.from_yaml("configs/experiments/chunking_1000.yaml")
    fingerprint = config.index_fingerprint()
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# YAML loading with inheritance
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*.

    Rules (from ARCHITECTURE_PLAN.md §6.4):
    - Nested dicts: merge recursively (override only specified keys)
    - Lists: full replacement (override provides the entire list)
    - Scalars: override replaces base
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_inheritance(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends:`` inheritance.

    The ``extends`` field is relative to the directory containing the
    child config file. Multiple levels of inheritance are supported.

    Raises:
        ConfigValidationError: If a circular inheritance chain is detected.
    """
    return _load_yaml_recursive(path, _visited=None)


def _load_yaml_recursive(
    path: str | Path,
    *,
    _visited: set[Path] | None,
) -> dict[str, Any]:
    """Internal: load YAML with cycle detection."""
    path = Path(path).resolve()

    if _visited is None:
        _visited = set()

    if path in _visited:
        raise ConfigValidationError([f"Circular YAML inheritance detected: {path}"])
    _visited.add(path)

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    parent_path = (path.parent / extends).resolve()
    parent = _load_yaml_recursive(parent_path, _visited=_visited)
    return _deep_merge(parent, raw)


# ---------------------------------------------------------------------------
# Generic component config (used by most pipeline stages)
# ---------------------------------------------------------------------------


@dataclass
class ComponentConfig:
    """Generic config for a pipeline component: a type name + params dict."""

    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ComponentConfig:
        if not data:
            return cls()
        return cls(
            type=data.get("type", ""),
            params=data.get("params", {}),
        )


# ---------------------------------------------------------------------------
# Dedicated config types (§6.2 — stages with important named fields)
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Reusable LLM configuration. Used by generator, query transformer,
    reranker, evaluator, supervisor, and agent LLMs."""

    provider: str = ""
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LLMConfig:
        if not data:
            return cls()
        return cls(
            provider=data.get("provider", ""),
            model_name=data.get("model_name", ""),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens"),
            params=data.get("params", {}),
        )


@dataclass
class RetrievalConfig:
    """Retrieval-specific config with top_k and filters as first-class fields."""

    type: str = "dense"
    top_k_retrieve: int = 10
    top_k_final: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RetrievalConfig:
        if not data:
            return cls()
        return cls(
            type=data.get("type", "dense"),
            top_k_retrieve=data.get("top_k_retrieve", 10),
            top_k_final=data.get("top_k_final", 5),
            filters=data.get("filters", {}),
            params=data.get("params", {}),
        )


@dataclass
class PromptConfig:
    """Prompt template config with independently testable fields."""

    type: str = "chat"
    system_template: str = ""
    context_format: str = "numbered"
    use_chain_of_thought: bool = False
    citation_style: str = "none"
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    max_context_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PromptConfig:
        if not data:
            return cls()
        return cls(
            type=data.get("type", "chat"),
            system_template=data.get("system_template", ""),
            context_format=data.get("context_format", "numbered"),
            use_chain_of_thought=data.get("use_chain_of_thought", False),
            citation_style=data.get("citation_style", "none"),
            few_shot_examples=data.get("few_shot_examples", []),
            max_context_tokens=data.get("max_context_tokens"),
        )


@dataclass
class MemoryConfig:
    """Conversation memory config for agent pipelines."""

    type: str = "none"
    window_size: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MemoryConfig:
        if not data:
            return cls()
        return cls(
            type=data.get("type", "none"),
            window_size=data.get("window_size", 5),
        )


# ---------------------------------------------------------------------------
# Source config (per-document ingest settings)
# ---------------------------------------------------------------------------


@dataclass
class SourceConfig:
    """Config for one source document."""

    name: str = ""
    path: str = ""
    ingest: ComponentConfig = field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourceConfig:
        if not data:
            return cls()
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            ingest=ComponentConfig.from_dict(data.get("ingest")),
        )


# ---------------------------------------------------------------------------
# Indexing config
# ---------------------------------------------------------------------------


@dataclass
class IndexingConfig:
    """Config for the indexing pipeline (sources → vectorstore)."""

    sources: list[SourceConfig] = field(default_factory=list)
    chunking: ComponentConfig = field(default_factory=ComponentConfig)
    embedding: ComponentConfig = field(default_factory=ComponentConfig)
    vectorstore: ComponentConfig = field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> IndexingConfig:
        if not data:
            return cls()
        return cls(
            sources=[SourceConfig.from_dict(s) for s in data.get("sources", [])],
            chunking=ComponentConfig.from_dict(data.get("chunking")),
            embedding=ComponentConfig.from_dict(data.get("embedding")),
            vectorstore=ComponentConfig.from_dict(data.get("vectorstore")),
        )


# ---------------------------------------------------------------------------
# Query config (linear pipeline)
# ---------------------------------------------------------------------------


@dataclass
class QueryConfig:
    """Config for the query pipeline (linear RAG mode)."""

    query_transform: ComponentConfig = field(default_factory=ComponentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: ComponentConfig = field(default_factory=ComponentConfig)
    generation: ComponentConfig = field(default_factory=ComponentConfig)
    generation_llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QueryConfig:
        if not data:
            return cls()
        return cls(
            query_transform=ComponentConfig.from_dict(data.get("query_transform")),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval")),
            reranking=ComponentConfig.from_dict(data.get("reranking")),
            generation=ComponentConfig.from_dict(data.get("generation")),
            generation_llm=LLMConfig.from_dict(data.get("generation_llm")),
            prompt=PromptConfig.from_dict(data.get("prompt")),
        )


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


@dataclass
class SupervisorConfig:
    """Config for the multi-agent supervisor."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    routing: str = "llm"
    routing_prompt: str = ""
    max_iterations: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SupervisorConfig:
        if not data:
            return cls()
        return cls(
            llm=LLMConfig.from_dict(data.get("llm")),
            routing=data.get("routing", "llm"),
            routing_prompt=data.get("routing_prompt", ""),
            max_iterations=data.get("max_iterations", 5),
        )


@dataclass
class AgentDefinitionConfig:
    """Config for one agent or tool in the agent roster."""

    name: str = ""
    type: str = "rag"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tool: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentDefinitionConfig:
        if not data:
            return cls()
        return cls(
            name=data.get("name", ""),
            type=data.get("type", "rag"),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval")),
            prompt=PromptConfig.from_dict(data.get("prompt")),
            tool=data.get("tool", ""),
        )


@dataclass
class AgentConfig:
    """Config for agent-based pipelines (single or multi-agent)."""

    mode: str = "single"
    llm: LLMConfig = field(default_factory=LLMConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agents: list[AgentDefinitionConfig] = field(default_factory=list)
    tools: list[AgentDefinitionConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentConfig:
        if not data:
            return cls()
        return cls(
            mode=data.get("mode", "single"),
            llm=LLMConfig.from_dict(data.get("llm")),
            supervisor=SupervisorConfig.from_dict(data.get("supervisor")),
            memory=MemoryConfig.from_dict(data.get("memory")),
            agents=[AgentDefinitionConfig.from_dict(a) for a in data.get("agents", [])],
            tools=[AgentDefinitionConfig.from_dict(t) for t in data.get("tools", [])],
        )


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Config for the evaluation system."""

    dataset: str = ""
    mode: str = "full"
    metrics: list[str] = field(
        default_factory=lambda: [
            "answer_correctness",
            "context_precision",
            "faithfulness",
            "context_entity_recall",
            "answer_relevancy",
        ]
    )
    retrieval_only_metrics: list[str] = field(
        default_factory=lambda: [
            "context_precision",
            "context_entity_recall",
        ]
    )
    num_runs: int = 3
    evaluator_llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluationConfig:
        if not data:
            return cls()
        return cls(
            dataset=data.get("dataset", ""),
            mode=data.get("mode", "full"),
            metrics=data.get(
                "metrics",
                [
                    "answer_correctness",
                    "context_precision",
                    "faithfulness",
                    "context_entity_recall",
                    "answer_relevancy",
                ],
            ),
            retrieval_only_metrics=data.get(
                "retrieval_only_metrics",
                [
                    "context_precision",
                    "context_entity_recall",
                ],
            ),
            num_runs=data.get("num_runs", 3),
            evaluator_llm=LLMConfig.from_dict(data.get("evaluator_llm")),
        )


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level config for an experiment run."""

    name: str = ""
    description: str = ""
    pipeline_mode: str = "linear"
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Build an ExperimentConfig from a raw dict (e.g. parsed YAML)."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            pipeline_mode=data.get("pipeline_mode", "linear"),
            indexing=IndexingConfig.from_dict(data.get("indexing")),
            query=QueryConfig.from_dict(data.get("query")),
            agent=AgentConfig.from_dict(data.get("agent")),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an ExperimentConfig from a YAML file, resolving inheritance."""
        raw = load_yaml_with_inheritance(path)
        return cls.from_dict(raw)

    def index_fingerprint(self) -> str:
        """Compute a SHA-256 fingerprint (12 hex chars) of indexing config.

        Includes: sources, chunking, embedding, vectorstore.
        Excludes: query, agent, evaluation, pipeline_mode.

        Two experiments with the same fingerprint can share a cached index.
        """
        data = {
            "sources": [
                {
                    "name": s.name,
                    "path": s.path,
                    "ingest": {"type": s.ingest.type, "params": s.ingest.params},
                }
                for s in self.indexing.sources
            ],
            "chunking": {
                "type": self.indexing.chunking.type,
                "params": self.indexing.chunking.params,
            },
            "embedding": {
                "type": self.indexing.embedding.type,
                "params": self.indexing.embedding.params,
            },
            "vectorstore": {
                "type": self.indexing.vectorstore.type,
                "params": self.indexing.vectorstore.params,
            },
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
    """Raised when experiment config validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(
            f"Config validation failed with {len(errors)} error(s):\n{bullet_list}"
        )


def validate_config(config: ExperimentConfig, registry: Any) -> None:
    """Validate an ExperimentConfig against the component registry.

    Checks that all referenced component types are registered,
    logical constraints hold, and required files exist.

    Args:
        config: The parsed experiment config.
        registry: A ComponentRegistry (or duck-typed equivalent with
            ``is_registered`` and ``list_category`` methods).

    Raises:
        ConfigValidationError: If any validation checks fail.
    """
    errors: list[str] = []

    # --- Indexing components (always validated) ---
    if not config.indexing.sources:
        errors.append("indexing.sources is empty — at least one source is required")

    for i, source in enumerate(config.indexing.sources):
        _check_registered(
            errors,
            registry,
            source.ingest.type,
            "ingest",
            f"indexing.sources[{i}].ingest.type",
        )
    _check_registered(
        errors,
        registry,
        config.indexing.chunking.type,
        "chunking",
        "indexing.chunking.type",
    )
    _check_registered(
        errors,
        registry,
        config.indexing.embedding.type,
        "embedding",
        "indexing.embedding.type",
    )
    _check_registered(
        errors,
        registry,
        config.indexing.vectorstore.type,
        "vectorstore",
        "indexing.vectorstore.type",
    )

    # --- Linear pipeline components ---
    if config.pipeline_mode == "linear":
        # Apply same defaults as pipeline/query.py
        qt_type = config.query.query_transform.type or "passthrough"
        _check_registered(
            errors,
            registry,
            qt_type,
            "query_transform",
            "query.query_transform.type",
        )

        _check_registered(
            errors,
            registry,
            config.query.retrieval.type,
            "retrieval",
            "query.retrieval.type",
        )

        rerank_type = config.query.reranking.type or "none"
        _check_registered(
            errors,
            registry,
            rerank_type,
            "reranking",
            "query.reranking.type",
        )

        retrieval_only = config.evaluation.mode == "retrieval_only"
        if not retrieval_only:
            if not config.query.generation.type:
                errors.append(
                    "query.generation.type is empty but evaluation.mode "
                    "is not 'retrieval_only' — a generator is required"
                )
            else:
                _check_registered(
                    errors,
                    registry,
                    config.query.generation.type,
                    "generation",
                    "query.generation.type",
                )
            _check_registered(
                errors,
                registry,
                config.query.prompt.type,
                "prompts",
                "query.prompt.type",
            )
            if not config.query.generation_llm.model_name:
                errors.append(
                    "query.generation_llm.model_name is empty but evaluation.mode "
                    "is not 'retrieval_only' — a model name is required for generation"
                )

        # top_k constraints
        if config.query.retrieval.top_k_retrieve <= 0:
            errors.append(
                f"query.retrieval.top_k_retrieve must be > 0 "
                f"(got {config.query.retrieval.top_k_retrieve})"
            )
        if config.query.retrieval.top_k_final <= 0:
            errors.append(
                f"query.retrieval.top_k_final must be > 0 "
                f"(got {config.query.retrieval.top_k_final})"
            )
        if config.query.retrieval.top_k_final > config.query.retrieval.top_k_retrieve:
            errors.append(
                f"query.retrieval.top_k_final ({config.query.retrieval.top_k_final}) "
                f"> query.retrieval.top_k_retrieve "
                f"({config.query.retrieval.top_k_retrieve})"
            )

    # --- Agent pipeline components ---
    elif config.pipeline_mode == "agent":
        _check_registered(
            errors,
            registry,
            config.agent.memory.type,
            "memory",
            "agent.memory.type",
        )
        if not config.agent.agents and not config.agent.tools:
            errors.append("pipeline_mode is 'agent' but no agents or tools are defined")

    elif config.pipeline_mode not in ("linear", "agent"):
        errors.append(
            f"pipeline_mode '{config.pipeline_mode}' is not recognized "
            "(expected 'linear' or 'agent')"
        )

    # --- Evaluation dataset ---
    if config.evaluation.dataset and not Path(config.evaluation.dataset).exists():
        errors.append(f"Evaluation dataset not found: '{config.evaluation.dataset}'")

    if errors:
        raise ConfigValidationError(errors)


def _check_registered(
    errors: list[str],
    registry: Any,
    type_name: str,
    category: str,
    config_path: str,
) -> None:
    """Append an error if *type_name* is not registered in *category*."""
    if not type_name:
        errors.append(f"{config_path} is empty (no component type specified)")
        return
    if not registry.is_registered(category, type_name):
        available = registry.list_category(category)
        errors.append(
            f"{config_path}: '{type_name}' is not registered in "
            f"category '{category}'. Available: {available}"
        )
