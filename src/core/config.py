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

    Rules (from ROADMAP.md §6.4):
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
class QueryTransformConfig:
    """Query-transformer config with ``fusion`` as a first-class field.

    Parallel to :class:`RetrievalConfig` — the fusion strategy applied
    when a transformer emits N>1 search queries (HyDE, multi-query) is
    too important to bury in a generic ``params`` dict. Promotes it to
    a discoverable, validated experimental variable.

    ``fusion`` is the *pipeline-level* fusion across transformer
    outputs (the intra-branch fusion inside HybridRetriever.retrieve_multi).
    It is independent of the hybrid retriever's own ``params.fusion``,
    which combines results across retrieval methods.
    """

    type: str = "passthrough"
    fusion: str = "rrf"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QueryTransformConfig:
        if not data:
            return cls()
        return cls(
            type=data.get("type", "passthrough"),
            fusion=data.get("fusion", "rrf"),
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
# Evaluation run config
# ---------------------------------------------------------------------------


@dataclass
class EvalRunConfig:
    """Execution settings for the RAGAS evaluator."""

    timeout: int = 600
    max_retries: int = 2
    max_wait: int = 60
    max_workers: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvalRunConfig:
        if not data:
            return cls()
        return cls(
            timeout=data.get("timeout", 600),
            max_retries=data.get("max_retries", 2),
            max_wait=data.get("max_wait", 60),
            max_workers=data.get("max_workers", 2),
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
    """Config for the indexing pipeline (sources → vectorstore + optional sparse)."""

    sources: list[SourceConfig] = field(default_factory=list)
    chunking: ComponentConfig = field(default_factory=ComponentConfig)
    embedding: ComponentConfig = field(default_factory=ComponentConfig)
    vectorstore: ComponentConfig = field(default_factory=ComponentConfig)
    sparse_index: ComponentConfig = field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> IndexingConfig:
        if not data:
            return cls()
        return cls(
            sources=[SourceConfig.from_dict(s) for s in data.get("sources", [])],
            chunking=ComponentConfig.from_dict(data.get("chunking")),
            embedding=ComponentConfig.from_dict(data.get("embedding")),
            vectorstore=ComponentConfig.from_dict(data.get("vectorstore")),
            sparse_index=ComponentConfig.from_dict(data.get("sparse_index")),
        )


# ---------------------------------------------------------------------------
# Query config (linear pipeline)
# ---------------------------------------------------------------------------


@dataclass
class QueryConfig:
    """Config for the query pipeline (linear RAG mode)."""

    query_transform: QueryTransformConfig = field(default_factory=QueryTransformConfig)
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
            query_transform=QueryTransformConfig.from_dict(data.get("query_transform")),
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
    run_config: EvalRunConfig = field(default_factory=EvalRunConfig)
    evaluator_llm: LLMConfig = field(default_factory=LLMConfig)
    evaluator_embedding: ComponentConfig = field(default_factory=ComponentConfig)

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
            run_config=EvalRunConfig.from_dict(data.get("run_config")),
            evaluator_llm=LLMConfig.from_dict(data.get("evaluator_llm")),
            evaluator_embedding=ComponentConfig.from_dict(
                data.get("evaluator_embedding")
            ),
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

        Includes: sources, chunking, embedding, vectorstore, and
        ``sparse_index`` (the latter only when its ``type`` is set).
        Excludes: query, agent, evaluation, pipeline_mode.

        Empty-config canonicalization: any ``ComponentConfig`` with an
        empty ``type`` is omitted from the hashed dict so adding a new
        optional indexing component (e.g. ``sparse_index`` in Phase A,
        a chunk enricher in Phase C) doesn't disturb the fingerprint
        for configs that don't use it.

        Two experiments with the same fingerprint can share a cached
        index.
        """
        data: dict[str, Any] = {
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
        if self.indexing.sparse_index.type:
            data["sparse_index"] = {
                "type": self.indexing.sparse_index.type,
                "params": self.indexing.sparse_index.params,
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


_SUPPORTED_EVAL_LLM_PROVIDERS: frozenset[str] = frozenset(
    {"ollama", "edenai", "google", "openai"}
)

_SUPPORTED_EVAL_MODES: frozenset[str] = frozenset({"full", "retrieval_only", "none"})

_SUPPORTED_QT_FUSION_MODES: frozenset[str] = frozenset({"rrf", "max", "none"})


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

    # --- Evaluation run config ---
    if config.evaluation.run_config.timeout <= 0:
        errors.append(
            f"evaluation.run_config.timeout must be > 0 "
            f"(got {config.evaluation.run_config.timeout})"
        )
    if config.evaluation.run_config.max_workers <= 0:
        errors.append(
            f"evaluation.run_config.max_workers must be > 0 "
            f"(got {config.evaluation.run_config.max_workers})"
        )
    if config.evaluation.run_config.max_wait <= 0:
        errors.append(
            f"evaluation.run_config.max_wait must be > 0 "
            f"(got {config.evaluation.run_config.max_wait})"
        )
    if config.evaluation.run_config.max_retries < 0:
        errors.append(
            f"evaluation.run_config.max_retries must be >= 0 "
            f"(got {config.evaluation.run_config.max_retries})"
        )

    # --- Evaluation mode ---
    if config.evaluation.mode not in _SUPPORTED_EVAL_MODES:
        errors.append(
            f"evaluation.mode: '{config.evaluation.mode}' is not supported. "
            f"Supported: {sorted(_SUPPORTED_EVAL_MODES)}"
        )

    # --- Evaluator provider + embedding ---
    # Mode "none" runs the pipeline against the dataset but skips scoring,
    # so the judge LLM and judge embedder are not needed.
    scoring_enabled = config.evaluation.mode != "none"

    eval_llm = config.evaluation.evaluator_llm
    if eval_llm.provider and eval_llm.provider not in _SUPPORTED_EVAL_LLM_PROVIDERS:
        errors.append(
            f"evaluation.evaluator_llm.provider: '{eval_llm.provider}' is not "
            f"supported. Supported: {sorted(_SUPPORTED_EVAL_LLM_PROVIDERS)}"
        )

    eval_emb = config.evaluation.evaluator_embedding
    if eval_emb.type:
        _check_registered(
            errors,
            registry,
            eval_emb.type,
            "embedding",
            "evaluation.evaluator_embedding.type",
        )
    elif scoring_enabled:
        errors.append(
            "evaluation.evaluator_embedding.type is empty — "
            "a dedicated evaluation embedder is required for consistent measurement"
        )

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

    if config.indexing.sparse_index.type:
        _check_registered(
            errors,
            registry,
            config.indexing.sparse_index.type,
            "sparse_index",
            "indexing.sparse_index.type",
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

        qt_fusion = config.query.query_transform.fusion
        if qt_fusion not in _SUPPORTED_QT_FUSION_MODES:
            errors.append(
                f"query.query_transform.fusion: '{qt_fusion}' is not supported. "
                f"Supported: {sorted(_SUPPORTED_QT_FUSION_MODES)}"
            )

        _check_registered(
            errors,
            registry,
            config.query.retrieval.type,
            "retrieval",
            "query.retrieval.type",
        )

        _validate_retrieval_dependencies(errors, config, registry)
        _validate_qt_branch_references(errors, config)

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


_SUPPORTED_FUSION_MODES: frozenset[str] = frozenset({"rrf", "weighted"})
_SUPPORTED_NORMALIZE_MODES: frozenset[str] = frozenset({"min_max", "none"})


def _validate_retrieval_dependencies(
    errors: list[str],
    config: ExperimentConfig,
    registry: Any,
) -> None:
    """Phase A retrieval validations beyond simple registry checks.

    Covers:
    - `bm25` retriever requires `indexing.sparse_index` to be set.
    - `hybrid` retriever: child specs, fusion mode, weights, nesting
      prohibition, sparse-index requirement when any child is `bm25`.
    """
    retrieval = config.query.retrieval

    if retrieval.type == "bm25":
        if not config.indexing.sparse_index.type:
            errors.append(
                "query.retrieval.type is 'bm25' but indexing.sparse_index "
                "is not configured"
            )
        return

    if retrieval.type != "hybrid":
        return

    params = retrieval.params or {}
    children = params.get("retrievers")
    if not isinstance(children, list):
        errors.append(
            "query.retrieval.params.retrievers must be a list of "
            "sub-retriever specs for hybrid retrieval"
        )
        return
    if len(children) < 2:
        errors.append(
            f"query.retrieval.params.retrievers must have at least 2 "
            f"entries for hybrid (got {len(children)})"
        )

    seen_names: set[str] = set()
    seen_types: dict[str, int] = {}
    has_bm25_child = False
    child_top_k_sum = 0

    for i, child in enumerate(children):
        path = f"query.retrieval.params.retrievers[{i}]"
        if not isinstance(child, dict):
            errors.append(f"{path} must be a dict")
            continue
        sub_type = child.get("type", "")
        if sub_type == "hybrid":
            errors.append(f"{path}.type: nested 'hybrid' is not allowed")
            continue
        if not sub_type:
            errors.append(f"{path}.type is empty")
        else:
            _check_registered(errors, registry, sub_type, "retrieval", f"{path}.type")
            seen_types[sub_type] = seen_types.get(sub_type, 0) + 1
            if sub_type == "bm25":
                has_bm25_child = True

        sub_name = str(child.get("name", sub_type))
        if sub_name in seen_names:
            errors.append(
                f"{path}.name: '{sub_name}' is duplicated in this hybrid "
                "(child names must be unique)"
            )
        else:
            seen_names.add(sub_name)

        sub_top_k = child.get("top_k")
        if not isinstance(sub_top_k, int) or sub_top_k <= 0:
            errors.append(f"{path}.top_k must be a positive int (got {sub_top_k!r})")
        else:
            child_top_k_sum += sub_top_k

    # Duplicate types without explicit names are disallowed — the
    # default name (the type) would collide.
    for sub_type, count in seen_types.items():
        if count > 1:
            explicit = sum(
                1
                for c in children
                if isinstance(c, dict) and c.get("type") == sub_type and c.get("name")
            )
            if explicit < count:
                errors.append(
                    f"query.retrieval.params.retrievers: type '{sub_type}' "
                    f"appears {count} times — each duplicate child must "
                    "supply an explicit unique 'name'"
                )

    if (
        child_top_k_sum
        and retrieval.top_k_retrieve > 0
        and child_top_k_sum < retrieval.top_k_retrieve
    ):
        errors.append(
            f"query.retrieval.params.retrievers: sum of child top_k "
            f"({child_top_k_sum}) < query.retrieval.top_k_retrieve "
            f"({retrieval.top_k_retrieve}) — fusion cannot fill the slate"
        )

    if has_bm25_child and config.indexing.sparse_index.type != "bm25":
        errors.append(
            "hybrid has a 'bm25' child but indexing.sparse_index.type "
            f"is '{config.indexing.sparse_index.type}' (expected 'bm25')"
        )

    fusion = params.get("fusion", "rrf")
    if fusion not in _SUPPORTED_FUSION_MODES:
        errors.append(
            f"query.retrieval.params.fusion: '{fusion}' is not supported. "
            f"Supported: {sorted(_SUPPORTED_FUSION_MODES)}"
        )

    if fusion == "rrf":
        rrf_k = params.get("rrf_k", 60)
        if not isinstance(rrf_k, int) or rrf_k <= 0:
            errors.append(
                f"query.retrieval.params.rrf_k must be a positive int (got {rrf_k!r})"
            )

    if fusion == "weighted":
        normalize = params.get("normalize", "min_max")
        if normalize not in _SUPPORTED_NORMALIZE_MODES:
            errors.append(
                f"query.retrieval.params.normalize: '{normalize}' is not "
                f"supported. Supported: {sorted(_SUPPORTED_NORMALIZE_MODES)}"
            )

        weights = params.get("weights")
        if not isinstance(weights, dict):
            errors.append(
                "query.retrieval.params.weights is required for weighted "
                "fusion and must be a dict keyed by sub-retriever name"
            )
        else:
            if set(weights.keys()) != seen_names:
                errors.append(
                    f"query.retrieval.params.weights keys "
                    f"{sorted(weights.keys())} must match sub-retriever "
                    f"names {sorted(seen_names)}"
                )
            weight_values = list(weights.values())
            if any(not isinstance(w, (int, float)) or w < 0 for w in weight_values):
                errors.append(
                    "query.retrieval.params.weights values must be "
                    f"non-negative numbers (got {weight_values})"
                )
            elif sum(weight_values) <= 0:
                errors.append(
                    "query.retrieval.params.weights must sum to > 0 "
                    f"(got {sum(weight_values)})"
                )


def _validate_qt_branch_references(
    errors: list[str],
    config: ExperimentConfig,
) -> None:
    """Static check that any transformer ``branch`` hints reference real
    hybrid children.

    Catches typos pre-runtime. The runtime guard in
    :meth:`HybridRetriever.retrieve_multi` is authoritative — this is a
    courtesy check that fails fast on a misconfiguration.

    No-op when retrieval is not ``hybrid`` (branch hints are silently
    ignored by non-hybrid retrievers, by design).
    """
    if config.query.retrieval.type != "hybrid":
        return

    children = config.query.retrieval.params.get("retrievers") or []
    if not isinstance(children, list):
        return  # the retrieval validator already flagged this.

    known_branches: set[str] = set()
    for child in children:
        if not isinstance(child, dict):
            continue
        sub_type = child.get("type", "")
        sub_name = str(child.get("name", sub_type))
        if sub_name:
            known_branches.add(sub_name)

    qt_params = config.query.query_transform.params or {}
    for key in ("branch", "original_branch"):
        value = qt_params.get(key)
        if value is None:
            continue
        if value not in known_branches:
            errors.append(
                f"query.query_transform.params.{key}: '{value}' does not "
                f"match any hybrid child name. Known children: "
                f"{sorted(known_branches)}"
            )
