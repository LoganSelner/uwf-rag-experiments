"""Config system for argobot-bench.

All experiment configuration is defined as Pydantic models here, loaded
from YAML files. Supports inheritance via ``extends: base.yaml`` with
deep-merge semantics (dicts merge recursively, lists replace, scalars
replace).

Usage:

    config = ExperimentConfig.from_yaml("configs/experiments/chunking_1000.yaml")
    fingerprint = config.index_fingerprint()

Each model keeps a ``from_dict`` classmethod (a thin wrapper over
``model_validate`` that maps an empty/``None`` input to all-defaults) so the
historical call sites and the per-section construction in tests stay intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
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


class ComponentConfig(BaseModel):
    """Generic config for a pipeline component: a type name + params dict."""

    type: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ComponentConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Dedicated config types (§6.2 — stages with important named fields)
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Reusable LLM configuration. Used by generator, query transformer,
    reranker, evaluator, supervisor, and agent LLMs."""

    provider: str = ""
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LLMConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class RetrievalConfig(BaseModel):
    """Retrieval-specific config with top_k and filters as first-class fields."""

    type: str = "dense"
    top_k_retrieve: int = 10
    top_k_final: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RetrievalConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class QueryTransformConfig(BaseModel):
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
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QueryTransformConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class PromptConfig(BaseModel):
    """Prompt template config with independently testable fields."""

    type: str = "chat"
    system_template: str = ""
    context_format: str = "numbered"
    use_chain_of_thought: bool = False
    citation_style: str = "none"
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)
    max_context_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PromptConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class MemoryConfig(BaseModel):
    """Conversation memory config for agent pipelines."""

    type: str = "none"
    window_size: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MemoryConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Evaluation run config
# ---------------------------------------------------------------------------


class EvalRunConfig(BaseModel):
    """Execution settings for the RAGAS evaluator."""

    timeout: int = 600
    max_retries: int = 2
    max_wait: int = 60
    max_workers: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvalRunConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Source config (per-document ingest settings)
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Config for one source document."""

    name: str = ""
    path: str = ""
    ingest: ComponentConfig = Field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourceConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Indexing config
# ---------------------------------------------------------------------------


class IndexingConfig(BaseModel):
    """Config for the indexing pipeline (sources → vectorstore + optional sparse)."""

    sources: list[SourceConfig] = Field(default_factory=list)
    chunking: ComponentConfig = Field(default_factory=ComponentConfig)
    embedding: ComponentConfig = Field(default_factory=ComponentConfig)
    vectorstore: ComponentConfig = Field(default_factory=ComponentConfig)
    sparse_index: ComponentConfig = Field(default_factory=ComponentConfig)
    chunk_enricher: ComponentConfig = Field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> IndexingConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Query config (linear pipeline)
# ---------------------------------------------------------------------------


class QueryConfig(BaseModel):
    """Config for the query pipeline (linear RAG mode).

    ``generator`` is a single :class:`LLMConfig` describing the answer model:
    ``generator.provider`` is the ``generation`` registry name and
    ``generator.params`` holds provider extras (e.g. EdenAI ``sub_provider``).
    It replaces the former redundant ``generation`` (type) + ``generation_llm``
    (provider/model) pair — the two always named the same provider.
    """

    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: ComponentConfig = Field(default_factory=ComponentConfig)
    generator: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QueryConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class SupervisorConfig(BaseModel):
    """Config for the multi-agent supervisor."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    routing: str = "llm"
    routing_prompt: str = ""
    max_iterations: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SupervisorConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class AgentDefinitionConfig(BaseModel):
    """Config for one agent or tool in the agent roster.

    For a D1 tool entry: ``type`` is the registry name in category ``tool``
    (e.g. ``rag``), ``name`` is the agent-facing tool name the LLM sees, and
    ``description`` overrides the tool's default description. ``retrieval`` /
    ``prompt`` / ``tool`` are reserved for the D2 multi-agent roster.
    """

    name: str = ""
    type: str = "rag"
    description: str = ""
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    tool: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentDefinitionConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


class AgentConfig(BaseModel):
    """Config for agent-based pipelines (single or multi-agent).

    ``max_iterations`` bounds the single-agent ReAct loop (reason→act→observe);
    ``system_prompt`` is the agent's standing instruction. ``supervisor`` and
    ``agents`` are for the multi-agent mode (Phase D2); single-agent mode reads
    ``tools``.
    """

    mode: str = "single"
    max_iterations: int = 5
    system_prompt: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agents: list[AgentDefinitionConfig] = Field(default_factory=list)
    tools: list[AgentDefinitionConfig] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Config for the evaluation system."""

    dataset: str = ""
    mode: str = "full"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "answer_correctness",
            "context_precision",
            "faithfulness",
            "context_entity_recall",
            "answer_relevancy",
        ]
    )
    retrieval_only_metrics: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_entity_recall",
        ]
    )
    num_runs: int = 3
    run_config: EvalRunConfig = Field(default_factory=EvalRunConfig)
    evaluator_llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluator_embedding: ComponentConfig = Field(default_factory=ComponentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluationConfig:
        if not data:
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Top-level config for an experiment run."""

    name: str = ""
    description: str = ""
    pipeline_mode: str = "linear"
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExperimentConfig:
        """Build an ExperimentConfig from a raw dict (e.g. parsed YAML)."""
        if not data:
            return cls()
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an ExperimentConfig from a YAML file, resolving inheritance."""
        raw = load_yaml_with_inheritance(path)
        return cls.from_dict(raw)

    def index_fingerprint(self) -> str:
        """Compute a SHA-256 fingerprint (12 hex chars) of indexing config.

        Includes: sources, chunking, embedding, vectorstore, and the
        optional ``sparse_index`` / ``chunk_enricher`` (each only when its
        ``type`` is set).
        Excludes: query, agent, evaluation, pipeline_mode.

        Empty-config canonicalization: any ``ComponentConfig`` with an
        empty ``type`` is omitted from the hashed dict so adding a new
        optional indexing component (e.g. ``sparse_index`` in Phase A,
        a chunk enricher in Phase C) doesn't disturb the fingerprint
        for configs that don't use it.

        Two experiments with the same fingerprint can share a cached
        index.
        """
        idx = self.indexing
        data: dict[str, Any] = {
            "sources": [s.model_dump() for s in idx.sources],
            "chunking": idx.chunking.model_dump(),
            "embedding": idx.embedding.model_dump(),
            "vectorstore": idx.vectorstore.model_dump(),
        }
        # Empty-config canonicalization: optional components join the
        # fingerprint only when their ``type`` is set, so adding an unused
        # optional indexing component never disturbs an existing fingerprint.
        if idx.sparse_index.type:
            data["sparse_index"] = idx.sparse_index.model_dump()
        if idx.chunk_enricher.type:
            data["chunk_enricher"] = idx.chunk_enricher.model_dump()
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

_SUPPORTED_QT_FUSION_MODES: frozenset[str] = frozenset({"rrf", "max"})

_SUPPORTED_AGENT_MODES: frozenset[str] = frozenset({"single", "multi"})


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

    if config.indexing.chunk_enricher.type:
        _check_registered(
            errors,
            registry,
            config.indexing.chunk_enricher.type,
            "chunk_enricher",
            "indexing.chunk_enricher.type",
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

        _validate_query_retrieval(errors, config, registry)
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
            if not config.query.generator.provider:
                errors.append(
                    "query.generator.provider is empty but evaluation.mode "
                    "is not 'retrieval_only' — a generator is required"
                )
            else:
                _check_registered(
                    errors,
                    registry,
                    config.query.generator.provider,
                    "generation",
                    "query.generator.provider",
                )
            _check_registered(
                errors,
                registry,
                config.query.prompt.type,
                "prompts",
                "query.prompt.type",
            )
            if not config.query.generator.model_name:
                errors.append(
                    "query.generator.model_name is empty but evaluation.mode "
                    "is not 'retrieval_only' — a model name is required for generation"
                )

    # --- Agent pipeline components ---
    elif config.pipeline_mode == "agent":
        _validate_agent(errors, config, registry)

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


def _validate_query_retrieval(
    errors: list[str],
    config: ExperimentConfig,
    registry: Any,
) -> None:
    """Validate the query-side retrieval stack shared by linear and agent modes.

    The agent's RAG tool reuses ``config.query``'s retriever + reranker, so both
    modes need the retrieval type registered, its dependencies (bm25/hybrid
    wiring) satisfied, and the top_k budget well-formed.
    """
    _check_registered(
        errors,
        registry,
        config.query.retrieval.type,
        "retrieval",
        "query.retrieval.type",
    )
    _validate_retrieval_dependencies(errors, config, registry)

    retrieval = config.query.retrieval
    if retrieval.top_k_retrieve <= 0:
        errors.append(
            f"query.retrieval.top_k_retrieve must be > 0 "
            f"(got {retrieval.top_k_retrieve})"
        )
    if retrieval.top_k_final <= 0:
        errors.append(
            f"query.retrieval.top_k_final must be > 0 (got {retrieval.top_k_final})"
        )
    if retrieval.top_k_final > retrieval.top_k_retrieve:
        errors.append(
            f"query.retrieval.top_k_final ({retrieval.top_k_final}) "
            f"> query.retrieval.top_k_retrieve ({retrieval.top_k_retrieve})"
        )


def _validate_agent(
    errors: list[str],
    config: ExperimentConfig,
    registry: Any,
) -> None:
    """Validate agent-pipeline config (Phase D1: single-agent ReAct).

    The single agent reasons over a roster of ``agent.tools`` (D2's supervisor
    ``agent.agents`` are not implemented yet) using ``agent.llm`` as its
    tool-calling reasoning model, and its RAG tool reuses the query retrieval +
    rerank stack.
    """
    agent = config.agent

    # Reasoning mode.
    if agent.mode not in _SUPPORTED_AGENT_MODES:
        errors.append(
            f"agent.mode: '{agent.mode}' is not supported. "
            f"Supported: {sorted(_SUPPORTED_AGENT_MODES)}"
        )
    elif agent.mode == "multi":
        errors.append(
            "agent.mode 'multi' (multi-agent supervisor) is not implemented yet "
            "(Phase D2). Use 'single'."
        )

    # Reasoning LLM — must be a registered generator with a model name.
    _check_registered(
        errors, registry, agent.llm.provider, "generation", "agent.llm.provider"
    )
    if not agent.llm.model_name:
        errors.append(
            "agent.llm.model_name is empty — a model name is required for the "
            "agent's reasoning LLM"
        )

    # Iteration budget.
    if agent.max_iterations <= 0:
        errors.append(f"agent.max_iterations must be > 0 (got {agent.max_iterations})")

    # Memory.
    _check_registered(
        errors, registry, agent.memory.type, "memory", "agent.memory.type"
    )

    # Tool roster.
    for i, tool in enumerate(agent.tools):
        _check_registered(errors, registry, tool.type, "tool", f"agent.tools[{i}].type")
    if not agent.agents and not agent.tools:
        errors.append("pipeline_mode is 'agent' but no agents or tools are defined")
    elif agent.mode == "single" and not agent.tools:
        errors.append(
            "agent.mode is 'single' but agent.tools is empty — a single agent "
            "needs at least one tool"
        )

    # The RAG tool reuses the query retrieval + rerank stack, so validate it.
    _validate_query_retrieval(errors, config, registry)
    rerank_type = config.query.reranking.type or "none"
    _check_registered(
        errors, registry, rerank_type, "reranking", "query.reranking.type"
    )

    # Agent mode always produces an answer via tool use; retrieval_only (which
    # skips generation) has no meaning here.
    if config.evaluation.mode == "retrieval_only":
        errors.append(
            "evaluation.mode 'retrieval_only' is not compatible with "
            "pipeline_mode 'agent' — the agent produces an answer via tool use; "
            "use 'full' or 'none'"
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
