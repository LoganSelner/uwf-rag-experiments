"""Pydantic config models + the index fingerprint.

The typed tree every experiment is described by. Each model inherits
:class:`ConfigModel`, which supplies a shared ``from_dict`` classmethod (a thin
wrapper over ``model_validate`` that maps an empty/``None`` input to all-defaults)
so the historical call sites and the per-section construction in tests stay
intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from ragbench.core.config.loading import load_yaml_with_inheritance
from ragbench.core.registry import RegistryLike

# ---------------------------------------------------------------------------
# Generic component config (used by most pipeline stages)
# ---------------------------------------------------------------------------


class ConfigModel(BaseModel):
    """Base for every experiment-config model.

    Supplies the shared ``from_dict`` constructor (a thin wrapper over
    ``model_validate`` that maps an empty/``None`` input to all-defaults),
    so the historical call sites and per-section construction in tests stay
    intact without each model re-declaring it.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        if not data:
            return cls()
        return cls.model_validate(data)


class ComponentConfig(ConfigModel):
    """Generic config for a pipeline component: a type name + params dict."""

    type: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dedicated config types (§6.2 — stages with important named fields)
# ---------------------------------------------------------------------------


class LLMConfig(ConfigModel):
    """Reusable LLM configuration. Used by the answer generator, query
    transformer, contextual enricher, evaluator judge, and agent reasoning
    LLMs."""

    provider: str = ""
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(ConfigModel):
    """Retrieval-specific config with top_k and filters as first-class fields."""

    type: str = "dense"
    top_k_retrieve: int = 10
    top_k_final: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class QueryTransformConfig(ConfigModel):
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
    generator: LLMConfig = Field(default_factory=LLMConfig)
    params: dict[str, Any] = Field(default_factory=dict)


class PromptConfig(ConfigModel):
    """Prompt template config with independently testable fields."""

    type: str = "chat"
    system_template: str = ""
    context_format: str = "numbered"
    use_chain_of_thought: bool = False
    citation_style: str = "none"
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)
    max_context_tokens: int | None = None


class MemoryConfig(ConfigModel):
    """Conversation memory config for agent pipelines."""

    type: str = "none"
    window_size: int = 5


# ---------------------------------------------------------------------------
# Evaluation run config
# ---------------------------------------------------------------------------


class EvalRunConfig(ConfigModel):
    """Execution settings for the RAGAS evaluator."""

    timeout: int = 600
    max_retries: int = 2
    max_wait: int = 60
    max_workers: int = 2


# ---------------------------------------------------------------------------
# Source config (per-document ingest settings)
# ---------------------------------------------------------------------------


class SourceConfig(ConfigModel):
    """Config for one source document."""

    name: str = ""
    path: str = ""
    ingest: ComponentConfig = Field(default_factory=ComponentConfig)


# ---------------------------------------------------------------------------
# Indexing config
# ---------------------------------------------------------------------------


class IndexingConfig(ConfigModel):
    """Config for the indexing pipeline (sources → vectorstore + optional sparse)."""

    sources: list[SourceConfig] = Field(default_factory=list)
    chunking: ComponentConfig = Field(default_factory=ComponentConfig)
    embedding: ComponentConfig = Field(default_factory=ComponentConfig)
    vectorstore: ComponentConfig = Field(default_factory=ComponentConfig)
    sparse_index: ComponentConfig = Field(default_factory=ComponentConfig)
    chunk_enricher: ComponentConfig = Field(default_factory=ComponentConfig)


# ---------------------------------------------------------------------------
# Query config (linear pipeline)
# ---------------------------------------------------------------------------


class QueryConfig(ConfigModel):
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


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class AgentDefinitionConfig(ConfigModel):
    """One entry in the agent's tool roster.

    ``type`` is the registry name in category ``tool`` (e.g. ``rag``), ``name``
    is the agent-facing tool name the LLM sees, and ``description`` overrides the
    tool's default description.

    Phase D2 (multi-agent supervisor) will reintroduce per-agent fields
    (retrieval / prompt / sub-tools) here or on a dedicated agent-roster model;
    they are intentionally omitted until that lands so the config surface only
    advertises what is wired today.
    """

    name: str = ""
    type: str = "rag"
    description: str = ""


class AgentConfig(ConfigModel):
    """Config for agent-based pipelines.

    ``max_iterations`` bounds the single-agent ReAct loop (reason→act→observe);
    ``system_prompt`` is the agent's standing instruction; ``tools`` is the
    roster the single agent reasons over. ``mode`` is the seam for the Phase D2
    multi-agent supervisor (``multi`` is rejected by the validator until D2
    lands, at which point the supervisor + agent-roster config is added here).
    """

    mode: str = "single"
    max_iterations: int = 5
    system_prompt: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: list[AgentDefinitionConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


class EvaluationConfig(ConfigModel):
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


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


def _source_content_digest(path: str) -> str:
    """Best-effort content fingerprint for a source file.

    Returns a SHA-256 of the file's bytes so that editing a source document *in
    place* (same path) changes the index fingerprint and triggers a rebuild —
    closing the silently-stale-cache hole where only paths/config were hashed.
    Falls back to a ``(size, mtime)`` marker when the bytes can't be read and to
    ``"absent"`` when no path is set. Every branch is deterministic, so a missing
    file still fingerprints stably (and fails later in ingest with a clear error)
    rather than serving a stale cache.
    """
    if not path:
        return "absent"
    p = Path(path)
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        try:
            st = p.stat()
            return f"size={st.st_size},mtime={int(st.st_mtime)}"
        except OSError:
            return "unreadable"


class ExperimentConfig(ConfigModel):
    """Top-level config for an experiment run."""

    name: str = ""
    description: str = ""
    pipeline_mode: str = "linear"
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an ExperimentConfig from a YAML file, resolving inheritance."""
        raw = load_yaml_with_inheritance(path)
        return cls.from_dict(raw)

    def index_fingerprint(self) -> str:
        """Compute a SHA-256 fingerprint (12 hex chars) of indexing config.

        Includes: sources (name, path, ingest config, **and a content digest
        of each source file** so editing a document in place rebuilds), chunking,
        embedding, vectorstore, and the optional ``sparse_index`` /
        ``chunk_enricher`` (each only when its ``type`` is set).
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
            "sources": [
                {**s.model_dump(), "content_digest": _source_content_digest(s.path)}
                for s in idx.sources
            ],
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
# Validation context (cross-config values a component needs to self-validate)
# ---------------------------------------------------------------------------


@dataclass
class ValidateContext:
    """Cross-config context a component needs to validate its own params.

    Symmetric with :class:`~ragbench.components.build.BuildContext`: the registry
    (for resolving referenced sub-components) plus the config-derived values a
    component can't see on its own (the configured sparse-index type, the
    stage-level retrieve budget). Replaces the ad-hoc ``validate_params(params,
    *, registry, sparse_index_type, top_k_retrieve)`` keyword soup with one
    object that can grow without re-threading every call site.

    Lives in ``core`` (not alongside ``BuildContext`` in ``components``) because
    ``core.validate_config`` constructs it and ``core`` must not import
    ``components``; the component ``validate_params`` classmethods import it from
    here (``components`` → ``core`` is the allowed direction).
    """

    registry: RegistryLike
    sparse_index_type: str = ""
    top_k_retrieve: int = 0
