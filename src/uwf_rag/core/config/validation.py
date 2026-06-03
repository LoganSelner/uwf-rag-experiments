"""Experiment-config validation.

``validate_config`` runs before pipeline construction; it checks that referenced
component types are registered, logical constraints hold, and required files
exist, collecting every problem into a single :class:`ConfigValidationError`.
Retrieval-type-specific rules are delegated to each retriever's
``validate_params`` classmethod via a :class:`ValidateContext`.
"""

from __future__ import annotations

from pathlib import Path

from uwf_rag.core.config.errors import ConfigValidationError
from uwf_rag.core.config.models import ExperimentConfig, ValidateContext
from uwf_rag.core.registry import RegistryLike

# Single source of truth for which providers the RAGAS evaluator judge LLM
# supports. Shared with ``evaluation.evaluator`` (whose per-provider builder
# dispatch must cover exactly this set — guarded by a test) so the allowed list
# and the implemented builders can't drift.
SUPPORTED_EVALUATOR_LLM_PROVIDERS: frozenset[str] = frozenset(
    {"ollama", "edenai", "google", "openai"}
)

_SUPPORTED_EVAL_MODES: frozenset[str] = frozenset({"full", "retrieval_only", "none"})

_SUPPORTED_QT_FUSION_MODES: frozenset[str] = frozenset({"rrf", "max"})

_SUPPORTED_AGENT_MODES: frozenset[str] = frozenset({"single", "multi"})


def validate_config(config: ExperimentConfig, registry: RegistryLike) -> None:
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
    if eval_llm.provider and eval_llm.provider not in SUPPORTED_EVALUATOR_LLM_PROVIDERS:
        errors.append(
            f"evaluation.evaluator_llm.provider: '{eval_llm.provider}' is not "
            f"supported. Supported: {sorted(SUPPORTED_EVALUATOR_LLM_PROVIDERS)}"
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
    registry: RegistryLike,
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
    registry: RegistryLike,
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
    registry: RegistryLike,
) -> None:
    """Validate agent-pipeline config (Phase D1: single-agent ReAct).

    The single agent reasons over a roster of ``agent.tools`` using ``agent.llm``
    as its tool-calling reasoning model, and its RAG tool reuses the query
    retrieval + rerank stack. Multi-agent (``mode: multi``) is Phase D2 and is
    rejected here for now.
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
    if not agent.tools:
        errors.append(
            "pipeline_mode is 'agent' but agent.tools is empty — a single agent "
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


def _validate_retrieval_dependencies(
    errors: list[str],
    config: ExperimentConfig,
    registry: RegistryLike,
) -> None:
    """Delegate retrieval-type-specific param validation to the component.

    The retriever class owns its param schema — BM25 needs a sparse index;
    hybrid validates its child roster, nesting prohibition, and fusion
    settings — via a ``validate_params`` classmethod. Core only resolves the
    class from the registry, supplies the cross-config context (sparse-index
    type, top_k budget), and folds the returned error strings into the
    aggregate so the collect-all ``ConfigValidationError`` UX is preserved.
    """
    retrieval = config.query.retrieval
    if not registry.is_registered("retrieval", retrieval.type):
        return  # an unregistered type is already flagged by _check_registered
    retriever_cls = registry.get("retrieval", retrieval.type)
    vctx = ValidateContext(
        registry=registry,
        sparse_index_type=config.indexing.sparse_index.type,
        top_k_retrieve=retrieval.top_k_retrieve,
    )
    errors.extend(retriever_cls.validate_params(retrieval.params or {}, vctx))


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
