"""RAGAS evaluation integration.

Runs the pipeline against an evaluation dataset, computes RAGAS metrics,
and aggregates results across multiple runs for statistical significance.

See ROADMAP.md Section 9 for full specification.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import statistics
from typing import Any

from uwf_rag.core.config import (
    SUPPORTED_EVALUATOR_LLM_PROVIDERS,
    EvaluationConfig,
    LLMConfig,
)
from uwf_rag.core.registry import registry
from uwf_rag.core.types import EvalSample, ExperimentResult, Queryable, ScoredSample

logger = logging.getLogger(__name__)


def _load_dataset(path: str) -> list[dict[str, str]]:
    """Load an evaluation dataset from JSONL.

    Expected format: one {query, reference} JSON object per line.
    An optional ``id`` field provides a stable identifier for each
    item.  If absent, sequential 1-based IDs are assigned.
    """
    data: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not data:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    for i, item in enumerate(data):
        if "query" not in item or "reference" not in item:
            raise ValueError(f"Dataset item {i} missing 'query' or 'reference': {item}")
        if "id" not in item:
            item["id"] = str(i + 1)
    return data


@functools.cache
def _build_metric_map() -> dict[str, Any]:
    """Build RAGAS metric instances.

    Imports from private submodules to avoid the deprecation warnings
    that ``ragas.metrics`` top-level imports trigger in RAGAS 0.4.x.
    """
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._answer_similarity import AnswerSimilarity
    from ragas.metrics._context_entities_recall import ContextEntityRecall
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.metrics._faithfulness import Faithfulness

    return {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "answer_correctness": AnswerCorrectness(),
        "answer_similarity": AnswerSimilarity(),
        "factual_correctness": FactualCorrectness(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "context_entity_recall": ContextEntityRecall(),
    }


def _wrap_embedder_for_ragas(embedder: Any) -> Any:
    """Wrap a ``BaseEmbedder`` as a RAGAS-native ``BaseRagasEmbedding``.

    Bridges the evaluator's dedicated embedding model (built from
    ``evaluator_embedding`` config via the component registry) to the
    interface that RAGAS ``evaluate()`` expects.

    The class is defined inside the function so the ``ragas.embeddings``
    import stays deferred (consistent with all other RAGAS imports in
    this module).
    """
    from ragas.embeddings import BaseRagasEmbedding

    class _EmbedderAdapter(BaseRagasEmbedding):
        def __init__(self, base_embedder: Any) -> None:
            super().__init__()
            self._embedder = base_embedder

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self._embedder.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self._embedder.embed_query(text)

        # Legacy LangChain interface — RAGAS 0.4.x metrics like
        # answer_relevancy still call embed_query/embed_documents
        # internally instead of the modern embed_text/embed_texts API.
        def embed_query(self, text: str) -> list[float]:
            return self._embedder.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embedder.embed_query(t) for t in texts]

    return _EmbedderAdapter(embedder)


class Evaluator:
    """Runs evaluation against a dataset using RAGAS metrics.

    Supports full mode (retrieve + generate + all metrics),
    retrieval-only mode (retrieve + retrieval metrics only), and
    none mode (run the pipeline against the dataset but skip scoring —
    intended for smoke tests that verify end-to-end execution without
    paying for an LLM judge).
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config
        self._embedder_adapter: Any | None = None

    @property
    def active_metrics(self) -> list[str]:
        """Return the metrics to compute based on evaluation mode."""
        if self._config.mode == "none":
            return []
        if self._config.mode == "retrieval_only":
            return list(self._config.retrieval_only_metrics)
        return list(self._config.metrics)

    def evaluate(
        self,
        pipeline: Queryable,
        *,
        experiment_name: str = "",
    ) -> ExperimentResult:
        """Run the full evaluation: multi-run execution + aggregation.

        The evaluator builds its own embedder from ``evaluator_embedding``
        config via the component registry, ensuring all experiments are
        measured with a consistent embedding model.

        Args:
            pipeline: Anything satisfying the Queryable protocol.
            experiment_name: Label for this experiment (used in results).

        Returns:
            ExperimentResult with aggregated and per-run metrics.
        """
        dataset = _load_dataset(self._config.dataset)
        num_runs = self._config.num_runs
        scoring_enabled = self._config.mode != "none"

        # Build dedicated evaluation embedder from config. Skipped in
        # mode="none" because no metrics are computed.
        eval_emb_cfg = self._config.evaluator_embedding
        if scoring_enabled and eval_emb_cfg.type:
            embedder_cls = registry.get("embedding", eval_emb_cfg.type)
            embedder_instance = embedder_cls(config=eval_emb_cfg.params)
            self._embedder_adapter = _wrap_embedder_for_ragas(embedder_instance)

        logger.info(
            "Starting evaluation: %d questions, %d runs, mode=%s",
            len(dataset),
            num_runs,
            self._config.mode,
        )

        per_run_metrics: list[dict[str, float]] = []
        per_run_samples: list[list[ScoredSample]] = []

        for run_idx in range(1, num_runs + 1):
            logger.info("Run %d/%d", run_idx, num_runs)
            samples = self._run_once(pipeline, dataset)
            if not samples:
                logger.error(
                    "All queries failed in run %d — skipping metrics.",
                    run_idx,
                )
                per_run_metrics.append({})
                per_run_samples.append([])
                continue

            if scoring_enabled:
                metrics, sample_scores = self._compute_metrics(samples)
            else:
                metrics = {}
                sample_scores = [dict[str, float | None]() for _ in samples]
            per_run_metrics.append(metrics)

            scored = [
                ScoredSample(
                    id=s.id,
                    query=s.query,
                    response=s.response,
                    retrieved_contexts=s.retrieved_contexts,
                    reference=s.reference,
                    scores=scores,
                    metadata=s.metadata,
                )
                for s, scores in zip(samples, sample_scores, strict=True)
            ]
            per_run_samples.append(scored)

            if scoring_enabled:
                logger.info("Run %d metrics: %s", run_idx, metrics)
            else:
                logger.info(
                    "Run %d: scoring skipped (mode=none); %d samples captured",
                    run_idx,
                    len(samples),
                )

        # Aggregate across runs
        aggregated = self._aggregate_metrics(per_run_metrics)
        logger.info("Aggregated metrics: %s", aggregated)

        return ExperimentResult(
            experiment_name=experiment_name,
            metrics=aggregated,
            per_run_metrics=per_run_metrics,
            per_run_samples=per_run_samples,
            num_runs=num_runs,
        )

    def _run_once(
        self,
        pipeline: Queryable,
        dataset: list[dict[str, str]],
    ) -> list[EvalSample]:
        """Run the pipeline on each dataset query, producing EvalSamples.

        Failed queries are logged and skipped. The run continues with
        remaining queries so one bad response doesn't crash the evaluation.
        """
        samples: list[EvalSample] = []
        failed = 0
        total = len(dataset)
        for i, item in enumerate(dataset, start=1):
            try:
                result = pipeline.query(item["query"])
                contexts = [rc.chunk.content for rc in result.retrieved_chunks]
                samples.append(
                    EvalSample(
                        id=item["id"],
                        query=item["query"],
                        response=result.answer,
                        retrieved_contexts=contexts,
                        reference=item["reference"],
                        metadata=result.metadata,
                    )
                )
            except Exception:
                failed += 1
                logger.exception(
                    "Query %d/%d failed: %s",
                    i,
                    total,
                    item["query"][:80],
                )
        if failed:
            logger.warning(
                "%d/%d queries failed in this run",
                failed,
                total,
            )
        return samples

    def _compute_metrics(
        self, samples: list[EvalSample]
    ) -> tuple[dict[str, float], list[dict[str, float | None]]]:
        """Compute RAGAS metrics on a list of EvalSamples.

        Uses the RAGAS 0.4.x API (EvaluationDataset + SingleTurnSample)
        with LangchainLLMWrapper for provider compatibility.

        Returns:
            A tuple of (aggregate_metrics, per_sample_scores).
            aggregate_metrics: mean per metric across all samples.
            per_sample_scores: one dict per sample with individual
            metric scores.  NaN values are converted to None.
        """
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample

        metric_map = _build_metric_map()
        active = self.active_metrics
        ragas_metrics = [metric_map[name] for name in active if name in metric_map]

        if not ragas_metrics:
            logger.warning("No valid RAGAS metrics to compute.")
            return {}, [{} for _ in samples]

        ragas_samples = [
            SingleTurnSample(
                user_input=s.query,
                response=s.response,
                retrieved_contexts=s.retrieved_contexts,
                reference=s.reference,
            )
            for s in samples
        ]

        eval_kwargs: dict[str, Any] = {
            "run_config": self._make_run_config(),
        }

        if (
            self._config.evaluator_llm.provider
            and self._config.evaluator_llm.model_name
        ):
            eval_kwargs["llm"] = self._build_evaluator_llm()

        if self._embedder_adapter is not None:
            eval_kwargs["embeddings"] = self._embedder_adapter

        ragas_result: Any = evaluate(
            dataset=EvaluationDataset(samples=ragas_samples),  # type: ignore[arg-type]
            metrics=ragas_metrics,
            **eval_kwargs,
        )

        # ragas_result.scores is a list of per-sample dicts,
        # e.g. [{"faithfulness": 0.9, "answer_correctness": 0.8}].
        scores_list: list[dict[str, Any]] = ragas_result.scores

        # Aggregate: mean per metric across samples.
        # Filter both None and NaN — RAGAS returns NaN when the
        # LLM judge fails to parse a response.
        result_dict: dict[str, float] = {}
        for name in active:
            values = [
                float(s[name]) for s in scores_list if name in s and s[name] is not None
            ]
            values = [v for v in values if not math.isnan(v)]
            if values:
                result_dict[name] = sum(values) / len(values)

        # Per-sample: preserve individual scores, NaN → None.
        per_sample: list[dict[str, float | None]] = []
        for scores in scores_list:
            sample_scores: dict[str, float | None] = {}
            for name in active:
                if name in scores:
                    val = scores[name]
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        sample_scores[name] = None
                    else:
                        sample_scores[name] = float(val)
            per_sample.append(sample_scores)

        return result_dict, per_sample

    def _make_run_config(self) -> Any:
        """Build a RAGAS RunConfig from evaluation settings."""
        from ragas import RunConfig

        cfg = self._config.run_config
        return RunConfig(
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            max_wait=cfg.max_wait,
            max_workers=cfg.max_workers,
        )

    def _build_evaluator_llm(self) -> Any:
        """Build a RAGAS-compatible LLM from config.

        Dispatches on ``evaluator_llm.provider`` to a per-provider builder that
        returns a LangChain chat model, then wraps it once in
        ``LangchainLLMWrapper`` (the shape RAGAS expects).
        """
        from ragas.llms.base import LangchainLLMWrapper

        llm_config = self._config.evaluator_llm
        builders = self._judge_builders()
        builder = builders.get(llm_config.provider)
        if builder is None:
            raise ValueError(
                f"Unsupported evaluator LLM provider: '{llm_config.provider}'. "
                f"Supported: {sorted(builders)}."
            )
        return LangchainLLMWrapper(builder(llm_config))

    @classmethod
    def _judge_builders(cls) -> dict[str, Any]:
        """Per-provider judge-LLM builders.

        Keyed by provider name; the key set is asserted to equal
        ``SUPPORTED_EVALUATOR_LLM_PROVIDERS`` (the set ``validate_config``
        checks against), so the validated allow-list and the implemented
        builders can't drift. These are LangChain chat models because RAGAS
        requires that shape — a separate construction path from the pipeline's
        ``build_generator`` (SDK clients) by necessity.
        """
        builders: dict[str, Any] = {
            "ollama": cls._build_ollama_judge,
            "edenai": cls._build_edenai_judge,
            "google": cls._build_google_judge,
            "openai": cls._build_openai_judge,
        }
        if set(builders) != SUPPORTED_EVALUATOR_LLM_PROVIDERS:
            raise RuntimeError(
                "Evaluator judge builders "
                f"{sorted(builders)} drifted from the validated provider set "
                f"{sorted(SUPPORTED_EVALUATOR_LLM_PROVIDERS)}."
            )
        return builders

    @staticmethod
    def _build_ollama_judge(llm_config: LLMConfig) -> Any:
        from langchain_ollama import OllamaLLM

        kwargs: dict[str, Any] = {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature,
        }
        base_url = llm_config.params.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaLLM(**kwargs)

    @staticmethod
    def _build_edenai_judge(llm_config: LLMConfig) -> Any:
        import os

        from langchain_community.chat_models.edenai import ChatEdenAI

        api_key = os.environ.get("EDENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "EDENAI_API_KEY environment variable is required "
                "for Eden AI evaluator LLM."
            )
        sub_provider = llm_config.params.get("sub_provider", "openai")
        return ChatEdenAI(
            provider=sub_provider,
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens or 1024,
            edenai_api_key=api_key,  # type: ignore[arg-type]
        )

    @staticmethod
    def _build_google_judge(llm_config: LLMConfig) -> Any:
        import os

        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is "
                "required for Google evaluator LLM."
            )
        return ChatGoogleGenerativeAI(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_output_tokens=llm_config.max_tokens or 1024,
            google_api_key=api_key,
        )

    @staticmethod
    def _build_openai_judge(llm_config: LLMConfig) -> Any:
        import os

        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required "
                "for OpenAI evaluator LLM."
            )
        return ChatOpenAI(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens or 1024,  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
        )

    @staticmethod
    def _aggregate_metrics(
        per_run: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics across runs as mean +/- std.

        Returns dict with keys like "answer_correctness" (mean)
        and "answer_correctness_std" (standard deviation).
        """
        if not per_run:
            return {}

        all_keys: set[str] = set()
        for run in per_run:
            all_keys.update(run.keys())
        aggregated: dict[str, float] = {}

        for key in all_keys:
            values = [
                run[key]
                for run in per_run
                if key in run and run[key] is not None and not math.isnan(run[key])
            ]
            if values:
                aggregated[key] = statistics.mean(values)
                if len(values) > 1:
                    aggregated[f"{key}_std"] = statistics.stdev(values)
                else:
                    aggregated[f"{key}_std"] = 0.0

        return aggregated
