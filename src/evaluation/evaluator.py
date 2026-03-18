"""RAGAS evaluation integration.

Runs the pipeline against an evaluation dataset, computes RAGAS metrics,
and aggregates results across multiple runs for statistical significance.

See ARCHITECTURE_PLAN.md Section 9 for full specification.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import statistics
from typing import Any

from core.config import EvaluationConfig
from core.types import EvalSample, ExperimentResult, Queryable, ScoredSample

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


class _EmbedderAdapter:
    """Adapts a BaseEmbedder to the LangChain Embeddings interface.

    This allows reusing the pipeline's already-loaded embedding model
    for RAGAS evaluation without extra API calls or downloads.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class Evaluator:
    """Runs evaluation against a dataset using RAGAS metrics.

    Supports full mode (retrieve + generate + all metrics) and
    retrieval-only mode (retrieve + retrieval metrics only).
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config
        self._embedder_adapter: Any | None = None

    @property
    def active_metrics(self) -> list[str]:
        """Return the metrics to compute based on evaluation mode."""
        if self._config.mode == "retrieval_only":
            return list(self._config.retrieval_only_metrics)
        return list(self._config.metrics)

    def evaluate(
        self,
        pipeline: Queryable,
        *,
        embedder: Any = None,
        experiment_name: str = "",
    ) -> ExperimentResult:
        """Run the full evaluation: multi-run execution + aggregation.

        Args:
            pipeline: Anything satisfying the Queryable protocol.
            embedder: Optional embedder to reuse for RAGAS (avoids
                extra model loads). Pass the pipeline's embedder if
                available.
            experiment_name: Label for this experiment (used in results).

        Returns:
            ExperimentResult with aggregated and per-run metrics.
        """
        dataset = _load_dataset(self._config.dataset)
        num_runs = self._config.num_runs

        # Reuse the caller's embedder for RAGAS (avoids extra API
        # calls — the HuggingFace model is already loaded in memory).
        if embedder is not None:
            from ragas.embeddings.base import LangchainEmbeddingsWrapper

            self._embedder_adapter = LangchainEmbeddingsWrapper(
                _EmbedderAdapter(embedder)  # type: ignore[arg-type]
            )

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
            metrics, sample_scores = self._compute_metrics(samples)
            per_run_metrics.append(metrics)

            scored = [
                ScoredSample(
                    id=s.id,
                    query=s.query,
                    response=s.response,
                    retrieved_contexts=s.retrieved_contexts,
                    reference=s.reference,
                    scores=scores,
                )
                for s, scores in zip(samples, sample_scores, strict=True)
            ]
            per_run_samples.append(scored)

            logger.info("Run %d metrics: %s", run_idx, metrics)

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
        from ragas import EvaluationDataset, RunConfig, evaluate
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
            "run_config": RunConfig(timeout=300),
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

    def _build_evaluator_llm(self) -> Any:
        """Build a RAGAS-compatible LLM from config.

        Returns a LangchainLLMWrapper around a LangChain chat model.
        """
        from ragas.llms.base import LangchainLLMWrapper

        llm_config = self._config.evaluator_llm

        if llm_config.provider == "ollama":
            from langchain_ollama import OllamaLLM

            return LangchainLLMWrapper(
                OllamaLLM(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                )
            )

        if llm_config.provider == "edenai":
            import os

            from dotenv import load_dotenv
            from langchain_community.chat_models.edenai import (
                ChatEdenAI,
            )

            load_dotenv()
            api_key = os.environ.get("EDENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "EDENAI_API_KEY environment variable is required "
                    "for Eden AI evaluator LLM."
                )

            sub_provider = llm_config.params.get("sub_provider", "openai")
            return LangchainLLMWrapper(
                ChatEdenAI(
                    provider=sub_provider,
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens or 1024,
                    edenai_api_key=api_key,  # type: ignore[arg-type]
                )
            )

        raise ValueError(
            f"Unsupported evaluator LLM provider: "
            f"'{llm_config.provider}'. "
            f"Supported: 'ollama', 'edenai'."
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

        all_keys = per_run[0].keys()
        aggregated: dict[str, float] = {}

        for key in all_keys:
            values = [
                run[key] for run in per_run if key in run and not math.isnan(run[key])
            ]
            if values:
                aggregated[key] = statistics.mean(values)
                if len(values) > 1:
                    aggregated[f"{key}_std"] = statistics.stdev(values)
                else:
                    aggregated[f"{key}_std"] = 0.0

        return aggregated
