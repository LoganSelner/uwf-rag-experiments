"""RAGAS evaluation integration.

Runs the pipeline against an evaluation dataset, computes RAGAS metrics,
and aggregates results across multiple runs for statistical significance.

See ARCHITECTURE_PLAN.md Section 9 for full specification.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING, Any

from datasets import Dataset

from core.config import EvaluationConfig
from core.types import EvalSample, ExperimentResult

if TYPE_CHECKING:
    from pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

# Map config metric names to RAGAS metric objects.
# Lazy import to avoid import-time RAGAS initialization.
_METRIC_REGISTRY: dict[str, str] = {
    "answer_correctness": "answer_correctness",
    "context_precision": "context_precision",
    "faithfulness": "faithfulness",
    "context_entity_recall": "context_entity_recall",
    "answer_relevancy": "answer_relevancy",
}


def _load_dataset(path: str) -> list[dict[str, str]]:
    """Load an evaluation dataset from JSON.

    Expected format: list of {query, reference} dicts.
    """
    with open(path) as f:
        data: list[dict[str, str]] = json.load(f)
    if not data:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    for i, item in enumerate(data):
        if "query" not in item or "reference" not in item:
            raise ValueError(f"Dataset item {i} missing 'query' or 'reference': {item}")
    return data


class Evaluator:
    """Runs evaluation against a dataset using RAGAS metrics.

    Supports full mode (retrieve + generate + all metrics) and
    retrieval-only mode (retrieve + retrieval metrics only).
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    @property
    def active_metrics(self) -> list[str]:
        """Return the metrics to compute based on evaluation mode."""
        if self._config.mode == "retrieval_only":
            return list(self._config.retrieval_only_metrics)
        return list(self._config.metrics)

    def evaluate(self, rag: RAGPipeline) -> ExperimentResult:
        """Run the full evaluation: multi-run execution + aggregation.

        Args:
            rag: A fully constructed RAGPipeline.

        Returns:
            ExperimentResult with aggregated and per-run metrics.
        """
        dataset = _load_dataset(self._config.dataset)
        num_runs = self._config.num_runs

        logger.info(
            "Starting evaluation: %d questions, %d runs, mode=%s",
            len(dataset),
            num_runs,
            self._config.mode,
        )

        per_run_metrics: list[dict[str, float]] = []

        for run_idx in range(1, num_runs + 1):
            logger.info("Run %d/%d", run_idx, num_runs)
            samples = self._run_once(rag, dataset)
            metrics = self._compute_metrics(samples)
            per_run_metrics.append(metrics)
            logger.info("Run %d metrics: %s", run_idx, metrics)

        # Aggregate across runs
        aggregated = self._aggregate_metrics(per_run_metrics)
        logger.info("Aggregated metrics: %s", aggregated)

        return ExperimentResult(
            experiment_name=rag._config.name,
            metrics=aggregated,
            per_run_metrics=per_run_metrics,
            num_runs=num_runs,
            config_snapshot={
                "name": rag._config.name,
                "description": rag._config.description,
                "pipeline_mode": rag._config.pipeline_mode,
                "evaluation_mode": self._config.mode,
                "index_fingerprint": rag._config.index_fingerprint(),
            },
        )

    def _run_once(
        self,
        rag: RAGPipeline,
        dataset: list[dict[str, str]],
    ) -> list[EvalSample]:
        """Run the pipeline on each dataset query, producing EvalSamples."""
        samples: list[EvalSample] = []
        for item in dataset:
            result = rag.query(item["query"])
            contexts = [rc.chunk.content for rc in result.retrieved_chunks]
            samples.append(
                EvalSample(
                    query=item["query"],
                    response=result.answer,
                    retrieved_contexts=contexts,
                    reference=item["reference"],
                )
            )
        return samples

    def _compute_metrics(self, samples: list[EvalSample]) -> dict[str, float]:
        """Compute RAGAS metrics on a list of EvalSamples.

        Steps (from ARCHITECTURE_PLAN.md §9.5):
        1. Build metric objects from active_metrics
        2. Construct a HuggingFace Dataset from EvalSamples
        3. Call ragas.evaluate(dataset, metrics)
        4. Return the scores dict
        """
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_entity_recall,
            context_precision,
            faithfulness,
        )

        metric_map: dict[str, Any] = {
            "answer_correctness": answer_correctness,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "context_entity_recall": context_entity_recall,
            "answer_relevancy": answer_relevancy,
        }

        active = self.active_metrics
        ragas_metrics = [metric_map[name] for name in active if name in metric_map]

        if not ragas_metrics:
            logger.warning("No valid RAGAS metrics to compute.")
            return {}

        # Build HuggingFace Dataset in RAGAS expected format
        hf_dataset = Dataset.from_dict(
            {
                "question": [s.query for s in samples],
                "answer": [s.response for s in samples],
                "contexts": [s.retrieved_contexts for s in samples],
                "ground_truth": [s.reference for s in samples],
            }
        )

        # Configure evaluator LLM if specified
        eval_kwargs: dict[str, Any] = {}
        if (
            self._config.evaluator_llm.provider
            and self._config.evaluator_llm.model_name
        ):
            eval_kwargs["llm"] = self._build_evaluator_llm()

        ragas_result: Any = evaluate(
            dataset=hf_dataset,
            metrics=ragas_metrics,
            **eval_kwargs,
        )

        return {
            name: float(ragas_result[name]) for name in active if name in ragas_result
        }

    def _build_evaluator_llm(self) -> Any:
        """Build an LLM wrapper for RAGAS evaluation.

        RAGAS expects a LangChain-compatible LLM. This method
        constructs one from the evaluator_llm config.
        """
        llm_config = self._config.evaluator_llm
        if llm_config.provider == "ollama":
            from langchain_community.llms import Ollama

            return Ollama(
                model=llm_config.model_name,
                temperature=llm_config.temperature,
            )
        raise ValueError(
            f"Unsupported evaluator LLM provider: "
            f"'{llm_config.provider}'. "
            f"Supported: 'ollama'."
        )

    @staticmethod
    def _aggregate_metrics(
        per_run: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics across runs as mean ± std.

        Returns dict with keys like "answer_correctness" (mean)
        and "answer_correctness_std" (standard deviation).
        """
        if not per_run:
            return {}

        all_keys = per_run[0].keys()
        aggregated: dict[str, float] = {}

        for key in all_keys:
            values = [run[key] for run in per_run if key in run]
            if values:
                aggregated[key] = statistics.mean(values)
                if len(values) > 1:
                    aggregated[f"{key}_std"] = statistics.stdev(values)
                else:
                    aggregated[f"{key}_std"] = 0.0

        return aggregated
