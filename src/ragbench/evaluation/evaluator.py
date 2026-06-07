"""RAGAS evaluation integration.

Runs the pipeline against an evaluation dataset, computes RAGAS metrics,
and aggregates results across multiple runs for statistical significance.

See ROADMAP.md Section 9 for full specification.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from typing import Any

from ragbench.core.config import (
    SUPPORTED_EVALUATOR_LLM_PROVIDERS,
    EvaluationConfig,
    LLMConfig,
)
from ragbench.core.registry import registry
from ragbench.core.types import EvalSample, ExperimentResult, Queryable, ScoredSample
from ragbench.evaluation import _ragas_adapter

logger = logging.getLogger(__name__)


def _load_dataset(path: str) -> list[dict[str, Any]]:
    """Load an evaluation dataset from JSONL.

    Expected format: one {query, reference} JSON object per line.
    An optional ``id`` field provides a stable identifier for each
    item.  If absent, sequential 1-based IDs are assigned.
    """
    data: list[dict[str, Any]] = []
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


# Optional per-sample label columns a Phase E dataset may carry. Propagated from
# the dataset item into EvalSample.metadata so slice-aware metrics (abstention
# FRR/MRR, conflict rates) and the multi-turn loop can read them; absent keys are
# simply skipped. An allowlist keeps query/reference/id first-class and stops
# arbitrary dataset columns from bloating every JSONL row.
_LABEL_FIELDS: tuple[str, ...] = (
    "answerable",
    "abstention_type",
    "category",
    "source_name",
    "injected_context",
    "corpus_fact",
    "conflict_type",
    "expected_behavior",
    "conversation_id",
    "turn",
    "depends_on_prior",
)


def _mean_latency(samples: list[EvalSample]) -> float:
    """Mean per-query wall-clock latency (seconds) across a run's samples.

    Surfaces the cost side of the linear-vs-agent comparison (the D2 milestone
    asks whether agentic control flow improves quality *or just adds latency*).
    Folded into the per-run ``metrics`` dict so ``_aggregate_metrics`` carries it
    to ``summary.json`` as a mean ± std like any other metric — and it works in
    ``mode="none"`` too, where the rest of ``metrics`` is empty.
    """
    latencies = [
        s.metadata["latency_s"]
        for s in samples
        if isinstance(s.metadata.get("latency_s"), (int, float))
    ]
    return statistics.fmean(latencies) if latencies else 0.0


def _slice_rate(
    samples: list[EvalSample],
    signals: list[float | None],
    *,
    label_key: str,
    label_value: Any,
    positive_value: float = 1.0,
) -> float | None:
    """Fraction of the ``metadata[label_key] == label_value`` slice whose signal
    equals ``positive_value``.

    The deterministic-reduction primitive behind the Phase E slice metrics — it
    expresses False Refusal Rate (abstained over the answerable slice) and Missed
    Refusal Rate (did-not-abstain over the unanswerable slice) by
    parameterization. Returns ``None`` when the slice is empty or every in-slice
    signal is ``None`` (the judge gave no verdict), so an absent/unscored slice
    reads as ``-`` in the comparison table rather than a misleading ``0.0``.
    Injected into the per-run metrics dict the same way as :func:`_mean_latency`.
    """
    in_slice = [
        sig
        for s, sig in zip(samples, signals, strict=True)
        if s.metadata.get(label_key) == label_value and sig is not None
    ]
    if not in_slice:
        return None
    hits = sum(1 for sig in in_slice if sig == positive_value)
    return hits / len(in_slice)


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
            self._embedder_adapter = _ragas_adapter.wrap_embedder(embedder_instance)

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
            # Latency is not a RAGAS metric; inject it into the per-run metrics
            # so aggregation + summary.json + the comparison table pick it up for
            # free (works in mode="none", where `metrics` is otherwise empty).
            metrics["latency_mean_s"] = _mean_latency(samples)
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
        dataset: list[dict[str, Any]],
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
                t0 = time.perf_counter()
                result = pipeline.query(item["query"])
                latency_s = time.perf_counter() - t0
                contexts = [rc.chunk.content for rc in result.retrieved_chunks]
                # Carry any Phase E label columns through to the saved sample.
                # result.metadata first (provenance), then labels (ground truth
                # wins over a colliding pipeline key), then the authoritative
                # latency.
                labels = {k: item[k] for k in _LABEL_FIELDS if k in item}
                samples.append(
                    EvalSample(
                        id=item["id"],
                        query=item["query"],
                        response=result.answer,
                        retrieved_contexts=contexts,
                        reference=item["reference"],
                        metadata={
                            **result.metadata,
                            **labels,
                            "latency_s": latency_s,
                        },
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
        """Compute metrics on a list of EvalSamples via the ragas adapter.

        All ragas specifics (dataset/metric construction, ``evaluate()``) live
        in :mod:`ragbench.evaluation._ragas_adapter`; this method only filters the
        metric names, hands the judge LLM + embedder to the adapter, and
        aggregates the returned per-sample scores.

        Returns:
            A tuple of (aggregate_metrics, per_sample_scores).
            aggregate_metrics: mean per metric across all samples.
            per_sample_scores: one dict per sample with individual
            metric scores.  NaN values are converted to None.
        """
        active = [m for m in self.active_metrics if m in _ragas_adapter.KNOWN_METRICS]
        if not active:
            logger.warning("No valid RAGAS metrics to compute.")
            return {}, [{} for _ in samples]

        llm = None
        if (
            self._config.evaluator_llm.provider
            and self._config.evaluator_llm.model_name
        ):
            llm = self._build_evaluator_llm()

        # Per-sample score dicts, e.g. [{"faithfulness": 0.9, ...}].
        scores_list = _ragas_adapter.run_eval(
            samples,
            active,
            llm=llm,
            embeddings=self._embedder_adapter,
            run_config=self._make_run_config(),
        )

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
        """Build a ragas RunConfig from evaluation settings (via the adapter)."""
        cfg = self._config.run_config
        return _ragas_adapter.make_run_config(
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            max_wait=cfg.max_wait,
            max_workers=cfg.max_workers,
        )

    def _build_evaluator_llm(self) -> Any:
        """Build a ragas-compatible judge LLM from config.

        Dispatches on ``evaluator_llm.provider`` to a per-provider builder that
        returns a LangChain chat model (provider knowledge, not ragas), then
        hands it to the ragas adapter for the one ragas-side wrap.
        """
        llm_config = self._config.evaluator_llm
        builders = self._judge_builders()
        builder = builders.get(llm_config.provider)
        if builder is None:
            raise ValueError(
                f"Unsupported evaluator LLM provider: '{llm_config.provider}'. "
                f"Supported: {sorted(builders)}."
            )
        return _ragas_adapter.wrap_langchain_llm(builder(llm_config))

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
        import os

        from langchain_ollama import OllamaLLM

        kwargs: dict[str, Any] = {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature,
        }
        # Explicit config wins, then $OLLAMA_BASE_URL, else OllamaLLM's localhost
        # default — same precedence as OllamaGenerator (see components/generators).
        base_url = llm_config.params.get("base_url") or os.environ.get(
            "OLLAMA_BASE_URL"
        )
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
