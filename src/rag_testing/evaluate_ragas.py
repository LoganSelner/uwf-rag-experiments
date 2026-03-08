"""RAGAS scoring for saved run artifacts."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings, load_settings
from .providers import build_embeddings, build_llm


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL lines into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Ragas imports are deferred inside _metric_map / evaluate_run because ragas has a
# significant startup cost (model initialisation, schema loading). Importing at module
# level would penalise every `import rag_testing` call even when evaluation is not
# being run. @cache ensures the metric objects are created exactly once per
# process, regardless of how many times evaluate_run is called.
@functools.cache
def _metric_map() -> dict[str, Any]:
    # RAGAS 0.4.x has two metric systems:
    #   - ragas.metrics.collections.*  — new API, requires InstructorLLM (llm_factory),
    #                                    incompatible with LangchainLLMWrapper / EdenAI.
    #   - ragas.metrics.*              — deprecated shim; fires DeprecationWarning on
    #                                    every import via __getattr__.
    # Importing directly from the private submodules (_faithfulness, etc.) is the
    # correct approach for the evaluate() + LangchainLLMWrapper pipeline: it uses
    # the legacy classes without triggering deprecation warnings.
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


def evaluate_run(
    run_dir: Path, metric_names: list[str], settings: Settings
) -> dict[str, float]:
    """Evaluate one run directory and write per-sample and aggregate scores."""
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import EvaluationResult, MultiTurnSample, SingleTurnSample
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper

    preds = _load_jsonl(run_dir / "predictions.jsonl")

    samples: list[SingleTurnSample | MultiTurnSample] = [
        SingleTurnSample(
            user_input=pred["question"],
            response=pred["answer"],
            retrieved_contexts=pred.get("contexts", []),
            reference=pred.get("ground_truth", ""),
        )
        for pred in preds
    ]

    metrics_lookup = _metric_map()
    metrics = [metrics_lookup[name] for name in metric_names if name in metrics_lookup]
    if not metrics:
        raise ValueError("No valid RAGAS metrics requested.")

    llm = LangchainLLMWrapper(build_llm(settings.models.llm))
    embeddings = LangchainEmbeddingsWrapper(
        build_embeddings(settings.models.embeddings)
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=settings.eval.ragas_timeout),
    )

    if not isinstance(result, EvaluationResult):  # pragma: no cover
        raise RuntimeError(f"Unexpected ragas result type: {type(result)}")

    # result.scores is list[dict[str, Any]] — one dict per sample, metric → value.
    # Build a DataFrame of just the numeric metric columns; non-numeric entries
    # (rare ragas internals) are dropped by select_dtypes. NaN values (LLM parse
    # failures for a specific row) are preserved and written as JSON null by
    # to_json(), making the difference between a genuine 0.0 and a failed parse
    # explicit in the output.
    scores_df = pd.DataFrame(result.scores)
    metric_cols = scores_df.select_dtypes(include="number").columns

    pred_meta = pd.DataFrame(
        [
            {
                "id": pred.get("id", ""),
                "question": pred["question"],
                "ground_truth": pred.get("ground_truth", ""),
                "answer": pred["answer"],
            }
            for pred in preds
        ]
    )

    per_sample_df = pd.concat([pred_meta, scores_df[metric_cols]], axis=1)
    (run_dir / "scores.jsonl").write_text(
        per_sample_df.to_json(orient="records", lines=True, force_ascii=False),
        encoding="utf-8",
    )

    # skipna=True excludes NaN rows from each metric's mean.
    # numeric_only=True drops any non-numeric columns that slipped through.
    # dropna() omits metrics where every row was NaN (no valid score at all).
    scores: dict[str, float] = {
        str(k): float(v)
        for k, v in scores_df.mean(skipna=True, numeric_only=True).dropna().items()
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(scores, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return scores


def main() -> None:
    """CLI entrypoint for evaluating a run with configured RAGAS metrics."""
    parser = argparse.ArgumentParser(
        description="Score a run directory with RAGAS metrics."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the run directory to evaluate. "
            "Defaults to the latest run for the configured method."
        ),
    )
    args = parser.parse_args()

    settings = load_settings()

    if args.run_dir is None:
        pipeline_name = (
            f"{settings.eval.retriever_type}"
            f"_{settings.eval.reranker_type}"
            f"_{settings.eval.generator_type}"
        )
        candidates = sorted(settings.eval.runs_dir.glob(f"*_{pipeline_name}"))
        if not candidates:
            raise FileNotFoundError(
                f"No run directories found for pipeline '{pipeline_name}' in "
                f"{settings.eval.runs_dir}. Run the pipeline first."
            )
        target = candidates[-1]
    else:
        target = Path(args.run_dir)
        if not target.is_dir():
            parser.error(f"Run directory does not exist: {target}")

    print(f"Evaluating run: {target}")
    scores = evaluate_run(target, settings.eval.metrics, settings)
    print(f"Saved per-sample scores to: {target / 'scores.jsonl'}")
    print(f"Saved aggregate metrics to: {target / 'metrics.json'}")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
