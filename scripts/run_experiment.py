#!/usr/bin/env python3
"""Run a single experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py configs/base.yaml
    python scripts/run_experiment.py configs/experiments/chunking_1000.yaml --no-cache
    python scripts/run_experiment.py configs/base.yaml --output-dir results/custom
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import warnings

from dotenv import load_dotenv
import torch

# Trigger component registration at the application boundary
import uwf_rag.components  # noqa: F401
from uwf_rag.core.config import ExperimentConfig, validate_config
from uwf_rag.core.git import get_git_dirty, get_git_sha
from uwf_rag.core.registry import registry
from uwf_rag.evaluation.evaluator import Evaluator
from uwf_rag.evaluation.results import save_experiment
from uwf_rag.pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RAG experiment from a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force rebuild index even if cache exists.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the default results directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Enable TF32 tensor cores for faster float32 matmul on Ampere+ GPUs.
    # Precision loss is negligible for embedding/reranking inference.
    torch.set_float32_matmul_precision("high")

    # Suppress non-actionable inductor warning about SM count — the GPU
    # simply doesn't have enough SMs for max_autotune_gemm benchmarking
    # and falls back to a heuristic automatically.
    warnings.filterwarnings("ignore", message="Not enough SMs to use max_autotune_gemm")

    # Load environment variables from .env before any component
    # needs them (API keys, etc.).
    load_dotenv()

    # Load and validate config — catches typos and constraint
    # violations before any heavyweight work (model loading, etc.).
    config = ExperimentConfig.from_yaml(args.config)
    validate_config(config, registry)
    logger.info("Running experiment: %s", config.name)

    # Capture git state early — before a long-running experiment
    # can change the working tree.
    git_info = {"sha": get_git_sha(), "dirty": get_git_dirty()}

    # Build pipeline
    rag = RAGPipeline.from_config(config, no_cache=args.no_cache)

    # Run evaluation — the evaluator builds its own embedder from
    # evaluator_embedding config for consistent cross-experiment measurement.
    evaluator = Evaluator(config.evaluation)
    experiment_result = evaluator.evaluate(
        rag,
        experiment_name=config.name,
    )

    # Save results
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    exp_dir = save_experiment(config, experiment_result, results_dir, git_info=git_info)

    logger.info("Experiment '%s' complete. Results saved to %s", config.name, exp_dir)

    # Print summary. Branch on the configured mode rather than on
    # `metrics` emptiness — empty metrics in full/retrieval_only runs
    # means scoring was attempted and produced nothing (all queries
    # failed, judge returned NaN, etc.), which is a failure to flag,
    # not a deliberate skip.
    print("\n=== Results ===")
    if config.evaluation.mode == "none":
        print(
            f"  Scoring skipped (evaluation.mode='none'). "
            f"Per-sample outputs saved to {exp_dir}."
        )
    elif not experiment_result.metrics:
        print(
            f"  No metrics produced (evaluation.mode="
            f"'{config.evaluation.mode}'). Scoring was attempted but "
            f"yielded no values — check logs for query or judge errors."
        )
    else:
        for name, value in sorted(experiment_result.metrics.items()):
            if not name.endswith("_std"):
                std = experiment_result.metrics.get(f"{name}_std", 0.0)
                print(f"  {name}: {value:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
