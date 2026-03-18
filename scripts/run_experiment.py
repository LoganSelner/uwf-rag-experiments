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
import sys

# Add src/ to path for bare imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Trigger component registration at the application boundary
import components  # noqa: F401
from core.config import ExperimentConfig
from core.git import get_git_dirty, get_git_sha
from evaluation.evaluator import Evaluator
from evaluation.results import save_experiment
from pipeline.rag import RAGPipeline

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

    # Load config
    config = ExperimentConfig.from_yaml(args.config)
    logger.info("Running experiment: %s", config.name)

    # Capture git state early — before a long-running experiment
    # can change the working tree.
    git_info = {"sha": get_git_sha(), "dirty": get_git_dirty()}

    # Build pipeline
    rag = RAGPipeline.from_config(config, no_cache=args.no_cache)

    # Run evaluation — pass embedder explicitly so the evaluator can
    # reuse the already-loaded model for RAGAS without importing pipeline internals.
    evaluator = Evaluator(config.evaluation)
    experiment_result = evaluator.evaluate(
        rag,
        embedder=rag.index_artifact.embedder,
        experiment_name=config.name,
    )

    # Save results
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    exp_dir = save_experiment(config, experiment_result, results_dir, git_info=git_info)

    logger.info("Experiment '%s' complete. Results saved to %s", config.name, exp_dir)

    # Print summary
    print("\n=== Results ===")
    for name, value in sorted(experiment_result.metrics.items()):
        if not name.endswith("_std"):
            std = experiment_result.metrics.get(f"{name}_std", 0.0)
            print(f"  {name}: {value:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
