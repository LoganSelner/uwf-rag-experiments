#!/usr/bin/env python3
"""Run a single experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py configs/base.yaml
    python scripts/run_experiment.py configs/experiments/retrieval/bm25.yaml --no-cache
    python scripts/run_experiment.py configs/base.yaml --output-dir results/custom
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Trigger component registration at the application boundary
import ragbench.components  # noqa: F401
from ragbench.evaluation.comparison import load_summary
from ragbench.experiment import (
    DEFAULT_RESULTS_DIR,
    configure_runtime,
    run_single_experiment,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RAG experiment from a YAML config file."
    )
    parser.add_argument("config", help="Path to the experiment YAML config file.")
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
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    configure_runtime()

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    exp_dir = run_single_experiment(
        args.config, output_dir=results_dir, no_cache=args.no_cache
    )

    _print_summary(exp_dir)


def _print_summary(exp_dir: Path) -> None:
    """Print a short results summary by reloading the saved summary.json.

    Branch on the configured mode rather than on `metrics` emptiness — empty
    metrics in a full/retrieval_only run means scoring was attempted and produced
    nothing (all queries failed, judge returned NaN, etc.), a failure to flag.
    """
    summary = load_summary(exp_dir)
    metrics: dict[str, float] = summary.get("metrics", {})
    mode = summary.get("config_snapshot", {}).get("evaluation_mode", "")

    print("\n=== Results ===")
    if mode == "none":
        print(
            f"  Scoring skipped (evaluation.mode='none'). "
            f"Per-sample outputs saved to {exp_dir}."
        )
    elif not metrics:
        print(
            f"  No metrics produced (evaluation.mode='{mode}'). Scoring was "
            f"attempted but yielded no values — check logs for query or judge errors."
        )
    else:
        for name, value in sorted(metrics.items()):
            if not name.endswith("_std"):
                std = metrics.get(f"{name}_std", 0.0)
                print(f"  {name}: {value:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
