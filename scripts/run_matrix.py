#!/usr/bin/env python3
"""Run a matrix of experiments from many YAML configs, then optionally compare.

The repo's whole point is controlled comparison, so this sweeps a set of configs
in one go. Experiments that share an indexing fingerprint reuse the cached index,
so a sweep over query-side variables builds each index only once.

Usage:
    # default: every config under configs/experiments/
    python scripts/run_matrix.py

    # an explicit set / a directory / a glob
    python scripts/run_matrix.py configs/experiments/chunking
    python scripts/run_matrix.py "configs/experiments/retrieval/*.yaml" --compare
    python scripts/run_matrix.py configs/base.yaml configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Trigger component registration at the application boundary
import uwf_rag.components  # noqa: F401
from uwf_rag.evaluation.comparison import compare_experiments, format_comparison_table
from uwf_rag.experiment import (
    DEFAULT_RESULTS_DIR,
    configure_runtime,
    resolve_config_paths,
    run_matrix,
)

logger = logging.getLogger(__name__)

_DEFAULT_PATTERN = "configs/experiments/**/*.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matrix of RAG experiments and optionally compare them."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            f"Config files, directories, or globs. Defaults to '{_DEFAULT_PATTERN}'."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force rebuild every index even if cache exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the matrix on the first failure (default: skip and continue).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the default results directory.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison table over the successful runs at the end.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to show in the comparison table (default: the standard set).",
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

    patterns = args.configs or [_DEFAULT_PATTERN]
    config_paths = resolve_config_paths(patterns)
    if not config_paths:
        parser.error(f"No configs matched: {patterns}")

    print(f"Matrix: {len(config_paths)} experiment(s)")
    for p in config_paths:
        print(f"  - {p}")

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    result_dirs = run_matrix(
        list(config_paths),
        output_dir=results_dir,
        no_cache=args.no_cache,
        continue_on_error=not args.stop_on_error,
    )

    print(f"\n{len(result_dirs)}/{len(config_paths)} experiment(s) succeeded.")

    if args.compare and result_dirs:
        rows = compare_experiments(result_dirs, metrics=args.metrics)
        print("\n" + format_comparison_table(rows, metrics=args.metrics))


if __name__ == "__main__":
    main()
