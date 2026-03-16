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

from pipeline.rag import run_experiment


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

    metrics = run_experiment(
        config_path=args.config,
        no_cache=args.no_cache,
        output_dir=args.output_dir,
    )

    print("\n=== Results ===")
    for name, value in sorted(metrics.items()):
        if not name.endswith("_std"):
            std = metrics.get(f"{name}_std", 0.0)
            print(f"  {name}: {value:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
