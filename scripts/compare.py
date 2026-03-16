#!/usr/bin/env python3
"""Compare multiple experiment results side by side.

Usage:
    python scripts/compare.py results/baseline results/chunking_1000
    python scripts/compare.py results/baseline results/reranked \
        --metrics answer_correctness faithfulness
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src/ to path for bare imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.comparison import print_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple experiment results.")
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths to experiment result directories.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Specific metrics to compare (default: all standard).",
    )
    args = parser.parse_args()

    print_comparison(args.results, args.metrics)


if __name__ == "__main__":
    main()
