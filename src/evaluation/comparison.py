"""Cross-experiment comparison utility.

Loads multiple experiment summaries and produces comparison tables
matching the ACMSE paper format (Table 2). See ARCHITECTURE_PLAN.md
Section 9.6.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default columns for comparison tables
DEFAULT_METRICS = [
    "answer_correctness",
    "context_precision",
    "faithfulness",
    "context_entity_recall",
    "answer_relevancy",
]

# Short display names for table headers
METRIC_SHORT_NAMES = {
    "answer_correctness": "Ans.Corr",
    "context_precision": "Ctx.Prec",
    "faithfulness": "Faith.",
    "context_entity_recall": "CER",
    "answer_relevancy": "Ans.Rel",
}


def load_summary(result_dir: str | Path) -> dict[str, Any]:
    """Load a summary.json from an experiment result directory."""
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json found in {result_dir}")
    with open(path) as f:
        return json.load(f)


def compare_experiments(
    result_dirs: list[str | Path],
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and compare multiple experiment results.

    Args:
        result_dirs: Paths to experiment result directories.
        metrics: Which metrics to include (default: all standard).

    Returns:
        List of dicts, one per experiment, with name + metric values.
    """
    cols = metrics or DEFAULT_METRICS
    rows: list[dict[str, Any]] = []

    for result_dir in result_dirs:
        try:
            summary = load_summary(result_dir)
        except FileNotFoundError:
            logger.warning("Skipping %s: no summary.json", result_dir)
            continue

        row: dict[str, Any] = {
            "experiment": summary.get("experiment_name", str(result_dir)),
        }
        exp_metrics = summary.get("metrics", {})
        for metric in cols:
            value = exp_metrics.get(metric)
            std = exp_metrics.get(f"{metric}_std")
            row[metric] = value
            row[f"{metric}_std"] = std

        rows.append(row)

    return rows


def format_comparison_table(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> str:
    """Format comparison data as a markdown table.

    Matches the ACMSE paper Table 2 format:
    | Experiment | Ans.Corr | Ctx.Prec | Faith. | CER | Ans.Rel |
    """
    cols = metrics or DEFAULT_METRICS

    # Header
    headers = ["Experiment"]
    headers.extend(METRIC_SHORT_NAMES.get(m, m) for m in cols)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("-" * len(h) for h in headers) + " |"

    # Rows
    lines = [header_line, separator]
    for row in rows:
        cells = [row.get("experiment", "")]
        for metric in cols:
            value = row.get(metric)
            std = row.get(f"{metric}_std")
            if value is None:
                cells.append("-")
            elif std is not None and std > 0:
                cells.append(f"{value:.3f} ± {std:.3f}")
            else:
                cells.append(f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def print_comparison(
    result_dirs: list[str | Path],
    metrics: list[str] | None = None,
) -> None:
    """Load experiments and print a comparison table to stdout."""
    rows = compare_experiments(result_dirs, metrics)
    if not rows:
        print("No experiment results found.")
        return
    print(format_comparison_table(rows, metrics))
