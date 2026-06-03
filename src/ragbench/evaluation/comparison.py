"""Cross-experiment comparison utility.

Loads multiple experiment summaries and produces comparison tables
matching the ACMSE paper format (Table 2). See ROADMAP.md
Section 9.6.

Supports three tiers of comparison (following MLflow/W&B patterns):
1. Aggregate metrics — ``compare_experiments()``
2. Per-sample drill-down — ``load_per_sample_scores()``
3. Config diffs — ``diff_configs()``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default columns for comparison tables
DEFAULT_METRICS = [
    "answer_correctness",
    "context_precision",
    "faithfulness",
    "context_entity_recall",
    "answer_relevancy",
]

# Short display names for table headers (used by format_comparison_table
# and the compare.py CLI for both markdown and Rich output).
METRIC_SHORT_NAMES: dict[str, str] = {
    "answer_correctness": "Ans.Corr",
    "context_precision": "Ctx.Prec",
    "faithfulness": "Faith.",
    "context_entity_recall": "CER",
    "answer_relevancy": "Ans.Rel",
    # Opt-in metrics (not in DEFAULT_METRICS)
    "answer_similarity": "Ans.Sim",
    "context_recall": "Ctx.Rec",
    "factual_correctness": "Fact.Corr",
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


def sort_rows(
    rows: list[dict[str, Any]],
    metric: str,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Return *rows* sorted by *metric* (descending by default).

    Missing or ``None`` values sort to the bottom.
    """

    def _sort_key(r: dict[str, Any]) -> float:
        v = r.get(metric)
        return float(v) if v is not None else float("-inf")

    return sorted(rows, key=_sort_key, reverse=not ascending)


def format_comparison_table(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> str:
    """Format comparison data as a markdown table.

    Matches the ACMSE paper Table 2 format:
    ``| Experiment | Ans.Corr | Ctx.Prec | Faith. | CER | Ans.Rel |``
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


def resolve_metric_name(name: str, metrics: list[str]) -> str | None:
    """Resolve a metric name from its full or short display name.

    Accepts either the canonical name (``faithfulness``) or the short
    display name (``Faith.``), and returns the canonical name if it
    appears in *metrics*.  Returns ``None`` when unrecognised.
    """
    if name in metrics:
        return name
    reverse = {v.lower(): k for k, v in METRIC_SHORT_NAMES.items()}
    canonical = reverse.get(name.lower())
    if canonical and canonical in metrics:
        return canonical
    return None


# ---------------------------------------------------------------------------
# Config loading and diffing
# ---------------------------------------------------------------------------


def load_config(result_dir: str | Path) -> dict[str, Any]:
    """Load the full config from an experiment result directory.

    Reads ``config.yaml`` if present (new format). Falls back to the
    ``config_snapshot`` inside ``summary.json`` for older results that
    predate config.yaml saving.
    """
    config_path = Path(result_dir) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    # Fallback for old results without config.yaml
    summary = load_summary(result_dir)
    return summary.get("config_snapshot", {})


def _flatten_dict(
    d: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested dict to dotted key paths.

    Example: ``{"a": {"b": 1}}`` → ``{"a.b": 1}``
    Lists are kept as leaf values (not indexed).
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(_flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


def diff_configs(
    dir_a: str | Path,
    dir_b: str | Path,
) -> dict[str, tuple[Any, Any]]:
    """Return config keys that differ between two experiments.

    Returns:
        Dict mapping dotted key paths to ``(value_a, value_b)`` tuples.
        Only keys where the values differ are included.
    """
    flat_a = _flatten_dict(load_config(dir_a))
    flat_b = _flatten_dict(load_config(dir_b))

    all_keys = sorted(set(flat_a) | set(flat_b))
    diffs: dict[str, tuple[Any, Any]] = {}
    sentinel = object()

    for key in all_keys:
        val_a = flat_a.get(key, sentinel)
        val_b = flat_b.get(key, sentinel)
        if val_a != val_b:
            diffs[key] = (
                val_a if val_a is not sentinel else None,
                val_b if val_b is not sentinel else None,
            )

    return diffs


# ---------------------------------------------------------------------------
# Per-sample comparison
# ---------------------------------------------------------------------------


def load_per_sample_scores(
    result_dirs: list[str | Path],
    run: int = 1,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load per-sample scores from multiple experiments for a given run.

    Reads ``run_N.jsonl`` from each result directory and returns a flat
    list of dicts suitable for tabular display or DataFrame construction.

    Args:
        result_dirs: Paths to experiment result directories.
        run: Which run number to load (default: 1).
        metrics: Subset of metrics to include (default: all available).

    Returns:
        List of dicts with keys: experiment, sample_id, query,
        and one key per metric score.
    """
    cols = metrics or DEFAULT_METRICS
    rows: list[dict[str, Any]] = []

    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        run_path = result_dir / f"run_{run}.jsonl"
        if not run_path.exists():
            logger.warning("Skipping %s: no run_%d.jsonl", result_dir, run)
            continue

        # Get experiment name from summary if available
        try:
            summary = load_summary(result_dir)
            exp_name = summary.get("experiment_name", result_dir.name)
        except FileNotFoundError:
            exp_name = result_dir.name

        with open(run_path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping corrupted JSONL line %d in %s",
                        line_num,
                        run_path,
                    )
                    continue
                row: dict[str, Any] = {
                    "experiment": exp_name,
                    "sample_id": sample.get("id", ""),
                    "query": sample.get("input", {}).get("query", ""),
                }
                scores = sample.get("scores", {})
                for metric in cols:
                    row[metric] = scores.get(metric)
                rows.append(row)

    return rows
