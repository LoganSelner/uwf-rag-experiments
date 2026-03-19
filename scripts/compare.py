#!/usr/bin/env python3
"""Compare multiple experiment results side by side.

Usage:
    python scripts/compare.py results/baseline results/chunking_1000
    python scripts/compare.py results/* --sort-by faithfulness
    python scripts/compare.py results/* --format csv --metrics answer_correctness
    python scripts/compare.py results/* --last 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Add src/ to path for bare imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.comparison import (
    DEFAULT_METRICS,
    METRIC_SHORT_NAMES,
    compare_experiments,
    diff_configs,
    format_comparison_table,
    load_per_sample_scores,
    resolve_metric_name,
    sort_rows,
)


def _metric_cell(value: float | None, std: float | None) -> Text:
    """Format a metric value as Rich Text with color-coded thresholds."""
    if value is None:
        return Text("---", style="dim")

    if value >= 0.8:
        style = "bold green"
    elif value >= 0.5:
        style = "yellow"
    else:
        style = "red"

    if std is not None and std > 0:
        text = f"{value:.3f} \u00b1 {std:.3f}"
    else:
        text = f"{value:.3f}"

    return Text(text, style=style)


def render_rich_table(
    rows: list[dict[str, Any]],
    metrics: list[str],
    console: Console | None = None,
) -> Table:
    """Build and print a Rich table from comparison data.

    Returns the Table object for testability.
    """
    console = console or Console()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title="Experiment Comparison",
    )

    table.add_column("Experiment", style="cyan", no_wrap=True)
    for metric in metrics:
        short = METRIC_SHORT_NAMES.get(metric, metric)
        table.add_column(short, justify="right", no_wrap=True)

    for row in rows:
        cells: list[str | Text] = [row.get("experiment", "")]
        for metric in metrics:
            value = row.get(metric)
            std = row.get(f"{metric}_std")
            cells.append(_metric_cell(value, std))
        table.add_row(*cells)

    console.print(table)
    return table


def _output_json(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    """Print comparison data as JSON."""
    output = []
    for row in rows:
        entry: dict[str, Any] = {"experiment": row.get("experiment", "")}
        for metric in metrics:
            entry[metric] = row.get(metric)
            std = row.get(f"{metric}_std")
            if std is not None:
                entry[f"{metric}_std"] = std
        output.append(entry)
    print(json.dumps(output, indent=2))


def _output_csv(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    """Print comparison data as CSV."""
    buf = io.StringIO()
    fieldnames = ["experiment"]
    for m in metrics:
        fieldnames.append(m)
        fieldnames.append(f"{m}_std")

    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})
    print(buf.getvalue(), end="")


def render_diff_table(
    result_dirs: list[str],
    console: Console | None = None,
) -> Table | None:
    """Render config differences between experiments as a Rich table."""
    console = console or Console()

    if len(result_dirs) < 2:
        console.print("[yellow]Need at least 2 experiments to diff configs.[/yellow]")
        return None

    # Diff first two experiments
    diffs = diff_configs(result_dirs[0], result_dirs[1])

    if not diffs:
        console.print("[green]Configs are identical.[/green]")
        return None

    exp_a = Path(result_dirs[0]).name
    exp_b = Path(result_dirs[1]).name

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title="Config Differences",
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column(exp_a, justify="left")
    table.add_column(exp_b, justify="left")

    for key, (val_a, val_b) in diffs.items():
        table.add_row(key, str(val_a), str(val_b))

    console.print(table)
    return table


def render_per_sample_table(
    result_dirs: list[str],
    metrics: list[str],
    run: int = 1,
    console: Console | None = None,
) -> Table | None:
    """Render per-sample scores across experiments as a Rich table."""
    console = console or Console()

    rows = load_per_sample_scores(result_dirs, run=run, metrics=metrics)
    if not rows:
        console.print("[yellow]No per-sample data found.[/yellow]")
        return None

    # Group by sample_id to build columns per experiment
    from collections import OrderedDict

    samples: OrderedDict[str, dict[str, Any]] = OrderedDict()
    experiments: list[str] = []
    for row in rows:
        exp = row["experiment"]
        if exp not in experiments:
            experiments.append(exp)
        sid = row["sample_id"]
        if sid not in samples:
            samples[sid] = {"query": row["query"]}
        for metric in metrics:
            samples[sid][f"{exp}_{metric}"] = row.get(metric)

    # Use only the first metric for a readable table
    display_metric = metrics[0] if metrics else DEFAULT_METRICS[0]
    short = METRIC_SHORT_NAMES.get(display_metric, display_metric)

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title=f"Per-Sample Comparison (run {run}, {short})",
    )
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Query", max_width=50)
    for exp in experiments:
        table.add_column(exp, justify="right", no_wrap=True)

    for sid, data in samples.items():
        cells: list[str | Text] = [sid, data.get("query", "")[:50]]
        for exp in experiments:
            value = data.get(f"{exp}_{display_metric}")
            cells.append(_metric_cell(value, None))
        table.add_row(*cells)

    console.print(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple experiment results side by side.",
    )
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
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json", "csv"),
        default="table",
        dest="output_format",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--sort-by",
        metavar="METRIC",
        default=None,
        help="Sort experiments by a metric column (descending by default). "
        "Accepts full name (faithfulness) or short name (Faith.).",
    )
    parser.add_argument(
        "--asc",
        action="store_true",
        default=False,
        help="Sort ascending instead of descending (use with --sort-by).",
    )
    parser.add_argument(
        "--last",
        metavar="N",
        type=int,
        default=None,
        help="Show only the N most recent experiment directories.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Show config differences between the first two experiments.",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        default=False,
        help="Show per-sample score breakdown across experiments.",
    )
    parser.add_argument(
        "--run",
        metavar="N",
        type=int,
        default=1,
        help="Which run to use for --per-sample (default: 1).",
    )
    args = parser.parse_args()

    result_dirs = args.results
    if args.last is not None:
        result_dirs = result_dirs[-args.last :]

    metrics = args.metrics or DEFAULT_METRICS

    # Special modes: --diff and --per-sample
    if args.diff:
        render_diff_table(result_dirs)
        return

    if args.per_sample:
        render_per_sample_table(result_dirs, metrics, run=args.run)
        return

    # Default: aggregate comparison table
    rows = compare_experiments(result_dirs, metrics)

    if not rows:
        print("No experiment results found.")
        return

    if args.sort_by is not None:
        sort_key = resolve_metric_name(args.sort_by, metrics)
        if sort_key is None:
            available = ", ".join(
                f"{m} ({METRIC_SHORT_NAMES.get(m, m)})" for m in metrics
            )
            parser.error(
                f"Unknown sort metric {args.sort_by!r}. Available: {available}"
            )
        rows = sort_rows(rows, sort_key, ascending=args.asc)

    if args.output_format == "table":
        render_rich_table(rows, metrics)
    elif args.output_format == "markdown":
        print(format_comparison_table(rows, metrics))
    elif args.output_format == "json":
        _output_json(rows, metrics)
    elif args.output_format == "csv":
        _output_csv(rows, metrics)


if __name__ == "__main__":
    main()
