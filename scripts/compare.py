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
from collections import OrderedDict
import csv
import io
import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ragbench.evaluation.comparison import (
    DECOUPLED_QUADRANTS,
    DECOUPLING_ANSWER_METRIC,
    DECOUPLING_RETRIEVAL_METRIC,
    DEFAULT_METRICS,
    METRIC_SHORT_NAMES,
    compare_experiments,
    decoupling_pair,
    decoupling_report,
    diff_configs,
    format_comparison_table,
    load_per_sample_scores,
    resolve_metric_name,
    sort_rows,
)

# ---------------------------------------------------------------------------
# Rich terminal rendering (script-layer only — depends on Rich)
# ---------------------------------------------------------------------------


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


def _delta_cell(delta: float | None) -> Text:
    """Format a signed A→B delta: green for a gain, red for a regression."""
    if delta is None:
        return Text("---", style="dim")
    style = "green" if delta > 0 else "red" if delta < 0 else "dim"
    return Text(f"{delta:+.3f}", style=style)


def render_rich_table(
    rows: list[dict[str, Any]],
    metrics: list[str],
    console: Console | None = None,
) -> Table:
    """Build and print a Rich table from comparison data."""
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


def render_decoupling_table(
    result_dirs: list[str],
    retrieval_metric: str,
    answer_metric: str,
    run: int = 1,
    console: Console | None = None,
) -> None:
    """Render the retrieval/answer decoupling analysis (Phase E, item 12).

    One dir → a quadrant-count summary plus the off-diagonal samples (retrieval
    helped but the answer didn't, and vice-versa). Two+ dirs → a per-sample A→B
    delta table for the first two, most-decoupled first.
    """
    console = console or Console()
    ret_short = METRIC_SHORT_NAMES.get(retrieval_metric, retrieval_metric)
    ans_short = METRIC_SHORT_NAMES.get(answer_metric, answer_metric)

    if len(result_dirs) == 1:
        report = decoupling_report(
            result_dirs[0],
            run=run,
            retrieval_metric=retrieval_metric,
            answer_metric=answer_metric,
        )
        counts = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title=(
                f"Decoupling quadrants — {report['experiment']} "
                f"({ret_short} vs {ans_short}, run {run})"
            ),
        )
        counts.add_column("Quadrant", style="cyan")
        counts.add_column("Count", justify="right")
        for quadrant, n in report["counts"].items():
            highlight = quadrant in DECOUPLED_QUADRANTS and n
            label = Text(quadrant, style="yellow") if highlight else Text(quadrant)
            counts.add_row(label, str(n))
        console.print(counts)

        off = [s for s in report["samples"] if s["quadrant"] in DECOUPLED_QUADRANTS]
        if not off:
            console.print(
                "[green]No decoupled samples at the current thresholds.[/green]"
            )
            return
        detail = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title="Decoupled samples",
        )
        detail.add_column("ID", style="dim", no_wrap=True)
        detail.add_column("Query", max_width=50)
        detail.add_column(ret_short, justify="right")
        detail.add_column(ans_short, justify="right")
        detail.add_column("Quadrant", style="cyan")
        for s in off:
            detail.add_row(
                str(s["sample_id"]),
                str(s["query"])[:50],
                _metric_cell(s.get(retrieval_metric), None),
                _metric_cell(s.get(answer_metric), None),
                s["quadrant"],
            )
        console.print(detail)
        return

    rows = decoupling_pair(
        result_dirs[0],
        result_dirs[1],
        run=run,
        retrieval_metric=retrieval_metric,
        answer_metric=answer_metric,
    )
    if not rows:
        console.print("[yellow]No samples shared between the two experiments.[/yellow]")
        return
    exp_a, exp_b = Path(result_dirs[0]).name, Path(result_dirs[1]).name
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title=f"Decoupling {exp_a} → {exp_b}  ({ret_short} vs {ans_short}, run {run})",
    )
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Query", max_width=44)
    table.add_column(f"Δ {ret_short}", justify="right")
    table.add_column(f"Δ {ans_short}", justify="right")
    table.add_column("Decoupled", justify="center")
    for r in rows:
        flag = (
            Text("yes", style="bold yellow")
            if r["decoupled"]
            else Text("-", style="dim")
        )
        table.add_row(
            str(r["sample_id"]),
            str(r["query"])[:44],
            _delta_cell(r["delta_retrieval"]),
            _delta_cell(r["delta_answer"]),
            flag,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
        help="Which run to use for --per-sample / --decoupling (default: 1).",
    )
    parser.add_argument(
        "--decoupling",
        action="store_true",
        default=False,
        help="Retrieval/answer decoupling analysis: 1 dir → quadrant counts + "
        "off-diagonal samples; 2 dirs → A→B per-sample deltas.",
    )
    parser.add_argument(
        "--decoupling-metrics",
        nargs=2,
        metavar=("RETRIEVAL", "ANSWER"),
        default=None,
        help="Override the (retrieval, answer) metric pair for --decoupling "
        f"(default: {DECOUPLING_RETRIEVAL_METRIC} {DECOUPLING_ANSWER_METRIC}).",
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

    if args.decoupling:
        ret_m, ans_m = (
            (args.decoupling_metrics[0], args.decoupling_metrics[1])
            if args.decoupling_metrics
            else (DECOUPLING_RETRIEVAL_METRIC, DECOUPLING_ANSWER_METRIC)
        )
        render_decoupling_table(result_dirs, ret_m, ans_m, run=args.run)
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
