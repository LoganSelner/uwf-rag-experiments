"""Run comparison: score collection and Rich table rendering for scored RAG runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
import yaml

from .config import load_settings

_META = (
    "retriever",
    "reranker",
    "llm",
    "embedding",
    "chunk",
    "top_k",
    "qa_file",
    "n_samples",
)


def collect_scores(runs_dir: Path) -> list[dict[str, object]]:
    """Collect scored run metadata and metric values from ``runs_dir``."""
    rows: list[dict[str, object]] = []
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        metrics: dict[str, object] = json.loads(
            metrics_file.read_text(encoding="utf-8")
        )
        row: dict[str, object] = {"run": run_dir.name}

        config_file = run_dir / "config_used.yaml"
        if config_file.exists():
            cfg: dict[str, Any] = yaml.safe_load(
                config_file.read_text(encoding="utf-8")
            )
            row["retriever"] = cfg.get("eval", {}).get("retriever_type", "")
            row["reranker"] = cfg.get("eval", {}).get("reranker_type", "")
            row["llm"] = cfg.get("models", {}).get("llm", {}).get("model", "")
            row["embedding"] = (
                cfg.get("models", {}).get("embeddings", {}).get("model", "")
            )
            idx = cfg.get("index", {})
            row["chunk"] = f"{idx.get('chunk_size', '')}/{idx.get('chunk_overlap', '')}"
            row["top_k"] = int(cfg.get("eval", {}).get("top_k", 0))
            qa_path = cfg.get("eval", {}).get("qa_path", "")
            row["qa_file"] = Path(qa_path).stem if qa_path else ""

        predictions_file = run_dir / "predictions.jsonl"
        if predictions_file.exists():
            row["n_samples"] = sum(
                1
                for line in predictions_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        row.update(metrics)
        rows.append(row)
    return rows


def main() -> None:
    """CLI entrypoint for tabular comparison of scored runs."""
    parser = argparse.ArgumentParser(
        description="Compare scored RAG runs in a formatted table."
    )
    parser.add_argument(
        "--last",
        metavar="N",
        type=int,
        default=None,
        help="Show only the N most recent runs.",
    )
    parser.add_argument(
        "--sort-by",
        metavar="COLUMN",
        default=None,
        help="Sort runs by this column, descending.",
    )
    args = parser.parse_args()

    settings = load_settings()
    rows = collect_scores(settings.eval.runs_dir)

    if not rows:
        print("No scored runs found.")
        return

    if args.last is not None:
        rows = rows[-args.last :]

    df = pd.DataFrame(rows).set_index("run")

    meta_cols = [c for c in _META if c in df.columns]
    metric_cols = sorted(c for c in df.columns if c not in meta_cols)

    df = df[[*meta_cols, *metric_cols]]

    if args.sort_by is not None:
        if args.sort_by not in df.columns:
            available = ", ".join(sorted(df.columns.tolist()))
            parser.error(f"Unknown column {args.sort_by!r}. Available: {available}")
        df = df.sort_values(args.sort_by, ascending=False)

    def _meta_cell(val: object) -> str:
        if pd.isna(val):  # type: ignore[call-overload]
            return ""
        try:
            f = float(val)  # type: ignore[arg-type]
            if f == int(f):
                return str(int(f))
        except (TypeError, ValueError):
            pass
        return str(val)

    def _metric_cell(val: object) -> Text:
        if pd.isna(val):  # type: ignore[call-overload]
            return Text("—", style="dim")
        f = float(val)  # type: ignore[arg-type]
        if f >= 0.8:
            style = "bold green"
        elif f >= 0.5:
            style = "yellow"
        else:
            style = "red"
        return Text(f"{f:.4f}", style=style)

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("run", style="dim", no_wrap=True)
    for col in meta_cols:
        table.add_column(col, no_wrap=True)
    for col in metric_cols:
        table.add_column(col, justify="right", no_wrap=True)

    for run_name, row_data in df.iterrows():
        cells: list[str | Text] = [str(run_name)]
        for col in meta_cols:
            cells.append(_meta_cell(row_data[col]))
        for col in metric_cols:
            cells.append(_metric_cell(row_data[col]))
        table.add_row(*cells)

    Console().print(table)


if __name__ == "__main__":
    main()
