"""CLI: Rich table comparison of scored RAG runs.

Thin presentation layer over ``results``.  For notebook or programmatic
access, use ``results.collect_runs`` and ``results.runs_to_dataframe``
directly — they have no Rich or argparse dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import load_settings
from .results import collect_runs, runs_to_dataframe

# ---------------------------------------------------------------------------
# Display transforms
# ---------------------------------------------------------------------------

# Analysis column → CLI column (shorter names for terminal width).
_COLUMN_RENAMES = {
    # Meta columns
    "retriever_type": "retriever",
    "reranker_type": "reranker",
    "generator_type": "generator",
    "llm_model": "llm",
    "embedding_model": "embedding",
    # Metric columns
    "answer_correctness": "correct",
    "answer_relevancy": "relevancy",
    "answer_similarity": "similarity",
    "context_entity_recall": "ctx_entity",
    "context_precision": "ctx_prec",
    "context_recall": "ctx_recall",
    "factual_correctness": "factual",
    "faithfulness": "faith",
}

# Ordered meta columns for the CLI table.
# Only universally relevant columns are shown here.  Pipeline-variant
# knobs (reranker_model, mmr_lambda, mmr_fetch_k) are available in
# the analysis DataFrame via runs_to_dataframe() for notebook use.
_META_COLUMNS = (
    "retriever",
    "reranker",
    "generator",
    "k",
    "llm",
    "embedding",
    "chunk",
    "qa_file",
    "n_samples",
)

# Columns available in the analysis DataFrame but not useful in CLI output.
# Pipeline-variant knobs are excluded from the terminal table; they are
# accessible via runs_to_dataframe() for notebook analysis.
_DROP_COLUMNS = {
    "git_sha",
    "reranker_model",
    "mmr_lambda",
    "mmr_fetch_k",
    "hybrid_alpha",
}


def _drop_constant_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Drop meta columns whose values are identical across all rows.

    Metric columns are always kept.  When only one run is displayed,
    all columns are kept (there is nothing to compare).
    """
    if len(df) <= 1:
        return df
    meta_cols = [c for c in _META_COLUMNS if c in df.columns]
    constant = [col for col in meta_cols if df[col].nunique(dropna=False) <= 1]
    return df.drop(columns=constant)


def _prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Transform an analysis DataFrame into a CLI-friendly display DataFrame.

    Applies column renames, creates composite columns (``chunk``,
    ``qa_file``), drops noise columns, and reorders for readability.
    """
    df = df.copy()

    # Shorten column names for terminal width.
    df = df.rename(columns=_COLUMN_RENAMES)

    # Composite: retrieval_k + top_k → "20/3" (fetched/kept).
    if "retrieval_k" in df.columns and "top_k" in df.columns:
        df["k"] = (
            df["retrieval_k"].astype("Int64").astype(str).replace("<NA>", "")
            + "/"
            + df["top_k"].astype("Int64").astype(str).replace("<NA>", "")
        )
        df = df.drop(columns=["retrieval_k", "top_k"])

    # Composite: chunk_size + chunk_overlap → "1000/100".
    if "chunk_size" in df.columns and "chunk_overlap" in df.columns:
        df["chunk"] = (
            df["chunk_size"].astype("Int64").astype(str).replace("<NA>", "")
            + "/"
            + df["chunk_overlap"].astype("Int64").astype(str).replace("<NA>", "")
        )
        df = df.drop(columns=["chunk_size", "chunk_overlap"])

    # Stem the QA file path for brevity.
    if "qa_path" in df.columns:
        df["qa_file"] = df["qa_path"].apply(
            lambda p: Path(str(p)).stem if pd.notna(p) else ""
        )
        df = df.drop(columns=["qa_path"])

    # Drop columns not useful in terminal output.
    df = df.drop(columns=[c for c in _DROP_COLUMNS if c in df.columns])

    # Reorder: meta columns first, then metric columns alphabetically.
    meta_cols = [c for c in _META_COLUMNS if c in df.columns]
    metric_cols = sorted(c for c in df.columns if c not in meta_cols)
    return df[[*meta_cols, *metric_cols]]


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------


def _meta_cell(val: object) -> str:
    """Format a metadata value for terminal display."""
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
    """Format a metric value with colour-coded thresholds."""
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


def _render_table(df: pd.DataFrame) -> None:
    """Render *df* as a Rich table to the console."""
    meta_cols = [c for c in _META_COLUMNS if c in df.columns]
    metric_cols = sorted(c for c in df.columns if c not in meta_cols)

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


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


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
        help="Sort runs by this column (descending by default).",
    )
    parser.add_argument(
        "--asc",
        action="store_true",
        default=False,
        help="Sort ascending instead of descending (use with --sort-by).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="show_all",
        help="Show all meta columns, even when identical across runs.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        dest="output_format",
        help="Output format: rich table (default), JSON, or CSV.",
    )
    args = parser.parse_args()

    settings = load_settings()
    records = collect_runs(settings.eval.runs_dir)

    if not records:
        print(f"No scored runs found in {settings.eval.runs_dir}")
        return

    if args.last is not None:
        records = records[-args.last :]

    df = runs_to_dataframe(records)
    df = _prepare_display_df(df)

    if not args.show_all:
        df = _drop_constant_meta(df)

    if args.sort_by is not None:
        if args.sort_by not in df.columns:
            available = ", ".join(sorted(df.columns.tolist()))
            parser.error(f"Unknown column {args.sort_by!r}. Available: {available}")
        df = df.sort_values(args.sort_by, ascending=args.asc)

    if args.output_format == "json":
        print(df.to_json(orient="index", indent=2))
    elif args.output_format == "csv":
        print(df.to_csv())
    else:
        _render_table(df)


if __name__ == "__main__":
    main()
