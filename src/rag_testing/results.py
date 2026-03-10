"""Run result collection and DataFrame export.

Data layer for reading scored run artifacts.  ``RunResult`` preserves full
run data; ``collect_runs`` reads run directories; ``runs_to_dataframe``
flattens records into a pandas DataFrame for analysis.

This module has no presentation dependencies (no Rich, no argparse) so it
can be imported freely in notebooks, scripts, or other tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Complete record of a single scored run.

    ``config`` is the full ``config_used.yaml`` content (nested dict).
    ``metrics`` holds aggregate scores from ``metrics.json``.
    ``n_samples`` is the number of predictions (lines in ``predictions.jsonl``).
    """

    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    n_samples: int | None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> RunResult | None:
    """Load a single run directory into a ``RunResult``.

    Returns ``None`` when *run_dir* does not contain a ``metrics.json``
    file (i.e. the run has not been scored yet).
    """
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    metrics: dict[str, float] = json.loads(metrics_file.read_text(encoding="utf-8"))

    config: dict[str, Any] = {}
    config_file = run_dir / "config_used.yaml"
    if config_file.exists():
        config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    n_samples: int | None = None
    predictions_file = run_dir / "predictions.jsonl"
    if predictions_file.exists():
        n_samples = sum(
            1
            for line in predictions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    return RunResult(
        name=run_dir.name,
        config=config,
        metrics=metrics,
        n_samples=n_samples,
    )


def collect_runs(runs_dir: Path) -> list[RunResult]:
    """Collect all scored runs from *runs_dir*.

    Returns one ``RunResult`` per subdirectory that contains a
    ``metrics.json`` file, sorted by directory name (chronological).
    """
    results: list[RunResult] = []
    for entry in sorted(runs_dir.glob("*")):
        if not entry.is_dir():
            continue
        result = load_run(entry)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def runs_to_dataframe(runs: list[RunResult]) -> pd.DataFrame:
    """Flatten *runs* into an analysis DataFrame.

    Config values are extracted as individual columns with their original
    types (int, float, str).  Column names follow the config structure
    (e.g. ``retriever_type``, ``chunk_size``) rather than display-friendly
    abbreviations — the CLI layer handles renaming for terminal output.

    Returns a DataFrame indexed by run name.
    """
    if not runs:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for r in runs:
        row: dict[str, object] = {"run": r.name}

        if r.config:
            eval_cfg = r.config.get("eval", {})
            models_cfg = r.config.get("models", {})
            index_cfg = r.config.get("index", {})

            row["git_sha"] = r.config.get("git_sha")
            row["retriever_type"] = eval_cfg.get("retriever_type")
            row["reranker_type"] = eval_cfg.get("reranker_type")
            row["reranker_model"] = eval_cfg.get("reranker_model")
            row["retrieval_k"] = eval_cfg.get("retrieval_k")
            row["top_k"] = eval_cfg.get("top_k")
            row["mmr_lambda"] = eval_cfg.get("mmr_lambda")
            row["mmr_fetch_k"] = eval_cfg.get("mmr_fetch_k")
            row["llm_model"] = models_cfg.get("llm", {}).get("model")
            row["embedding_model"] = models_cfg.get("embeddings", {}).get("model")
            row["chunk_size"] = index_cfg.get("chunk_size")
            row["chunk_overlap"] = index_cfg.get("chunk_overlap")
            row["qa_path"] = eval_cfg.get("qa_path")

        row["n_samples"] = r.n_samples
        row.update(r.metrics)
        rows.append(row)

    return pd.DataFrame(rows).set_index("run")
