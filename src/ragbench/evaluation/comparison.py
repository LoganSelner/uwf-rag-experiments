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

from collections import Counter
import json
import logging
import math
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
    # Cost side of the agentic comparison (appended so the established quality
    # columns keep their order). Present in every run, including mode="none".
    "latency_mean_s",
]

# Short display names for table headers (used by format_comparison_table
# and the compare.py CLI for both markdown and Rich output).
METRIC_SHORT_NAMES: dict[str, str] = {
    "answer_correctness": "Ans.Corr",
    "context_precision": "Ctx.Prec",
    "faithfulness": "Faith.",
    "context_entity_recall": "CER",
    "answer_relevancy": "Ans.Rel",
    "latency_mean_s": "Lat(s)",
    # Opt-in metrics (not in DEFAULT_METRICS)
    "answer_similarity": "Ans.Sim",
    "context_recall": "Ctx.Rec",
    "factual_correctness": "Fact.Corr",
    # Phase E slice metrics (only meaningful on the Phase E protocol datasets;
    # surface via --metrics or the dedicated reports, not DEFAULT_METRICS).
    "false_refusal_rate": "FRR",
    "missed_refusal_rate": "MRR",
    "error_detection_rate": "ErrDet",
    "corpus_preference_rate": "CorpPref",
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


# ---------------------------------------------------------------------------
# Retrieval/answer decoupling analysis (Phase E, item 12)
# ---------------------------------------------------------------------------

# The canonical (retrieval-quality, answer-quality) metric pair the decoupling
# report contrasts. Module constants so the CLI and tests share one definition.
DECOUPLING_RETRIEVAL_METRIC = "context_precision"
DECOUPLING_ANSWER_METRIC = "answer_correctness"

# All quadrant labels classify_decoupling can return, in display order. The two
# off-diagonal labels are the finding the report exists to surface.
DECOUPLING_QUADRANTS = (
    "both_good",
    "both_poor",
    "retrieval_good_answer_poor",
    "retrieval_poor_answer_good",
    "mixed",
    "incomplete",
)

# The decoupled quadrants — better retrieval that did not help the answer, or a
# correct answer despite weak retrieval (a faithfulness/parametric-knowledge
# risk). These are the rows a reader should inspect.
DECOUPLED_QUADRANTS = (
    "retrieval_good_answer_poor",
    "retrieval_poor_answer_good",
)


def _is_missing(value: float | None) -> bool:
    """True if a score is absent (None) or NaN (the judge failed to parse)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def classify_decoupling(
    retrieval_score: float | None,
    answer_score: float | None,
    *,
    hi: float = 0.7,
    lo: float = 0.4,
) -> str:
    """Bucket one sample into a retrieval/answer decoupling quadrant.

    Returns one of :data:`DECOUPLING_QUADRANTS`. A score is "good" at ``>= hi``
    and "poor" at ``<= lo``; the ``(lo, hi)`` dead-band is deliberate so a
    borderline sample lands in ``"mixed"`` rather than masquerading as decoupled.
    ``None``/``NaN`` on either side yields ``"incomplete"``.

    The two off-diagonal results are the headline:
    ``retrieval_good_answer_poor`` (retrieval succeeded but the generator failed
    to use it) and ``retrieval_poor_answer_good`` (answered correctly despite
    weak retrieval — parametric knowledge or lucky generation, a faithfulness
    risk).
    """
    if _is_missing(retrieval_score) or _is_missing(answer_score):
        return "incomplete"
    # Narrow Optional for the type checker — _is_missing already excluded None.
    assert retrieval_score is not None and answer_score is not None
    r_good, r_poor = retrieval_score >= hi, retrieval_score <= lo
    a_good, a_poor = answer_score >= hi, answer_score <= lo
    if r_good and a_good:
        return "both_good"
    if r_poor and a_poor:
        return "both_poor"
    if r_good and a_poor:
        return "retrieval_good_answer_poor"
    if r_poor and a_good:
        return "retrieval_poor_answer_good"
    return "mixed"


def decoupling_report(
    result_dir: str | Path,
    *,
    run: int = 1,
    retrieval_metric: str = DECOUPLING_RETRIEVAL_METRIC,
    answer_metric: str = DECOUPLING_ANSWER_METRIC,
    hi: float = 0.7,
    lo: float = 0.4,
) -> dict[str, Any]:
    """Single-experiment retrieval/answer decoupling analysis.

    Reads ``run_N.jsonl`` via :func:`load_per_sample_scores` (no re-parsing),
    classifies every sample with :func:`classify_decoupling`, and returns a dict
    with ``experiment``, ``run``, the two metric names, ``counts`` (every
    quadrant → n, zeros included for stable output), and ``samples`` (one row per
    question carrying its two scores + ``quadrant``).

    No correlation coefficient is computed: Phase E slices are 6-50 samples,
    where a Pearson/Spearman r would be noise. Quadrant counts plus the named
    off-diagonal samples are the honest artifact at this n.
    """
    rows = load_per_sample_scores(
        [result_dir], run=run, metrics=[retrieval_metric, answer_metric]
    )
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for r in rows:
        quadrant = classify_decoupling(
            r.get(retrieval_metric), r.get(answer_metric), hi=hi, lo=lo
        )
        counts[quadrant] += 1
        samples.append(
            {
                "sample_id": r.get("sample_id", ""),
                "query": r.get("query", ""),
                retrieval_metric: r.get(retrieval_metric),
                answer_metric: r.get(answer_metric),
                "quadrant": quadrant,
            }
        )
    experiment = rows[0]["experiment"] if rows else Path(result_dir).name
    return {
        "experiment": experiment,
        "run": run,
        "retrieval_metric": retrieval_metric,
        "answer_metric": answer_metric,
        "counts": {q: counts.get(q, 0) for q in DECOUPLING_QUADRANTS},
        "samples": samples,
    }


def _delta(b: float | None, a: float | None) -> float | None:
    """``b - a``, or ``None`` if either side is missing/NaN."""
    if _is_missing(a) or _is_missing(b):
        return None
    return float(b) - float(a)  # type: ignore[arg-type]


def decoupling_pair(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    run: int = 1,
    retrieval_metric: str = DECOUPLING_RETRIEVAL_METRIC,
    answer_metric: str = DECOUPLING_ANSWER_METRIC,
) -> list[dict[str, Any]]:
    """Cross-experiment decoupling: where did A→B move retrieval and the answer
    in OPPOSITE directions?

    Joins the two experiments' per-sample scores by ``sample_id`` and computes
    ``delta_retrieval`` / ``delta_answer`` (B minus A). A sample is ``decoupled``
    when retrieval improved but the answer did not, or retrieval regressed but
    the answer did not — the roadmap's exact framing ("better retrieval did NOT
    improve answers, and vice-versa"). Rows are sorted most-decoupled first.
    Samples present in only one experiment are dropped.
    """
    a_rows = load_per_sample_scores(
        [dir_a], run=run, metrics=[retrieval_metric, answer_metric]
    )
    b_by_id = {
        r["sample_id"]: r
        for r in load_per_sample_scores(
            [dir_b], run=run, metrics=[retrieval_metric, answer_metric]
        )
    }
    out: list[dict[str, Any]] = []
    for ra in a_rows:
        rb = b_by_id.get(ra["sample_id"])
        if rb is None:
            continue
        d_ret = _delta(rb.get(retrieval_metric), ra.get(retrieval_metric))
        d_ans = _delta(rb.get(answer_metric), ra.get(answer_metric))
        decoupled = (
            d_ret is not None
            and d_ans is not None
            and ((d_ret > 0 and d_ans <= 0) or (d_ret < 0 and d_ans >= 0))
        )
        out.append(
            {
                "sample_id": ra["sample_id"],
                "query": ra.get("query", ""),
                f"{retrieval_metric}_a": ra.get(retrieval_metric),
                f"{retrieval_metric}_b": rb.get(retrieval_metric),
                f"{answer_metric}_a": ra.get(answer_metric),
                f"{answer_metric}_b": rb.get(answer_metric),
                "delta_retrieval": d_ret,
                "delta_answer": d_ans,
                "decoupled": decoupled,
            }
        )

    def _magnitude(row: dict[str, Any]) -> float:
        # Decoupled rows first, ordered by how far the two metrics diverged;
        # coupled rows sink to the bottom.
        if not row["decoupled"]:
            return -1.0
        return abs(row["delta_retrieval"] or 0.0) + abs(row["delta_answer"] or 0.0)

    out.sort(key=_magnitude, reverse=True)
    return out
