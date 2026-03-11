"""Pipeline runner: per-run orchestration and artifact writing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

import yaml

from .config import Settings, load_settings
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def load_eval_rows(path: Path) -> list[dict[str, str]]:
    """Load evaluation rows from CSV, normalising ids and skipping blank questions."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = (row.get("question") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            row_id = (row.get("id") or str(i)).strip()
            if not question:
                continue
            rows.append(
                {
                    "id": row_id,
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )
    return rows


def _get_git_sha() -> str:
    """Return the short current git SHA, or ``unknown`` when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _to_yaml_safe(obj: Any) -> Any:
    """Recursively convert Path objects to strings for yaml.safe_dump."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    return obj


def run_once(settings: Settings) -> Path:
    """Execute one full pipeline run and write artifacts to a new run directory."""
    pipeline = RAGPipeline.from_settings(settings)
    rows = load_eval_rows(settings.eval.qa_path)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = settings.eval.runs_dir / f"{timestamp}_{pipeline.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.jsonl"
    total = len(rows)

    failed = 0
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for i, row in enumerate(rows, start=1):
            logger.info("[%d/%d] %s", i, total, row["id"])
            record: dict[str, Any] = {
                "id": row["id"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
            }
            try:
                result = pipeline.answer(row["question"])
                record["answer"] = result.answer
                record["contexts"] = result.contexts
            except Exception:
                logger.exception("Row %s failed after retries", row["id"])
                failed += 1
                record["answer"] = None
                record["contexts"] = []
                record["error"] = "pipeline_error"
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()

    if failed:
        logger.warning(
            "Run complete: %d/%d succeeded (%d failed)",
            total - failed,
            total,
            failed,
        )

    config_doc = {"git_sha": _get_git_sha(), **_to_yaml_safe(asdict(settings))}
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config_doc, sort_keys=False),
        encoding="utf-8",
    )

    return run_dir


def main() -> None:
    """CLI entrypoint for generating run predictions."""
    parser = argparse.ArgumentParser(
        description="Run the configured RAG pipeline over the eval CSV."
    )
    parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = load_settings()
    run_dir = run_once(settings)
    print(f"Saved run artifacts to: {run_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
