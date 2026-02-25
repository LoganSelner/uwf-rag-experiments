from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from rag_testing import run_pipeline
from rag_testing.config import Settings
from rag_testing.pipeline import PipelineResult
from rag_testing.run_pipeline import load_eval_rows, run_once

# ---------------------------------------------------------------------------
# load_eval_rows
# ---------------------------------------------------------------------------


def test_load_eval_rows_with_id_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text(
        "id,question,ground_truth\nq1,What is X?,X is Y\n", encoding="utf-8"
    )
    rows = load_eval_rows(csv_path)
    assert rows == [{"id": "q1", "question": "What is X?", "ground_truth": "X is Y"}]


def test_load_eval_rows_auto_assigns_id_when_column_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text(
        "question,ground_truth\nFirst?,Truth1\nSecond?,Truth2\n", encoding="utf-8"
    )
    rows = load_eval_rows(csv_path)
    assert rows[0]["id"] == "1"
    assert rows[1]["id"] == "2"


def test_load_eval_rows_skips_empty_question(tmp_path: Path) -> None:
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text(
        "id,question,ground_truth\nq1,,missing\nq2,Valid?,yes\n", encoding="utf-8"
    )
    rows = load_eval_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "q2"


def test_load_eval_rows_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text("id,question,ground_truth\n", encoding="utf-8")
    assert load_eval_rows(csv_path) == []


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


def _make_mock_pipeline(name: str = "dense_none_stuff") -> MagicMock:
    mock = MagicMock()
    mock.name = name
    mock.answer.return_value = PipelineResult(
        answer="Test answer", contexts=["chunk one", "chunk two"]
    )
    return mock


def test_run_once_predictions_jsonl_content(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_pipeline.RAGPipeline, "from_settings", lambda _: _make_mock_pipeline()
    )
    settings.eval.qa_path.write_text(
        "id,question,ground_truth\nq1,Test question?,Ground truth\n", encoding="utf-8"
    )

    run_dir = run_once(settings)

    lines = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    row = lines[0]
    assert row["id"] == "q1"
    assert row["question"] == "Test question?"
    assert row["answer"] == "Test answer"
    assert row["contexts"] == ["chunk one", "chunk two"]
    assert row["ground_truth"] == "Ground truth"


def test_run_once_config_used_yaml_has_string_paths(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_pipeline.RAGPipeline, "from_settings", lambda _: _make_mock_pipeline()
    )
    settings.eval.qa_path.write_text(
        "id,question,ground_truth\nq1,Q?,G\n", encoding="utf-8"
    )

    run_dir = run_once(settings)

    config_used = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    # Path objects would fail yaml.safe_dump; strings confirm _to_yaml_safe worked
    assert isinstance(config_used["index"]["source_dir"], str)
    assert isinstance(config_used["index"]["persist_dir"], str)
    assert isinstance(config_used["eval"]["runs_dir"], str)
    assert isinstance(config_used["eval"]["qa_path"], str)
    # git_sha must be present and non-empty (may be "unknown" outside a git repo)
    assert isinstance(config_used["git_sha"], str)
    assert config_used["git_sha"]


def test_run_once_run_dir_named_after_pipeline(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_pipeline.RAGPipeline,
        "from_settings",
        lambda _: _make_mock_pipeline("my_pipeline"),
    )
    settings.eval.qa_path.write_text(
        "id,question,ground_truth\nq1,Q?,G\n", encoding="utf-8"
    )

    run_dir = run_once(settings)
    assert run_dir.name.endswith("_my_pipeline")
    assert run_dir.is_dir()
