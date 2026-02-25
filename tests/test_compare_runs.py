from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_testing.compare_runs import collect_scores

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_DEFAULT_CFG: dict[str, object] = {
    "models": {
        "llm": {"model": "gpt-4"},
        "embeddings": {"model": "text-embedding-3-small"},
    },
    "index": {"chunk_size": 1000, "chunk_overlap": 100},
    "eval": {
        "retriever_type": "dense",
        "reranker_type": "none",
        "top_k": 3,
        "qa_path": "data/queries/test.csv",
    },
}


def _write_config(run_dir: Path) -> None:
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(_DEFAULT_CFG), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# collect_scores — filtering
# ---------------------------------------------------------------------------


def test_collect_scores_empty_runs_dir(tmp_path: Path) -> None:
    assert collect_scores(tmp_path) == []


def test_collect_scores_skips_dirs_without_scores_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text("{}\n")

    assert collect_scores(tmp_path) == []


def test_collect_scores_skips_top_level_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("not a run dir")
    assert collect_scores(tmp_path) == []


# ---------------------------------------------------------------------------
# collect_scores — metric values
# ---------------------------------------------------------------------------


def test_collect_scores_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.85, "answer_relevancy": 0.9}),
        encoding="utf-8",
    )

    rows = collect_scores(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run"] == "20240101_120000_team_baseline"
    assert rows[0]["faithfulness"] == 0.85
    assert rows[0]["answer_relevancy"] == 0.9


def test_collect_scores_multiple_runs_sorted(tmp_path: Path) -> None:
    for name in ["20240102_team_baseline", "20240101_team_baseline"]:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(
            json.dumps({"faithfulness": 0.5}), encoding="utf-8"
        )

    rows = collect_scores(tmp_path)

    assert len(rows) == 2
    # sorted() puts 20240101 before 20240102
    assert rows[0]["run"] == "20240101_team_baseline"
    assert rows[1]["run"] == "20240102_team_baseline"


def test_collect_scores_mixed_runs_only_scored_included(tmp_path: Path) -> None:
    scored = tmp_path / "20240101_run_a"
    scored.mkdir()
    (scored / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.7}), encoding="utf-8"
    )

    unscored = tmp_path / "20240101_run_b"
    unscored.mkdir()
    # No metrics.json in this dir

    rows = collect_scores(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run"] == "20240101_run_a"


# ---------------------------------------------------------------------------
# collect_scores — config_used.yaml metadata
# ---------------------------------------------------------------------------


def test_collect_scores_reads_config_used_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.8}), encoding="utf-8"
    )
    _write_config(run_dir)

    rows = collect_scores(tmp_path)

    assert rows[0]["retriever"] == "dense"
    assert rows[0]["reranker"] == "none"
    assert rows[0]["llm"] == "gpt-4"
    assert rows[0]["embedding"] == "text-embedding-3-small"
    assert rows[0]["chunk"] == "1000/100"
    assert rows[0]["top_k"] == 3
    assert rows[0]["qa_file"] == "test"


def test_collect_scores_no_config_yaml_omits_meta_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.8}), encoding="utf-8"
    )
    # No config_used.yaml written

    rows = collect_scores(tmp_path)

    assert "retriever" not in rows[0]
    assert "reranker" not in rows[0]
    assert "llm" not in rows[0]
    assert "chunk" not in rows[0]


# ---------------------------------------------------------------------------
# collect_scores — n_samples
# ---------------------------------------------------------------------------


def test_collect_scores_n_samples_from_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.8}), encoding="utf-8"
    )
    (run_dir / "predictions.jsonl").write_text(
        '{"q":"a"}\n{"q":"b"}\n', encoding="utf-8"
    )

    rows = collect_scores(tmp_path)

    assert rows[0]["n_samples"] == 2


def test_collect_scores_no_predictions_omits_n_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.8}), encoding="utf-8"
    )

    rows = collect_scores(tmp_path)

    assert "n_samples" not in rows[0]


# ---------------------------------------------------------------------------
# collect_scores — n_metric columns removed
# ---------------------------------------------------------------------------


def test_collect_scores_no_n_metric_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101_120000_team_baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"faithfulness": 0.8}), encoding="utf-8"
    )
    (run_dir / "scores.jsonl").write_text('{"faithfulness": 0.8}\n', encoding="utf-8")

    rows = collect_scores(tmp_path)

    assert "n_faithfulness" not in rows[0]
