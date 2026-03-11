from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_testing.config import Settings

# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_reads_valid_lines(tmp_path: Path) -> None:
    from rag_testing.evaluate_ragas import _load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"a": 1}\n{"b": 2}\n',
        encoding="utf-8",
    )
    rows = _load_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    from rag_testing.evaluate_ragas import _load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text('{"x": 1}\n\n{"y": 2}\n\n', encoding="utf-8")
    rows = _load_jsonl(p)
    assert len(rows) == 2


def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    from rag_testing.evaluate_ragas import _load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text("", encoding="utf-8")
    assert _load_jsonl(p) == []


# ---------------------------------------------------------------------------
# evaluate_run — metric validation
# ---------------------------------------------------------------------------


def test_evaluate_run_all_unknown_metrics_raises(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":[],"ground_truth":"G"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())

    with pytest.raises(ValueError, match="No valid RAGAS metrics"):
        evaluate_ragas.evaluate_run(run_dir, ["nonexistent_metric"], settings)


# ---------------------------------------------------------------------------
# evaluate_run — happy path (ragas mocked)
# ---------------------------------------------------------------------------


def _patch_ragas(
    monkeypatch: pytest.MonkeyPatch, return_scores: list[dict]
) -> MagicMock:
    """Patch the ragas symbols that evaluate_run imports at call-time.

    return_scores is a list of per-row score dicts, matching the real
    EvaluationResult.scores API (one dict per prediction row).
    """
    import ragas
    import ragas.embeddings.base
    from ragas.evaluation import EvaluationResult
    import ragas.llms.base

    mock_result = MagicMock(spec=EvaluationResult)
    mock_result.scores = return_scores

    mock_evaluate = MagicMock(return_value=mock_result)
    monkeypatch.setattr(ragas, "evaluate", mock_evaluate)
    monkeypatch.setattr(ragas.llms.base, "LangchainLLMWrapper", lambda x: x)
    monkeypatch.setattr(
        ragas.embeddings.base, "LangchainEmbeddingsWrapper", lambda x: x
    )
    return mock_evaluate


def test_evaluate_run_writes_metrics_json(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":["ctx"],"ground_truth":"GT"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    _patch_ragas(monkeypatch, [{"faithfulness": 0.9}])

    scores = evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    assert scores == {"faithfulness": 0.9}
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == {"faithfulness": 0.9}


def test_evaluate_run_filters_non_numeric_values(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-numeric entries in result.scores rows are dropped from the output."""
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":[],"ground_truth":"G"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    _patch_ragas(monkeypatch, [{"faithfulness": 0.85, "label": "some_string"}])

    scores = evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    assert "label" not in scores
    assert scores["faithfulness"] == pytest.approx(0.85)


def test_evaluate_run_nan_rows_excluded_from_mean(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN values from failed LLM parses are excluded; mean is over valid rows only."""
    import math

    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q1?","answer":"A1","contexts":[],"ground_truth":"G1"}\n'
        '{"question":"Q2?","answer":"A2","contexts":[],"ground_truth":"G2"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    # Row 1 parsed, row 2 failed → NaN
    _patch_ragas(monkeypatch, [{"faithfulness": 0.8}, {"faithfulness": float("nan")}])

    scores = evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    assert "faithfulness" in scores
    assert not math.isnan(scores["faithfulness"])
    assert scores["faithfulness"] == pytest.approx(0.8)


def test_evaluate_run_all_nan_metric_omitted(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metric where every row is NaN is omitted from the output entirely."""
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":[],"ground_truth":"G"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    _patch_ragas(monkeypatch, [{"faithfulness": float("nan")}])

    scores = evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    assert "faithfulness" not in scores


def test_evaluate_run_passes_run_config_timeout(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RunConfig with the configured timeout is forwarded to ragas.evaluate."""
    from ragas import RunConfig

    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":[],"ground_truth":"G"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    mock_evaluate = _patch_ragas(monkeypatch, [{"faithfulness": 0.9}])

    evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    _, call_kwargs = mock_evaluate.call_args
    rc = call_kwargs["run_config"]
    assert isinstance(rc, RunConfig)
    assert rc.timeout == settings.eval.ragas_timeout


def test_evaluate_run_unknown_metrics_silently_dropped(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown metric names are dropped; only known ones reach ragas.evaluate."""
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"question":"Q?","answer":"A","contexts":[],"ground_truth":"G"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    mock_evaluate = _patch_ragas(monkeypatch, [{"faithfulness": 0.8}])

    evaluate_ragas.evaluate_run(run_dir, ["faithfulness", "totally_made_up"], settings)

    _, call_kwargs = mock_evaluate.call_args
    assert len(call_kwargs["metrics"]) == 1


# ---------------------------------------------------------------------------
# evaluate_run — per-sample JSONL output
# ---------------------------------------------------------------------------


def test_evaluate_run_writes_scores_jsonl(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scores.jsonl has one line per prediction with all metric scores.

    NaN values (parse failures for a row) are written as JSON null so the
    distinction between a genuine 0.0 and a failed parse is unambiguous.
    """
    import math

    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"id":"1","question":"Q1?","answer":"A1","contexts":[],"ground_truth":"G1"}\n'
        '{"id":"2","question":"Q2?","answer":"A2","contexts":[],"ground_truth":"G2"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    _patch_ragas(monkeypatch, [{"faithfulness": 0.9}, {"faithfulness": float("nan")}])

    evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    scores_path = run_dir / "scores.jsonl"
    assert scores_path.exists()

    lines = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2

    row1, row2 = lines
    assert row1["question"] == "Q1?"
    assert row1["answer"] == "A1"
    assert row1["ground_truth"] == "G1"
    assert row1["faithfulness"] == pytest.approx(0.9)

    assert row2["question"] == "Q2?"
    assert row2["faithfulness"] is None  # NaN → null
    assert not math.isnan(row2.get("faithfulness") or 0)  # null is not NaN


# ---------------------------------------------------------------------------
# evaluate_run — failed row filtering
# ---------------------------------------------------------------------------


def test_evaluate_run_skips_rows_with_null_answer(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows with answer=null (failed pipeline rows) are excluded from evaluation."""
    from rag_testing import evaluate_ragas

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        '{"id":"1","question":"Q1?","answer":"A1","contexts":["c"],"ground_truth":"G1"}\n'
        '{"id":"2","question":"Q2?","answer":null,"contexts":[],'
        '"ground_truth":"G2","error":"ConnectionError: transient"}\n'
        '{"id":"3","question":"Q3?","answer":"A3","contexts":["c"],"ground_truth":"G3"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_ragas, "build_llm", lambda _: MagicMock())
    monkeypatch.setattr(evaluate_ragas, "build_embeddings", lambda _: MagicMock())
    mock_evaluate = _patch_ragas(
        monkeypatch, [{"faithfulness": 0.8}, {"faithfulness": 0.9}]
    )

    scores = evaluate_ragas.evaluate_run(run_dir, ["faithfulness"], settings)

    # Only 2 rows (not 3) should reach ragas.evaluate
    _, call_kwargs = mock_evaluate.call_args
    dataset = call_kwargs["dataset"]
    assert len(dataset.samples) == 2

    assert scores["faithfulness"] == pytest.approx(0.85)
