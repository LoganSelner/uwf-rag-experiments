"""Tests for the CLI display transforms in compare_runs."""

from __future__ import annotations

import pandas as pd

from rag_testing.compare_runs import (
    _drop_constant_meta,
    _meta_cell,
    _metric_cell,
    _prepare_display_df,
    _resolve_column,
)
from rag_testing.results import RunResult, runs_to_dataframe

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_SAMPLE_CONFIG: dict[str, object] = {
    "git_sha": "abc1234",
    "models": {
        "llm": {"model": "gpt-4"},
        "embeddings": {"model": "text-embedding-3-small"},
    },
    "index": {"chunk_size": 1000, "chunk_overlap": 100},
    "eval": {
        "retriever_type": "dense",
        "reranker_type": "none",
        "reranker_model": "",
        "retrieval_k": 20,
        "top_k": 3,
        "mmr_lambda": 0.5,
        "mmr_fetch_k": 20,
        "generator_type": "stuff",
        "hybrid_alpha": 0.5,
        "qa_path": "data/queries/test.csv",
    },
}


def _make_display_df(
    config: dict[str, object] | None = None,
    n_samples: int | None = 5,
) -> pd.DataFrame:
    """Build an analysis DataFrame and transform it for display."""
    record = RunResult(
        name="run_a",
        config=config or _SAMPLE_CONFIG,
        metrics={"faithfulness": 0.8},
        n_samples=n_samples,
    )
    return _prepare_display_df(runs_to_dataframe([record]))


# ---------------------------------------------------------------------------
# _prepare_display_df
# ---------------------------------------------------------------------------


class TestPrepareDisplayDf:
    def test_renames_columns(self) -> None:
        df = _make_display_df()

        assert "retriever" in df.columns
        assert "reranker" in df.columns
        assert "generator" in df.columns
        assert "llm" in df.columns
        assert "embedding" in df.columns
        # Original analysis names should be gone.
        assert "retriever_type" not in df.columns
        assert "reranker_type" not in df.columns
        assert "generator_type" not in df.columns
        assert "llm_model" not in df.columns
        assert "embedding_model" not in df.columns

    def test_composite_k_column(self) -> None:
        df = _make_display_df()

        assert "k" in df.columns
        assert df.loc["run_a", "k"] == "20/3"
        assert "retrieval_k" not in df.columns
        assert "top_k" not in df.columns

    def test_composite_chunk_column(self) -> None:
        df = _make_display_df()

        assert "chunk" in df.columns
        assert df.loc["run_a", "chunk"] == "1000/100"
        assert "chunk_size" not in df.columns
        assert "chunk_overlap" not in df.columns

    def test_composite_columns_with_missing_config(self) -> None:
        records = [
            RunResult("full", _SAMPLE_CONFIG, {"f": 0.8}, 5),
            RunResult("empty", {}, {"f": 0.7}, None),
        ]
        df = _prepare_display_df(runs_to_dataframe(records))

        assert df.loc["full", "k"] == "20/3"
        assert df.loc["full", "chunk"] == "1000/100"
        # Missing config → empty string, not "/".
        assert df.loc["empty", "k"] == ""
        assert df.loc["empty", "chunk"] == ""

    def test_qa_file_stems_path(self) -> None:
        df = _make_display_df()

        assert "qa_file" in df.columns
        assert df.loc["run_a", "qa_file"] == "test"
        assert "qa_path" not in df.columns

    def test_drops_cli_noise_columns(self) -> None:
        df = _make_display_df()
        for col in (
            "git_sha",
            "reranker_model",
            "mmr_lambda",
            "mmr_fetch_k",
            "hybrid_alpha",
        ):
            assert col not in df.columns

    def test_meta_columns_before_metrics(self) -> None:
        df = _make_display_df()
        cols = list(df.columns)
        retriever_idx = cols.index("retriever")
        faith_idx = cols.index("faith")
        assert retriever_idx < faith_idx

    def test_empty_config_still_shows_metrics(self) -> None:
        df = _make_display_df(config={}, n_samples=None)

        assert "faith" in df.columns
        assert df.loc["run_a", "faith"] == 0.8

    def test_preserves_metric_values(self) -> None:
        df = _make_display_df()
        assert df.loc["run_a", "faith"] == 0.8


# ---------------------------------------------------------------------------
# _drop_constant_meta
# ---------------------------------------------------------------------------


def _config_with(**overrides: object) -> dict[str, object]:
    """Return a copy of _SAMPLE_CONFIG with eval-level overrides applied."""
    cfg: dict[str, object] = {
        "git_sha": "abc1234",
        "models": _SAMPLE_CONFIG["models"],
        "index": _SAMPLE_CONFIG["index"],
        "eval": {**_SAMPLE_CONFIG["eval"], **overrides},  # type: ignore[arg-type]
    }
    return cfg


class TestDropConstantMeta:
    def test_drops_constant_columns(self) -> None:
        records = [
            RunResult("run_a", _config_with(retriever_type="dense"), {"f": 0.7}, 5),
            RunResult("run_b", _config_with(retriever_type="mmr"), {"f": 0.9}, 5),
        ]
        df = _prepare_display_df(runs_to_dataframe(records))
        df = _drop_constant_meta(df)

        # retriever varies → kept.
        assert "retriever" in df.columns
        # reranker, llm, embedding, chunk, etc. are constant → dropped.
        assert "reranker" not in df.columns
        assert "llm" not in df.columns
        assert "chunk" not in df.columns
        # Metrics are always kept.
        assert "f" in df.columns

    def test_keeps_all_for_single_run(self) -> None:
        df = _make_display_df()
        before_cols = set(df.columns)
        df = _drop_constant_meta(df)
        assert set(df.columns) == before_cols

    def test_never_drops_metrics(self) -> None:
        records = [
            RunResult("run_a", _config_with(), {"f": 0.8}, 5),
            RunResult("run_b", _config_with(), {"f": 0.8}, 5),
        ]
        df = _prepare_display_df(runs_to_dataframe(records))
        df = _drop_constant_meta(df)

        # Even though metric value is constant, it's not a meta column.
        assert "f" in df.columns


# ---------------------------------------------------------------------------
# _meta_cell
# ---------------------------------------------------------------------------


class TestMetaCell:
    def test_float_with_integer_value_renders_as_int(self) -> None:
        assert _meta_cell(1000.0) == "1000"

    def test_string_passthrough(self) -> None:
        assert _meta_cell("dense") == "dense"

    def test_nan_renders_empty(self) -> None:
        assert _meta_cell(float("nan")) == ""


# ---------------------------------------------------------------------------
# _metric_cell
# ---------------------------------------------------------------------------


class TestMetricCell:
    def test_green_threshold(self) -> None:
        result = _metric_cell(0.8)
        assert str(result) == "0.8000"
        assert result.style == "bold green"

    def test_yellow_threshold(self) -> None:
        result = _metric_cell(0.5)
        assert result.style == "yellow"

    def test_red_threshold(self) -> None:
        result = _metric_cell(0.49)
        assert result.style == "red"

    def test_nan_renders_dash(self) -> None:
        result = _metric_cell(float("nan"))
        assert str(result) == "—"
        assert result.style == "dim"


# ---------------------------------------------------------------------------
# _resolve_column
# ---------------------------------------------------------------------------


class TestResolveColumn:
    def test_display_name_resolves(self) -> None:
        df = _make_display_df()
        assert _resolve_column("faith", df) == "faith"

    def test_canonical_name_resolves(self) -> None:
        df = _make_display_df()
        assert _resolve_column("faithfulness", df) == "faith"

    def test_unknown_name_returns_none(self) -> None:
        df = _make_display_df()
        assert _resolve_column("nonexistent", df) is None
