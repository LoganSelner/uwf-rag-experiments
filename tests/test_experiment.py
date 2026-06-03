"""Tests for src/uwf_rag/experiment.py — single-run + matrix orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from uwf_rag.experiment import (
    resolve_config_paths,
    run_matrix,
    run_single_experiment,
)


class TestRunSingleExperiment:
    def test_runs_full_sequence_and_returns_dir(self) -> None:
        with (
            patch("uwf_rag.experiment.ExperimentConfig") as mock_cfg,
            patch("uwf_rag.experiment.validate_config") as mock_validate,
            patch("uwf_rag.experiment.RAGPipeline") as mock_rag,
            patch("uwf_rag.experiment.Evaluator") as mock_eval,
            patch(
                "uwf_rag.experiment.save_experiment",
                return_value=Path("results/exp"),
            ) as mock_save,
            patch(
                "uwf_rag.experiment.capture_git_info",
                return_value={"sha": "abc", "dirty": False},
            ),
        ):
            mock_cfg.from_yaml.return_value = MagicMock(name="config")

            out = run_single_experiment("configs/x.yaml")

            assert out == Path("results/exp")
            mock_cfg.from_yaml.assert_called_once_with("configs/x.yaml")
            mock_validate.assert_called_once()
            mock_rag.from_config.assert_called_once()
            mock_eval.return_value.evaluate.assert_called_once()
            mock_save.assert_called_once()

    def test_uses_supplied_git_info_without_recapturing(self) -> None:
        with (
            patch("uwf_rag.experiment.ExperimentConfig") as mock_cfg,
            patch("uwf_rag.experiment.validate_config"),
            patch("uwf_rag.experiment.RAGPipeline"),
            patch("uwf_rag.experiment.Evaluator"),
            patch("uwf_rag.experiment.save_experiment", return_value=Path("r")),
            patch("uwf_rag.experiment.capture_git_info") as mock_capture,
        ):
            mock_cfg.from_yaml.return_value = MagicMock()
            run_single_experiment("c.yaml", git_info={"sha": "x", "dirty": True})
            mock_capture.assert_not_called()


class TestRunMatrix:
    def test_continues_past_a_failing_config(self) -> None:
        def fake_run(cfg: str, **kwargs: object) -> Path:
            if "bad" in str(cfg):
                raise RuntimeError("boom")
            return Path(f"results/{cfg}")

        with (
            patch("uwf_rag.experiment.run_single_experiment", side_effect=fake_run),
            patch("uwf_rag.experiment.capture_git_info", return_value={}),
        ):
            dirs = run_matrix(["good1.yaml", "bad.yaml", "good2.yaml"])
            assert len(dirs) == 2

    def test_stop_on_error_reraises(self) -> None:
        with (
            patch(
                "uwf_rag.experiment.run_single_experiment",
                side_effect=RuntimeError("boom"),
            ),
            patch("uwf_rag.experiment.capture_git_info", return_value={}),
        ):
            with pytest.raises(RuntimeError):
                run_matrix(["a.yaml"], continue_on_error=False)

    def test_shares_one_git_snapshot_across_runs(self) -> None:
        with (
            patch(
                "uwf_rag.experiment.run_single_experiment", return_value=Path("r")
            ) as mock_run,
            patch(
                "uwf_rag.experiment.capture_git_info", return_value={"sha": "S"}
            ) as mock_capture,
        ):
            run_matrix(["a.yaml", "b.yaml"])
            mock_capture.assert_called_once()  # one snapshot for the whole sweep
            for call in mock_run.call_args_list:
                assert call.kwargs["git_info"] == {"sha": "S"}


class TestResolveConfigPaths:
    def test_directory_expands_recursively(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("x: 1")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.yaml").write_text("y: 2")
        out = resolve_config_paths([str(tmp_path)])
        assert sorted(p.name for p in out) == ["a.yaml", "b.yaml"]

    def test_plain_file_passed_through(self, tmp_path: Path) -> None:
        f = tmp_path / "c.yaml"
        f.write_text("z: 3")
        assert resolve_config_paths([str(f)]) == [Path(str(f))]

    def test_glob_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("x")
        (tmp_path / "b.yaml").write_text("y")
        out = resolve_config_paths([str(tmp_path / "*.yaml")])
        assert len(out) == 2

    def test_deduplicates(self, tmp_path: Path) -> None:
        f = tmp_path / "a.yaml"
        f.write_text("x")
        assert len(resolve_config_paths([str(f), str(f)])) == 1
