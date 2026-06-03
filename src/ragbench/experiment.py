"""Experiment orchestration — the top of the package.

:func:`run_single_experiment` runs one config end-to-end (load → validate → build
→ evaluate → save); :func:`run_matrix` runs many and returns their result dirs.
These are the importable, testable entry points the CLI scripts (``scripts/
run_experiment.py``, ``scripts/run_matrix.py``) wrap.

This module imports from ``pipeline`` and ``evaluation`` — it is the orchestration
layer that sits above both. Nothing in ``pipeline/`` imports it, so the dependency
rule (``pipeline`` never imports ``evaluation``) is preserved.
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path
from typing import Any

from ragbench.core.config import ExperimentConfig, validate_config
from ragbench.core.git import get_git_dirty, get_git_sha
from ragbench.core.registry import registry
from ragbench.evaluation.evaluator import Evaluator
from ragbench.evaluation.results import save_experiment
from ragbench.pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")


def configure_runtime() -> None:
    """Load ``.env`` and apply GPU perf settings. Idempotent; called by the CLIs.

    Imports ``torch``/``dotenv`` lazily so importing this module (e.g. in tests)
    stays cheap and free of heavyweight side effects.
    """
    import warnings

    from dotenv import load_dotenv
    import torch

    load_dotenv()
    # TF32 tensor cores: faster float32 matmul on Ampere+; precision loss is
    # negligible for embedding/reranking inference.
    torch.set_float32_matmul_precision("high")
    # Non-actionable inductor warning when the GPU lacks SMs for autotuning.
    warnings.filterwarnings("ignore", message="Not enough SMs to use max_autotune_gemm")


def capture_git_info() -> dict[str, Any]:
    """Snapshot repo state for reproducibility (captured once per run / matrix)."""
    return {"sha": get_git_sha(), "dirty": get_git_dirty()}


def run_single_experiment(
    config_path: str | Path,
    *,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    no_cache: bool = False,
    git_info: dict[str, Any] | None = None,
) -> Path:
    """Run one experiment end-to-end and return its result directory.

    Sequence: load config → validate (catches typos before any model loads) →
    build pipeline (index cached by fingerprint) → evaluate → save. ``git_info``
    may be supplied by :func:`run_matrix` so every experiment in a sweep shares
    one repo snapshot; otherwise it is captured here.
    """
    config = ExperimentConfig.from_yaml(config_path)
    validate_config(config, registry)
    logger.info("Running experiment: %s", config.name)

    if git_info is None:
        git_info = capture_git_info()

    rag = RAGPipeline.from_config(config, no_cache=no_cache)
    evaluator = Evaluator(config.evaluation)
    result = evaluator.evaluate(rag, experiment_name=config.name)
    exp_dir = save_experiment(config, result, output_dir, git_info=git_info)
    logger.info("Experiment '%s' complete. Results saved to %s", config.name, exp_dir)
    return exp_dir


def run_matrix(
    config_paths: list[str | Path],
    *,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    no_cache: bool = False,
    continue_on_error: bool = True,
) -> list[Path]:
    """Run several experiments in sequence; return the result dirs that succeeded.

    Captures one git snapshot for the whole sweep. With ``continue_on_error`` a
    failing config is logged and skipped so one bad run doesn't abort the matrix.
    Experiments sharing an indexing fingerprint reuse the cached index, so a
    sweep over query-side variables builds each index only once.
    """
    git_info = capture_git_info()
    result_dirs: list[Path] = []
    total = len(config_paths)
    for i, cfg in enumerate(config_paths, start=1):
        logger.info("[%d/%d] Running %s", i, total, cfg)
        try:
            exp_dir = run_single_experiment(
                cfg, output_dir=output_dir, no_cache=no_cache, git_info=git_info
            )
            result_dirs.append(exp_dir)
        except Exception:
            logger.exception("Experiment failed, skipping: %s", cfg)
            if not continue_on_error:
                raise
    logger.info(
        "Matrix complete: %d/%d experiments produced results", len(result_dirs), total
    )
    return result_dirs


def resolve_config_paths(patterns: list[str]) -> list[Path]:
    """Expand CLI args (files, directories, or globs) into a sorted config list.

    - A directory expands to its ``**/*.yaml`` (recursive).
    - A glob pattern expands via ``glob.glob`` (absolute or relative, ``**``).
    - A plain path is taken as-is.

    Duplicates are removed and the result is sorted for a stable run order.
    """
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.yaml")))
        elif any(ch in pat for ch in "*?["):
            # glob.glob handles absolute + relative patterns and ``**``.
            out.extend(sorted(Path(m) for m in _glob.glob(pat, recursive=True)))
        else:
            out.append(p)
    # De-dup while preserving the sorted-ish order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique
