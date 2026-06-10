"""Experiment results writer.

Serializes an ``ExperimentResult`` and its associated config to disk.
The output directory is treated as an atomic unit: it is cleared and
rewritten on each save to prevent stale files from previous runs.

Output layout::

    results/<experiment_name>/
        summary.json    — aggregated metrics + config snapshot + metadata
        config.yaml     — full resolved config for reproducibility
        run_1.jsonl     — per-sample scored data for run 1
        run_2.jsonl
        ...
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import yaml

from ragbench.core.config import ExperimentConfig
from ragbench.core.types import ExperimentResult

logger = logging.getLogger(__name__)


def _build_config_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    """Build a curated config subset for summary.json.

    This captures the key experimental variables at a glance,
    without duplicating the full config (which lives in config.yaml).
    """
    return {
        "name": config.name,
        "description": config.description,
        "pipeline_mode": config.pipeline_mode,
        "evaluation_mode": config.evaluation.mode,
        "index_fingerprint": config.index_fingerprint(),
        "chunking_type": config.indexing.chunking.type,
        "chunk_size": config.indexing.chunking.params.get("chunk_size"),
        "chunk_overlap": config.indexing.chunking.params.get("chunk_overlap"),
        "breakpoint_threshold_type": config.indexing.chunking.params.get(
            "breakpoint_threshold_type"
        ),
        "chunk_enricher_type": config.indexing.chunk_enricher.type,
        "context_scope": config.indexing.chunk_enricher.params.get("context_scope"),
        "embedding_type": config.indexing.embedding.type,
        "embedding_model": config.indexing.embedding.params.get("model_name", ""),
        "vectorstore_type": config.indexing.vectorstore.type,
        "query_transform_type": config.query.query_transform.type or "passthrough",
        "query_transform_fusion": config.query.query_transform.fusion,
        "query_transform_num_queries": (
            config.query.query_transform.params.get("num_queries")
            or config.query.query_transform.params.get("num_hypotheticals")
            or 1
        ),
        "retrieval_type": config.query.retrieval.type,
        "top_k_retrieve": config.query.retrieval.top_k_retrieve,
        "top_k_final": config.query.retrieval.top_k_final,
        "reranking_type": config.query.reranking.type or "none",
        "generation_type": config.query.generator.provider,
        "generation_model": config.query.generator.model_name,
        "evaluator_embedding_type": config.evaluation.evaluator_embedding.type,
    }


def _methodology_note(config: ExperimentConfig) -> dict[str, str]:
    """LLM-as-judge caveats recorded into summary.json (pure documentation).

    A standing reminder, written beside every result, that scored metrics depend
    on a biased judge and how to read the Phase E slice metrics. The per-sample
    judge verdicts are in the run JSONL, so a held-out slice can be human
    spot-checked against these notes.
    """
    notes = {
        "judge_bias": (
            "LLM-as-judge carries position, verbosity, and self-enhancement bias. "
            "Cross-run std is mostly judge variance (generation is temperature=0), "
            "not answer-sampling spread. Spot-check a held-out slice of the "
            "per-sample verdicts in run_N.jsonl against human judgment."
        ),
    }
    protocol = config.evaluation.protocol
    if protocol == "abstention":
        notes["abstention"] = (
            "False/Missed Refusal Rate are reported together, never as one number. "
            "Per-sample 'abstained' verdicts are in run_N.jsonl; context-grounding "
            "metrics are not meaningful on the unanswerable slice."
        )
    elif protocol == "conflict":
        notes["conflict"] = (
            "The injected passage is prepended (a position-robustness test) and "
            "never enters retrieval metrics. A binary AspectCritic limits verbosity "
            "bias; read corpus_fact_retrieved before treating a low "
            "corpus_preference_rate as trust in the injected falsehood."
        )
    elif protocol == "multi_turn":
        notes["multi_turn"] = (
            "Per-turn scoring; history threads the model's actual answers, so "
            "errors propagate. Slice by depends_on_prior and turn index — the "
            "later, non-standalone turns are the hard cases."
        )
    return notes


def save_experiment(
    config: ExperimentConfig,
    result: ExperimentResult,
    base_dir: Path,
    git_info: dict[str, Any] | None = None,
) -> Path:
    """Save experiment results to disk.

    Clears the experiment directory before writing to prevent stale
    files from previous runs (e.g., leftover ``run_3.jsonl`` when
    the current execution only had 1 run).

    Args:
        config: The resolved experiment configuration.
        result: The evaluation result to save.
        base_dir: Parent directory (e.g., ``Path("results")``).
        git_info: Dict with ``"sha"`` and ``"dirty"`` keys, captured
            at experiment start for reproducibility.

    Returns:
        Path to the experiment output directory.
    """
    exp_dir = base_dir / config.name

    # Clear and recreate — the directory is 100% generated output.
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)

    git_info = git_info or {}

    # Summary
    summary = {
        "experiment_name": result.experiment_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": result.metrics,
        "num_runs": result.num_runs,
        "config_snapshot": _build_config_snapshot(config),
        "methodology": _methodology_note(config),
        "git_sha": git_info.get("sha", "unknown"),
        "git_dirty": git_info.get("dirty", False),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Full resolved config for reproducibility
    config_dict = config.model_dump()
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Per-run results: JSONL with per-sample data
    for i, run_samples in enumerate(result.per_run_samples, 1):
        with open(exp_dir / f"run_{i}.jsonl", "w") as f:
            for sample in run_samples:
                f.write(json.dumps(sample.to_result_dict(), default=str) + "\n")

    return exp_dir
