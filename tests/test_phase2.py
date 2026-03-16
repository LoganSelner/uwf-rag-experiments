"""Tests for Phase 2: prompts, generators, evaluation, comparison, dispatcher.

Tests use mocks to avoid Ollama/RAGAS dependencies in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Trigger component registration
import components  # noqa: F401
from core.config import ExperimentConfig
from core.registry import registry
from core.types import (
    Chunk,
    ExperimentResult,
    GenerationResult,
    RetrievedChunk,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# -----------------------------------------------------------------------
# ChatPromptTemplate
# -----------------------------------------------------------------------


class TestChatPromptTemplate:
    def test_registered(self) -> None:
        assert registry.is_registered("prompts", "chat")

    def test_basic_format(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "You are a helpful assistant.",
                "context_format": "numbered",
            }
        )
        chunks = [
            RetrievedChunk(
                chunk=Chunk(content="Chunk one.", chunk_id="c1", metadata={}),
                score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(content="Chunk two.", chunk_id="c2", metadata={}),
                score=0.8,
            ),
        ]
        messages = template.format("What is X?", chunks)

        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "You are a helpful assistant." in messages[0]["content"]
        assert "[1] Chunk one." in messages[0]["content"]
        assert "[2] Chunk two." in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is X?"

    def test_plain_context_format(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
                "context_format": "plain",
            }
        )
        chunks = [
            RetrievedChunk(
                chunk=Chunk(content="A", chunk_id="c1", metadata={}),
                score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(content="B", chunk_id="c2", metadata={}),
                score=0.8,
            ),
        ]
        messages = template.format("Q?", chunks)
        system = messages[0]["content"]
        assert "---" in system  # plain uses dividers
        assert "[1]" not in system

    def test_chain_of_thought(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
                "use_chain_of_thought": True,
            }
        )
        messages = template.format("Q?", [])
        system = messages[0]["content"]
        assert "step by step" in system.lower()

    def test_citation_inline(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
                "citation_style": "inline",
            }
        )
        chunks = [
            RetrievedChunk(
                chunk=Chunk(content="X", chunk_id="c1", metadata={}),
                score=0.9,
            ),
        ]
        messages = template.format("Q?", chunks)
        assert "[1], [2]" in messages[0]["content"]

    def test_citation_verbatim(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
                "citation_style": "verbatim",
            }
        )
        messages = template.format("Q?", [])
        assert "verbatim" in messages[0]["content"].lower()

    def test_few_shot_examples(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
                "few_shot_examples": [
                    {"query": "What is A?", "answer": "A is X."},
                ],
            }
        )
        messages = template.format("Q?", [])
        # System + few-shot user + few-shot assistant + actual user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is A?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "A is X."

    def test_history_included(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "Answer.",
            }
        )
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = template.format("New Q?", [], history)
        # system + 2 history + user
        assert len(messages) == 4
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["content"] == "Previous answer"

    def test_no_chunks_no_context_section(self) -> None:
        from components.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            config={
                "system_template": "You are helpful.",
            }
        )
        messages = template.format("Q?", [])
        assert "CONTEXT:" not in messages[0]["content"]


# -----------------------------------------------------------------------
# OllamaGenerator
# -----------------------------------------------------------------------


class TestOllamaGenerator:
    def test_registered(self) -> None:
        assert registry.is_registered("generation", "ollama")

    def test_generate_with_string_prompt(self) -> None:
        from components.generators import OllamaGenerator

        mock_response = {
            "message": {"content": "The answer is 42."},
            "total_duration": 1000,
            "eval_count": 10,
        }

        gen = OllamaGenerator(
            config={
                "llm": {"model_name": "test-model", "temperature": 0.0},
            }
        )

        with patch.object(gen, "_call_api", return_value=mock_response):
            result = gen.generate("What is the answer?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is 42."
        assert result.metadata["model"] == "test-model"
        assert result.metadata["provider"] == "ollama"

    def test_generate_with_message_list(self) -> None:
        from components.generators import OllamaGenerator

        mock_response = {
            "message": {"content": "Response."},
        }

        gen = OllamaGenerator(
            config={
                "llm": {"model_name": "test-model"},
            }
        )

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Question."},
        ]

        with patch.object(gen, "_call_api", return_value=mock_response):
            result = gen.generate(messages)

        assert result.answer == "Response."

    def test_connection_error(self) -> None:
        from urllib.error import URLError

        from components.generators import OllamaGenerator

        gen = OllamaGenerator(
            config={
                "llm": {"model_name": "test-model"},
                "base_url": "http://localhost:99999",
            }
        )

        with patch.object(gen, "_call_api", side_effect=URLError("Connection refused")):
            with pytest.raises(URLError):
                gen.generate("test")


# -----------------------------------------------------------------------
# Comparison utility
# -----------------------------------------------------------------------


class TestComparison:
    def test_load_summary(self, tmp_path: Path) -> None:
        from evaluation.comparison import load_summary

        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        summary = {
            "experiment_name": "test",
            "metrics": {"answer_correctness": 0.85},
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        loaded = load_summary(exp_dir)
        assert loaded["experiment_name"] == "test"

    def test_load_summary_missing(self, tmp_path: Path) -> None:
        from evaluation.comparison import load_summary

        with pytest.raises(FileNotFoundError):
            load_summary(tmp_path / "nonexistent")

    def test_compare_experiments(self, tmp_path: Path) -> None:
        from evaluation.comparison import compare_experiments

        for name, score in [("exp1", 0.85), ("exp2", 0.90)]:
            d = tmp_path / name
            d.mkdir()
            with open(d / "summary.json", "w") as f:
                json.dump(
                    {
                        "experiment_name": name,
                        "metrics": {
                            "answer_correctness": score,
                            "answer_correctness_std": 0.01,
                        },
                    },
                    f,
                )

        rows = compare_experiments(
            [tmp_path / "exp1", tmp_path / "exp2"],
            metrics=["answer_correctness"],
        )
        assert len(rows) == 2
        assert rows[0]["answer_correctness"] == 0.85
        assert rows[1]["answer_correctness"] == 0.90

    def test_format_comparison_table(self) -> None:
        from evaluation.comparison import format_comparison_table

        rows = [
            {
                "experiment": "baseline",
                "answer_correctness": 0.815,
                "answer_correctness_std": 0.02,
            },
        ]
        table = format_comparison_table(rows, metrics=["answer_correctness"])
        assert "baseline" in table
        assert "0.815" in table
        assert "±" in table


# -----------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------


class TestEvaluator:
    def test_active_metrics_full_mode(self) -> None:
        from core.config import EvaluationConfig
        from evaluation.evaluator import Evaluator

        config = EvaluationConfig(mode="full")
        evaluator = Evaluator(config)
        assert "answer_correctness" in evaluator.active_metrics
        assert "faithfulness" in evaluator.active_metrics

    def test_active_metrics_retrieval_only(self) -> None:
        from core.config import EvaluationConfig
        from evaluation.evaluator import Evaluator

        config = EvaluationConfig(mode="retrieval_only")
        evaluator = Evaluator(config)
        assert "context_precision" in evaluator.active_metrics
        assert "answer_correctness" not in evaluator.active_metrics

    def test_load_dataset(self, tmp_path: Path) -> None:
        from evaluation.evaluator import _load_dataset

        data = [
            {"query": "Q1?", "reference": "A1"},
            {"query": "Q2?", "reference": "A2"},
        ]
        path = tmp_path / "test_dataset.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = _load_dataset(str(path))
        assert len(loaded) == 2
        assert loaded[0]["query"] == "Q1?"

    def test_load_dataset_missing_fields(self, tmp_path: Path) -> None:
        from evaluation.evaluator import _load_dataset

        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump([{"query": "Q?"}], f)

        with pytest.raises(ValueError, match="missing"):
            _load_dataset(str(path))

    def test_aggregate_metrics(self) -> None:
        from evaluation.evaluator import Evaluator

        per_run = [
            {"accuracy": 0.8, "precision": 0.9},
            {"accuracy": 0.9, "precision": 0.8},
            {"accuracy": 0.85, "precision": 0.85},
        ]
        agg = Evaluator._aggregate_metrics(per_run)
        assert abs(agg["accuracy"] - 0.85) < 0.01
        assert "accuracy_std" in agg
        assert agg["accuracy_std"] > 0


# -----------------------------------------------------------------------
# RAGPipeline dispatcher
# -----------------------------------------------------------------------


class TestRAGPipeline:
    def _mock_index(self) -> None:
        """Patch IndexingPipeline.run_or_load_cache to skip real indexing."""
        from unittest.mock import MagicMock

        from core.types import IndexArtifact

        mock_artifact = IndexArtifact(
            vectorstore=MagicMock(),
            embedder=MagicMock(),
            stats={"mocked": True},
        )
        return mock_artifact

    def test_agent_mode_raises(self) -> None:
        from pipeline.rag import RAGPipeline

        config = ExperimentConfig.from_dict(
            {
                "pipeline_mode": "agent",
            }
        )
        mock_artifact = self._mock_index()
        with patch(
            "pipeline.rag.IndexingPipeline.run_or_load_cache",
            return_value=mock_artifact,
        ):
            with pytest.raises(NotImplementedError, match="Agent pipeline"):
                RAGPipeline.from_config(config)

    def test_invalid_mode_raises(self) -> None:
        from pipeline.rag import RAGPipeline

        config = ExperimentConfig.from_dict(
            {
                "pipeline_mode": "invalid",
            }
        )
        mock_artifact = self._mock_index()
        with patch(
            "pipeline.rag.IndexingPipeline.run_or_load_cache",
            return_value=mock_artifact,
        ):
            with pytest.raises(ValueError, match="Unknown pipeline_mode"):
                RAGPipeline.from_config(config)

    def test_save_results(self, tmp_path: Path) -> None:
        from pipeline.rag import _save_results

        config = ExperimentConfig.from_dict({"name": "test_exp"})
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"accuracy": 0.85, "accuracy_std": 0.01},
            per_run_metrics=[
                {"accuracy": 0.84},
                {"accuracy": 0.86},
            ],
            num_runs=2,
            config_snapshot={"name": "test_exp"},
        )

        _save_results(config, result, tmp_path)

        summary_path = tmp_path / "test_exp" / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["experiment_name"] == "test_exp"
        assert summary["metrics"]["accuracy"] == 0.85

        assert (tmp_path / "test_exp" / "run_1.json").exists()
        assert (tmp_path / "test_exp" / "run_2.json").exists()
