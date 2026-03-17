"""Tests for Phase 2: prompts, generators, evaluation, comparison, dispatcher.

Tests use mocks to avoid Ollama/RAGAS dependencies in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Trigger component registration
import components  # noqa: F401
from core.config import ExperimentConfig, LLMConfig
from core.registry import registry
from core.types import (
    Chunk,
    ExperimentResult,
    GenerationResult,
    RetrievedChunk,
    ScoredSample,
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
            with pytest.raises(RuntimeError, match="after retries"):
                gen.generate("test")


# -----------------------------------------------------------------------
# EdenAIGenerator
# -----------------------------------------------------------------------


class TestEdenAIGenerator:
    def test_registered(self) -> None:
        assert registry.is_registered("generation", "edenai")

    def test_requires_sub_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDENAI_API_KEY", "test-key")
        from components.generators import EdenAIGenerator

        with pytest.raises(ValueError, match="sub_provider"):
            EdenAIGenerator(
                config={
                    "llm": {"model_name": "gpt-4o"},
                }
            )

    @patch("dotenv.load_dotenv")
    def test_requires_api_key(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EDENAI_API_KEY", raising=False)
        from components.generators import EdenAIGenerator

        with pytest.raises(ValueError, match="EDENAI_API_KEY"):
            EdenAIGenerator(
                config={
                    "llm": {"model_name": "gpt-4o"},
                    "sub_provider": "openai",
                }
            )

    @patch("langchain_community.chat_models.edenai.ChatEdenAI")
    def test_generate_string_prompt(
        self,
        mock_chat_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDENAI_API_KEY", "test-key")
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="The answer is Paris.")
        mock_chat_cls.return_value = mock_instance

        from components.generators import EdenAIGenerator

        gen = EdenAIGenerator(
            config={
                "llm": {"model_name": "gpt-4o", "temperature": 0.0},
                "sub_provider": "openai",
            }
        )
        result = gen.generate("What is the capital of France?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is Paris."
        assert result.metadata["provider"] == "edenai"
        assert result.metadata["sub_provider"] == "openai"
        assert result.metadata["model"] == "gpt-4o"

    @patch("langchain_community.chat_models.edenai.ChatEdenAI")
    def test_generate_message_list(
        self,
        mock_chat_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDENAI_API_KEY", "test-key")
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Response.")
        mock_chat_cls.return_value = mock_instance

        from components.generators import EdenAIGenerator

        gen = EdenAIGenerator(
            config={
                "llm": {"model_name": "gpt-4o"},
                "sub_provider": "openai",
            }
        )
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = gen.generate(messages)
        assert result.answer == "Response."

    @patch("langchain_community.chat_models.edenai.ChatEdenAI")
    def test_api_error_raises_runtime(
        self,
        mock_chat_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDENAI_API_KEY", "test-key")
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Rate limited")
        mock_chat_cls.return_value = mock_instance

        from components.generators import EdenAIGenerator

        gen = EdenAIGenerator(
            config={
                "llm": {"model_name": "gpt-4o"},
                "sub_provider": "openai",
            }
        )
        with pytest.raises(RuntimeError, match="Eden AI API call failed"):
            gen.generate("test")


# -----------------------------------------------------------------------
# LLMConfig params
# -----------------------------------------------------------------------


class TestLLMConfigParams:
    def test_params_parsed(self) -> None:
        config = LLMConfig.from_dict(
            {
                "provider": "edenai",
                "model_name": "gpt-4o",
                "params": {"sub_provider": "openai"},
            }
        )
        assert config.params == {"sub_provider": "openai"}
        assert config.provider == "edenai"

    def test_params_defaults_to_empty(self) -> None:
        config = LLMConfig.from_dict({"provider": "ollama"})
        assert config.params == {}

    def test_params_empty_on_none(self) -> None:
        config = LLMConfig.from_dict(None)
        assert config.params == {}


# -----------------------------------------------------------------------
# EmbedderAdapter
# -----------------------------------------------------------------------


class TestEmbedderAdapter:
    def test_embed_query(self) -> None:
        from evaluation.evaluator import _EmbedderAdapter

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        adapter = _EmbedderAdapter(mock_embedder)
        result = adapter.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test query")

    def test_embed_documents(self) -> None:
        from evaluation.evaluator import _EmbedderAdapter

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        adapter = _EmbedderAdapter(mock_embedder)
        result = adapter.embed_documents(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert mock_embedder.embed_query.call_count == 2


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
        path = tmp_path / "test_dataset.jsonl"
        with open(path, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

        loaded = _load_dataset(str(path))
        assert len(loaded) == 2
        assert loaded[0]["query"] == "Q1?"

    def test_load_dataset_assigns_sequential_ids(self, tmp_path: Path) -> None:
        from evaluation.evaluator import _load_dataset

        path = tmp_path / "no_ids.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"query": "Q1?", "reference": "A1"}) + "\n")
            f.write(json.dumps({"query": "Q2?", "reference": "A2"}) + "\n")

        loaded = _load_dataset(str(path))
        assert loaded[0]["id"] == "1"
        assert loaded[1]["id"] == "2"

    def test_load_dataset_preserves_existing_ids(self, tmp_path: Path) -> None:
        from evaluation.evaluator import _load_dataset

        path = tmp_path / "with_ids.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"id": "q42", "query": "Q1?", "reference": "A1"}) + "\n")

        loaded = _load_dataset(str(path))
        assert loaded[0]["id"] == "q42"

    def test_load_dataset_missing_fields(self, tmp_path: Path) -> None:
        from evaluation.evaluator import _load_dataset

        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"query": "Q?"}) + "\n")

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
        sample_1 = ScoredSample(
            id="1",
            query="Q1?",
            response="A1",
            retrieved_contexts=["c1"],
            reference="R1",
            scores={"accuracy": 0.84},
        )
        sample_2 = ScoredSample(
            id="1",
            query="Q1?",
            response="A1",
            retrieved_contexts=["c1"],
            reference="R1",
            scores={"accuracy": 0.86},
        )
        result = ExperimentResult(
            experiment_name="test_exp",
            metrics={"accuracy": 0.85, "accuracy_std": 0.01},
            per_run_metrics=[
                {"accuracy": 0.84},
                {"accuracy": 0.86},
            ],
            per_run_samples=[[sample_1], [sample_2]],
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

        # Run files are JSONL with per-sample data
        run_1 = tmp_path / "test_exp" / "run_1.jsonl"
        run_2 = tmp_path / "test_exp" / "run_2.jsonl"
        assert run_1.exists()
        assert run_2.exists()

        with open(run_1) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["id"] == "1"
        assert lines[0]["input"]["query"] == "Q1?"
        assert lines[0]["output"]["response"] == "A1"
        assert lines[0]["scores"]["accuracy"] == 0.84


# -----------------------------------------------------------------------
# Retry logic
# -----------------------------------------------------------------------


class TestRetryLogic:
    def test_is_retryable_rejects_programming_errors(self) -> None:
        from components.generators import _is_retryable

        assert _is_retryable(TypeError("bad arg")) is False
        assert _is_retryable(ValueError("bad value")) is False
        assert _is_retryable(KeyError("missing")) is False
        assert _is_retryable(AttributeError("no attr")) is False
        assert _is_retryable(SyntaxError("bad syntax")) is False

    def test_is_retryable_accepts_transient_errors(self) -> None:
        from components.generators import _is_retryable

        assert _is_retryable(ConnectionError("refused")) is True
        assert _is_retryable(TimeoutError("timed out")) is True
        assert _is_retryable(RuntimeError("server 500")) is True
        assert _is_retryable(OSError("network")) is True

    def test_ollama_retries_on_transient_error(self) -> None:
        from urllib.error import URLError

        from components.generators import OllamaGenerator

        gen = OllamaGenerator(config={"llm": {"model_name": "test"}})

        success_response = MagicMock()
        success_response.__enter__ = MagicMock(return_value=success_response)
        success_response.__exit__ = MagicMock(return_value=False)
        success_response.read.return_value = b'{"message": {"content": "ok"}}'

        with patch("components.generators.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [
                URLError("transient"),
                URLError("transient"),
                success_response,
            ]
            result = gen._call_api({"model": "test", "messages": []})
            assert mock_urlopen.call_count == 3
            assert result["message"]["content"] == "ok"

    @patch("langchain_community.chat_models.edenai.ChatEdenAI")
    def test_edenai_retries_on_transient_error(
        self,
        mock_chat_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDENAI_API_KEY", "test-key")
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = [
            ConnectionError("transient"),
            ConnectionError("transient"),
            MagicMock(content="success"),
        ]
        mock_chat_cls.return_value = mock_instance

        from components.generators import EdenAIGenerator

        gen = EdenAIGenerator(
            config={
                "llm": {"model_name": "gpt-4"},
                "sub_provider": "openai",
            }
        )
        result = gen.generate("test")
        assert result.answer == "success"
        assert mock_instance.invoke.call_count == 3


# -----------------------------------------------------------------------
# Per-row error handling
# -----------------------------------------------------------------------


class TestPerRowErrorHandling:
    def test_run_once_skips_failed_queries(self) -> None:
        from core.config import EvaluationConfig
        from evaluation.evaluator import Evaluator

        config = EvaluationConfig(mode="full")
        evaluator = Evaluator(config)

        mock_rag = MagicMock()
        good_result = GenerationResult(
            query="Q",
            answer="A",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk=Chunk(content="ctx", chunk_id="c1", metadata={}),
                    score=0.9,
                ),
            ],
        )
        mock_rag.query.side_effect = [
            good_result,
            RuntimeError("LLM down"),
            good_result,
        ]

        dataset = [
            {"id": "1", "query": "Q1?", "reference": "R1"},
            {"id": "2", "query": "Q2?", "reference": "R2"},
            {"id": "3", "query": "Q3?", "reference": "R3"},
        ]

        samples = evaluator._run_once(mock_rag, dataset)
        assert len(samples) == 2
        assert samples[0].id == "1"
        assert samples[1].id == "3"

    def test_run_once_all_fail_returns_empty(self) -> None:
        from core.config import EvaluationConfig
        from evaluation.evaluator import Evaluator

        config = EvaluationConfig(mode="full")
        evaluator = Evaluator(config)

        mock_rag = MagicMock()
        mock_rag.query.side_effect = RuntimeError("down")

        dataset = [
            {"id": "1", "query": "Q1?", "reference": "R1"},
            {"id": "2", "query": "Q2?", "reference": "R2"},
        ]

        samples = evaluator._run_once(mock_rag, dataset)
        assert samples == []

    def test_evaluate_handles_empty_run(self) -> None:
        from core.config import EvaluationConfig
        from evaluation.evaluator import Evaluator

        config = EvaluationConfig(
            mode="full",
            dataset="dummy.jsonl",
            num_runs=1,
        )
        evaluator = Evaluator(config)

        mock_rag = MagicMock()
        mock_rag.query.side_effect = RuntimeError("down")
        mock_rag.index_artifact.embedder = MagicMock()

        with patch(
            "evaluation.evaluator._load_dataset",
            return_value=[{"id": "1", "query": "Q?", "reference": "R"}],
        ):
            result = evaluator.evaluate(mock_rag)

        assert result.per_run_metrics == [{}]
        assert result.per_run_samples == [[]]
        assert result.metrics == {}


# -----------------------------------------------------------------------
# Additional RAGAS metrics
# -----------------------------------------------------------------------


class TestAdditionalMetrics:
    @pytest.mark.slow
    def test_metric_map_has_all_eight(self) -> None:
        from evaluation.evaluator import _build_metric_map

        _build_metric_map.cache_clear()
        metric_map = _build_metric_map()
        expected = {
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "answer_similarity",
            "factual_correctness",
            "context_precision",
            "context_recall",
            "context_entity_recall",
        }
        assert set(metric_map.keys()) == expected

    def test_default_metrics_unchanged(self) -> None:
        from core.config import EvaluationConfig

        config = EvaluationConfig()
        assert len(config.metrics) == 5
        assert "answer_similarity" not in config.metrics
        assert "factual_correctness" not in config.metrics
        assert "context_recall" not in config.metrics
        assert "context_recall" not in config.retrieval_only_metrics


# -----------------------------------------------------------------------
# Git SHA tracking
# -----------------------------------------------------------------------


class TestGitSHA:
    def test_get_git_sha_returns_string(self) -> None:
        from pipeline.rag import _get_git_sha

        sha = _get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_get_git_sha_graceful_fallback(self) -> None:
        from pipeline.rag import _get_git_sha

        with patch(
            "pipeline.rag.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert _get_git_sha() == "unknown"

    def test_save_results_includes_git_sha(self, tmp_path: Path) -> None:
        from pipeline.rag import _save_results

        config = ExperimentConfig.from_dict({"name": "sha_test"})
        sample = ScoredSample(
            id="1",
            query="Q",
            response="A",
            retrieved_contexts=[],
            reference="R",
            scores={"acc": 0.9},
        )
        result = ExperimentResult(
            experiment_name="sha_test",
            metrics={"acc": 0.9},
            per_run_metrics=[{"acc": 0.9}],
            per_run_samples=[[sample]],
            num_runs=1,
            config_snapshot={"name": "sha_test"},
        )

        _save_results(config, result, tmp_path)

        with open(tmp_path / "sha_test" / "summary.json") as f:
            summary = json.load(f)
        assert "git_sha" in summary
        assert isinstance(summary["git_sha"], str)


# -----------------------------------------------------------------------
# Rich CLI comparison table
# -----------------------------------------------------------------------


class TestRichComparison:
    """Tests for Rich CLI rendering in scripts/compare.py."""

    def test_metric_cell_green(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(0.85, 0.01)
        assert "0.850" in cell.plain
        assert cell.style == "bold green"

    def test_metric_cell_yellow(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(0.65, 0.0)
        assert "0.650" in cell.plain
        assert cell.style == "yellow"

    def test_metric_cell_red(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(0.3, None)
        assert "0.300" in cell.plain
        assert cell.style == "red"

    def test_metric_cell_none(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(None, None)
        assert cell.style == "dim"
        assert "---" in cell.plain

    def test_metric_cell_std_displayed(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(0.85, 0.02)
        assert "\u00b1" in cell.plain
        assert "0.020" in cell.plain

    def test_metric_cell_no_std_when_zero(self) -> None:
        from compare import _metric_cell

        cell = _metric_cell(0.85, 0.0)
        assert "\u00b1" not in cell.plain

    def test_resolve_metric_name_full_name(self) -> None:
        from evaluation.comparison import resolve_metric_name

        metrics = ["faithfulness", "answer_correctness"]
        assert resolve_metric_name("faithfulness", metrics) == "faithfulness"

    def test_resolve_metric_name_short_name(self) -> None:
        from evaluation.comparison import resolve_metric_name

        metrics = ["faithfulness", "answer_correctness"]
        assert resolve_metric_name("Faith.", metrics) == "faithfulness"

    def test_resolve_metric_name_unknown(self) -> None:
        from evaluation.comparison import resolve_metric_name

        metrics = ["faithfulness"]
        assert resolve_metric_name("nonexistent", metrics) is None

    def test_sort_rows_descending(self) -> None:
        from evaluation.comparison import sort_rows

        rows = [
            {"experiment": "a", "faithfulness": 0.7},
            {"experiment": "b", "faithfulness": 0.9},
            {"experiment": "c", "faithfulness": 0.5},
        ]
        result = sort_rows(rows, "faithfulness")
        assert [r["experiment"] for r in result] == ["b", "a", "c"]

    def test_sort_rows_ascending(self) -> None:
        from evaluation.comparison import sort_rows

        rows = [
            {"experiment": "a", "faithfulness": 0.7},
            {"experiment": "b", "faithfulness": 0.9},
        ]
        result = sort_rows(rows, "faithfulness", ascending=True)
        assert [r["experiment"] for r in result] == ["a", "b"]

    def test_sort_rows_none_values_sort_to_bottom(self) -> None:
        from evaluation.comparison import sort_rows

        rows = [
            {"experiment": "a", "faithfulness": None},
            {"experiment": "b", "faithfulness": 0.9},
        ]
        result = sort_rows(rows, "faithfulness")
        assert result[0]["experiment"] == "b"
        assert result[1]["experiment"] == "a"

    def test_render_rich_table_structure(self) -> None:
        import io

        from compare import render_rich_table
        from rich.console import Console

        rows = [
            {"experiment": "exp1", "faithfulness": 0.9, "faithfulness_std": 0.01},
            {"experiment": "exp2", "faithfulness": 0.7, "faithfulness_std": 0.0},
        ]
        table = render_rich_table(
            rows,
            metrics=["faithfulness"],
            console=Console(file=io.StringIO()),
        )
        assert table.row_count == 2
        assert len(table.columns) == 2  # Experiment + faithfulness

    def test_new_metric_short_names_exist(self) -> None:
        from evaluation.comparison import METRIC_SHORT_NAMES

        assert "answer_similarity" in METRIC_SHORT_NAMES
        assert "context_recall" in METRIC_SHORT_NAMES
        assert "factual_correctness" in METRIC_SHORT_NAMES
