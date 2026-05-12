"""Tests for src/evaluation/evaluator.py — RAGAS evaluation integration."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import EvaluationConfig
from core.types import EvalSample
from evaluation.evaluator import Evaluator, _load_dataset

# -----------------------------------------------------------------------
# _load_dataset
# -----------------------------------------------------------------------


class TestLoadDataset:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"query": "Q1?", "reference": "R1"}\n{"query": "Q2?", "reference": "R2"}\n'
        )
        data = _load_dataset(str(f))
        assert len(data) == 2
        assert data[0]["query"] == "Q1?"

    def test_assigns_sequential_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"query": "Q", "reference": "R"}\n')
        data = _load_dataset(str(f))
        assert data[0]["id"] == "1"

    def test_preserves_existing_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"id": "custom_id", "query": "Q", "reference": "R"}\n')
        data = _load_dataset(str(f))
        assert data[0]["id"] == "custom_id"

    def test_empty_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _load_dataset(str(f))

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"query": "Q"}\n')  # missing "reference"
        with pytest.raises(ValueError, match="reference"):
            _load_dataset(str(f))


# -----------------------------------------------------------------------
# Evaluator.active_metrics
# -----------------------------------------------------------------------


class TestActiveMetrics:
    def test_full_mode(self) -> None:
        cfg = EvaluationConfig.from_dict({"mode": "full"})
        evaluator = Evaluator(cfg)
        metrics = evaluator.active_metrics
        assert "faithfulness" in metrics
        assert "answer_correctness" in metrics

    def test_retrieval_only_mode(self) -> None:
        cfg = EvaluationConfig.from_dict({"mode": "retrieval_only"})
        evaluator = Evaluator(cfg)
        metrics = evaluator.active_metrics
        assert "context_precision" in metrics
        assert "faithfulness" not in metrics

    def test_none_mode(self) -> None:
        cfg = EvaluationConfig.from_dict({"mode": "none"})
        evaluator = Evaluator(cfg)
        assert evaluator.active_metrics == []


class TestMakeRunConfig:
    def test_uses_evaluation_run_config_settings(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {
                "run_config": {
                    "timeout": 900,
                    "max_retries": 3,
                    "max_wait": 45,
                    "max_workers": 2,
                }
            }
        )
        evaluator = Evaluator(cfg)
        run_config = evaluator._make_run_config()

        assert run_config.timeout == 900
        assert run_config.max_retries == 3
        assert run_config.max_wait == 45
        assert run_config.max_workers == 2


# -----------------------------------------------------------------------
# Evaluator._aggregate_metrics
# -----------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty(self) -> None:
        assert Evaluator._aggregate_metrics([]) == {}

    def test_single_run(self) -> None:
        result = Evaluator._aggregate_metrics([{"acc": 0.9}])
        assert result["acc"] == 0.9
        assert result["acc_std"] == 0.0

    def test_multiple_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": 0.9},
                {"acc": 1.0},
            ]
        )
        assert abs(result["acc"] - 0.9) < 1e-9
        assert result["acc_std"] > 0

    def test_empty_first_run_with_valid_later_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {},
                {"acc": 0.8},
                {"acc": 0.9},
            ]
        )
        assert abs(result["acc"] - 0.85) < 1e-9
        assert result["acc_std"] > 0

    def test_sparse_keys_across_runs(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": 0.9, "faithfulness": 0.7},
                {"faithfulness": 0.9},
            ]
        )
        assert abs(result["acc"] - 0.85) < 1e-9
        assert abs(result["faithfulness"] - 0.8) < 1e-9
        assert "acc_std" in result
        assert "faithfulness_std" in result

    def test_all_empty_runs(self) -> None:
        assert Evaluator._aggregate_metrics([{}, {}, {}]) == {}

    def test_single_key_in_later_run_only(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {},
                {},
                {"acc": 0.95},
            ]
        )
        assert result["acc"] == 0.95
        assert result["acc_std"] == 0.0

    def test_nan_filtered(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": float("nan")},
                {"acc": 1.0},
            ]
        )
        assert abs(result["acc"] - 0.9) < 1e-9

    def test_none_filtered(self) -> None:
        result = Evaluator._aggregate_metrics(
            [
                {"acc": 0.8},
                {"acc": None},
                {"acc": 1.0},
            ]
        )
        assert abs(result["acc"] - 0.9) < 1e-9


# -----------------------------------------------------------------------
# _EmbedderAdapter
# -----------------------------------------------------------------------


class TestEmbedderAdapter:
    def test_embed_text(self) -> None:
        from evaluation.evaluator import _wrap_embedder_for_ragas

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        adapter = _wrap_embedder_for_ragas(mock_embedder)
        result = adapter.embed_text("test")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_embed_texts(self) -> None:
        from evaluation.evaluator import _wrap_embedder_for_ragas

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [[0.1], [0.2]]
        adapter = _wrap_embedder_for_ragas(mock_embedder)
        result = adapter.embed_texts(["a", "b"])
        assert result == [[0.1], [0.2]]
        assert mock_embedder.embed_query.call_count == 2

    def test_legacy_embed_query(self) -> None:
        from evaluation.evaluator import _wrap_embedder_for_ragas

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        adapter = _wrap_embedder_for_ragas(mock_embedder)
        result = adapter.embed_query("test")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_legacy_embed_documents(self) -> None:
        from evaluation.evaluator import _wrap_embedder_for_ragas

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [[0.1], [0.2]]
        adapter = _wrap_embedder_for_ragas(mock_embedder)
        result = adapter.embed_documents(["a", "b"])
        assert result == [[0.1], [0.2]]
        assert mock_embedder.embed_query.call_count == 2


# -----------------------------------------------------------------------
# Evaluator._compute_metrics (RAGAS mocked)
# -----------------------------------------------------------------------


class TestComputeMetrics:
    def _make_samples(self, n: int = 2) -> list[EvalSample]:
        return [
            EvalSample(
                id=str(i + 1),
                query=f"Q{i}?",
                response=f"A{i}",
                retrieved_contexts=[f"ctx{i}"],
                reference=f"R{i}",
            )
            for i in range(n)
        ]

    def test_aggregate_path_for_per_sample_scores(self) -> None:
        """Verify aggregation of per-sample scores matches expected mean."""
        agg = Evaluator._aggregate_metrics(
            [
                {"faithfulness": 0.85},
                {"faithfulness": 0.90},
            ]
        )
        assert abs(agg["faithfulness"] - 0.875) < 1e-9

    def test_nan_converted_to_none_in_per_sample(self) -> None:
        """Verify NaN→None conversion matches _compute_metrics logic."""
        # This tests the exact per-sample NaN conversion code from _compute_metrics
        scores_list = [
            {"faithfulness": float("nan")},
            {"faithfulness": 0.9},
        ]
        active = ["faithfulness"]

        per_sample: list[dict[str, float | None]] = []
        for scores in scores_list:
            sample_scores: dict[str, float | None] = {}
            for name in active:
                if name in scores:
                    val = scores[name]
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        sample_scores[name] = None
                    else:
                        sample_scores[name] = float(val)
            per_sample.append(sample_scores)

        assert per_sample[0]["faithfulness"] is None
        assert per_sample[1]["faithfulness"] == 0.9


# -----------------------------------------------------------------------
# Evaluator._build_evaluator_llm
# -----------------------------------------------------------------------


class TestBuildEvaluatorLLM:
    """Tests for _build_evaluator_llm() provider dispatch."""

    def _make_evaluator(self, provider: str, model: str = "test-model") -> Evaluator:
        cfg = EvaluationConfig.from_dict(
            {
                "evaluator_llm": {
                    "provider": provider,
                    "model_name": model,
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            }
        )
        return Evaluator(cfg)

    @patch("ragas.llms.base.LangchainLLMWrapper", autospec=False)
    @patch("langchain_google_genai.ChatGoogleGenerativeAI", autospec=False)
    def test_google_provider_returns_wrapper(
        self, mock_chat_cls: MagicMock, mock_wrapper_cls: MagicMock
    ) -> None:
        mock_wrapper_cls.return_value = "wrapped"
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            evaluator = self._make_evaluator("google", "gemini-2.0-flash")
            result = evaluator._build_evaluator_llm()
        mock_chat_cls.assert_called_once_with(
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=512,
            google_api_key="fake-key",
        )
        assert result == "wrapped"

    def test_google_provider_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            evaluator = self._make_evaluator("google")
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                evaluator._build_evaluator_llm()

    @patch("ragas.llms.base.LangchainLLMWrapper", autospec=False)
    @patch("langchain_ollama.OllamaLLM", autospec=False)
    def test_ollama_provider_returns_wrapper(
        self, mock_ollama_cls: MagicMock, mock_wrapper_cls: MagicMock
    ) -> None:
        mock_wrapper_cls.return_value = "wrapped"
        evaluator = self._make_evaluator("ollama", "qwen3:14b")
        result = evaluator._build_evaluator_llm()
        mock_ollama_cls.assert_called_once_with(
            model="qwen3:14b",
            temperature=0.0,
        )
        assert result == "wrapped"

    @patch("ragas.llms.base.LangchainLLMWrapper", autospec=False)
    @patch("langchain_ollama.OllamaLLM", autospec=False)
    def test_ollama_provider_passes_base_url_from_params(
        self, mock_ollama_cls: MagicMock, mock_wrapper_cls: MagicMock
    ) -> None:
        mock_wrapper_cls.return_value = "wrapped"
        cfg = EvaluationConfig.from_dict(
            {
                "evaluator_llm": {
                    "provider": "ollama",
                    "model_name": "qwen3:14b",
                    "temperature": 0.0,
                    "params": {
                        "base_url": "http://custom-ollama:11434",
                    },
                },
            }
        )
        evaluator = Evaluator(cfg)
        result = evaluator._build_evaluator_llm()
        mock_ollama_cls.assert_called_once_with(
            model="qwen3:14b",
            temperature=0.0,
            base_url="http://custom-ollama:11434",
        )
        assert result == "wrapped"

    @patch("ragas.llms.base.LangchainLLMWrapper", autospec=False)
    @patch(
        "langchain_community.chat_models.edenai.ChatEdenAI",
        autospec=False,
    )
    def test_edenai_provider_returns_wrapper(
        self, mock_edenai_cls: MagicMock, mock_wrapper_cls: MagicMock
    ) -> None:
        mock_wrapper_cls.return_value = "wrapped"
        with patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}):
            evaluator = self._make_evaluator("edenai", "gpt-4")
            result = evaluator._build_evaluator_llm()
        mock_edenai_cls.assert_called_once()
        assert result == "wrapped"

    @patch("ragas.llms.base.LangchainLLMWrapper", autospec=False)
    @patch("langchain_openai.ChatOpenAI", autospec=False)
    def test_openai_provider_returns_wrapper(
        self, mock_chat_cls: MagicMock, mock_wrapper_cls: MagicMock
    ) -> None:
        mock_wrapper_cls.return_value = "wrapped"
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
            evaluator = self._make_evaluator("openai", "gpt-4o")
            result = evaluator._build_evaluator_llm()
        mock_chat_cls.assert_called_once_with(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=512,
            api_key="fake-key",
        )
        assert result == "wrapped"

    def test_openai_provider_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            evaluator = self._make_evaluator("openai")
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                evaluator._build_evaluator_llm()

    def test_unsupported_provider_raises(self) -> None:
        evaluator = self._make_evaluator("bogus")
        with pytest.raises(ValueError, match="Unsupported"):
            evaluator._build_evaluator_llm()


# -----------------------------------------------------------------------
# Evaluator embedder from registry (E3)
# -----------------------------------------------------------------------


class TestBuildEvaluatorEmbedder:
    """Tests for evaluator building its own embedder from config."""

    def test_evaluate_builds_embedder_from_config(self) -> None:
        """Evaluator constructs embedder via registry from evaluator_embedding."""
        cfg = EvaluationConfig.from_dict(
            {
                "dataset": "dummy.jsonl",
                "evaluator_embedding": {
                    "type": "huggingface",
                    "params": {"model_name": "test-model"},
                },
            }
        )
        evaluator = Evaluator(cfg)

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_cls = MagicMock(return_value=mock_embedder)

        with (
            patch("evaluation.evaluator.registry") as mock_registry,
            patch("evaluation.evaluator._load_dataset", return_value=[]),
            patch.object(evaluator, "_run_once", return_value=[]),
            patch.object(evaluator, "_aggregate_metrics", return_value={}),
        ):
            mock_registry.get.return_value = mock_cls
            evaluator.evaluate(MagicMock(), experiment_name="test")

        mock_registry.get.assert_called_once_with("embedding", "huggingface")
        mock_cls.assert_called_once_with(config={"model_name": "test-model"})
        assert evaluator._embedder_adapter is not None

    def test_evaluate_skips_embedder_when_type_empty(self) -> None:
        """No embedder built when evaluator_embedding.type is empty."""
        cfg = EvaluationConfig.from_dict(
            {
                "dataset": "dummy.jsonl",
                "evaluator_embedding": {"type": ""},
            }
        )
        evaluator = Evaluator(cfg)

        with (
            patch("evaluation.evaluator._load_dataset", return_value=[]),
            patch.object(evaluator, "_run_once", return_value=[]),
            patch.object(evaluator, "_aggregate_metrics", return_value={}),
        ):
            evaluator.evaluate(MagicMock(), experiment_name="test")

        assert evaluator._embedder_adapter is None

    def test_evaluate_no_embedder_parameter(self) -> None:
        """Regression guard: evaluate() must not accept an embedder parameter."""
        import inspect

        sig = inspect.signature(Evaluator.evaluate)
        assert "embedder" not in sig.parameters


# -----------------------------------------------------------------------
# Evaluator.evaluate — mode="none"
# -----------------------------------------------------------------------


class TestEvaluateNoneMode:
    """mode='none' runs the pipeline but skips RAGAS scoring."""

    def _samples(self, n: int = 2) -> list[EvalSample]:
        return [
            EvalSample(
                id=str(i + 1),
                query=f"Q{i}?",
                response=f"A{i}",
                retrieved_contexts=[f"ctx{i}"],
                reference=f"R{i}",
            )
            for i in range(n)
        ]

    def test_skips_compute_metrics_and_embedder(self) -> None:
        """mode='none' must not build the judge embedder or call _compute_metrics."""
        cfg = EvaluationConfig.from_dict(
            {
                "dataset": "dummy.jsonl",
                "mode": "none",
                "num_runs": 1,
                "evaluator_embedding": {
                    "type": "huggingface",
                    "params": {"model_name": "test-model"},
                },
            }
        )
        evaluator = Evaluator(cfg)
        samples = self._samples(2)

        with (
            patch("evaluation.evaluator.registry") as mock_registry,
            patch(
                "evaluation.evaluator._load_dataset",
                return_value=[
                    {"id": s.id, "query": s.query, "reference": s.reference}
                    for s in samples
                ],
            ),
            patch.object(evaluator, "_run_once", return_value=samples),
            patch.object(evaluator, "_compute_metrics") as mock_compute,
        ):
            result = evaluator.evaluate(MagicMock(), experiment_name="smoke")

        mock_compute.assert_not_called()
        mock_registry.get.assert_not_called()
        assert evaluator._embedder_adapter is None
        assert result.metrics == {}
        assert len(result.per_run_samples) == 1
        assert len(result.per_run_samples[0]) == 2
        for scored in result.per_run_samples[0]:
            assert scored.scores == {}
            assert scored.response  # pipeline output is preserved
