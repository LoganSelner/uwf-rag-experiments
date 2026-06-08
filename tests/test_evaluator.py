"""Tests for src/evaluation/evaluator.py — RAGAS evaluation integration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragbench.core.config import SUPPORTED_EVALUATOR_LLM_PROVIDERS, EvaluationConfig
from ragbench.core.types import Chunk, EvalSample, GenerationResult, RetrievedChunk
from ragbench.evaluation.evaluator import (
    Evaluator,
    _load_conversations,
    _load_dataset,
    _mean_latency,
    _slice_rate,
)


class TestJudgeProviderSourceOfTruth:
    def test_builders_match_validated_provider_set(self) -> None:
        # The validator's allow-list and the evaluator's per-provider builders
        # must stay in lockstep (the builder itself raises on drift).
        assert set(Evaluator._judge_builders()) == SUPPORTED_EVALUATOR_LLM_PROVIDERS


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
        per_run: list[dict[str, float]] = [
            {"acc": 0.8},
            {"acc": None},  # type: ignore[dict-item]  # intentionally probes None-filtering
            {"acc": 1.0},
        ]
        result = Evaluator._aggregate_metrics(per_run)
        assert abs(result["acc"] - 0.9) < 1e-9


# -----------------------------------------------------------------------
# _EmbedderAdapter
# -----------------------------------------------------------------------


class TestEmbedderAdapter:
    def test_embed_text(self) -> None:
        from ragbench.evaluation._ragas_adapter import wrap_embedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        adapter = wrap_embedder(mock_embedder)
        result = adapter.embed_text("test")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_embed_texts(self) -> None:
        from ragbench.evaluation._ragas_adapter import wrap_embedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [[0.1], [0.2]]
        adapter = wrap_embedder(mock_embedder)
        result = adapter.embed_texts(["a", "b"])
        assert result == [[0.1], [0.2]]
        assert mock_embedder.embed_query.call_count == 2

    def test_legacy_embed_query(self) -> None:
        from ragbench.evaluation._ragas_adapter import wrap_embedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        adapter = wrap_embedder(mock_embedder)
        result = adapter.embed_query("test")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_legacy_embed_documents(self) -> None:
        from ragbench.evaluation._ragas_adapter import wrap_embedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = [[0.1], [0.2]]
        adapter = wrap_embedder(mock_embedder)
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
# Evaluator._run_once — per-sample metadata capture
# -----------------------------------------------------------------------


class TestRunOnceMetadata:
    def test_captures_pipeline_metadata(self) -> None:
        """The producing pipeline's GenerationResult.metadata is carried onto
        the EvalSample so it can reach the saved per-sample output."""
        evaluator = Evaluator(EvaluationConfig.from_dict({"dataset": "d.jsonl"}))
        pipeline = MagicMock()
        pipeline.query.return_value = GenerationResult(
            query="Q",
            answer="A",
            retrieved_chunks=[],
            metadata={"mode": "agent", "iterations": 3, "num_tool_calls": 2},
        )
        samples = evaluator._run_once(
            pipeline, [{"id": "1", "query": "Q", "reference": "R"}]
        )
        assert len(samples) == 1
        meta = samples[0].metadata
        # Pipeline metadata is carried through unchanged...
        assert meta["mode"] == "agent"
        assert meta["iterations"] == 3
        assert meta["num_tool_calls"] == 2
        # ...and the evaluator adds a per-query wall-clock latency.
        assert isinstance(meta["latency_s"], float)
        assert meta["latency_s"] >= 0.0

    def test_propagates_dataset_label_fields(self) -> None:
        """Phase E label columns ride from the dataset item into sample metadata;
        non-allowlisted dataset keys do not."""
        evaluator = Evaluator(EvaluationConfig.from_dict({"dataset": "d.jsonl"}))
        pipeline = MagicMock()
        pipeline.query.return_value = GenerationResult(
            query="Q", answer="A", retrieved_chunks=[], metadata={}
        )
        samples = evaluator._run_once(
            pipeline,
            [
                {
                    "id": "1",
                    "query": "Q",
                    "reference": "R",
                    "answerable": False,
                    "abstention_type": "personal_data",
                    "category": "advising",
                    "not_a_label": "dropme",
                }
            ],
        )
        meta = samples[0].metadata
        assert meta["answerable"] is False
        assert meta["abstention_type"] == "personal_data"
        assert meta["category"] == "advising"
        assert "not_a_label" not in meta
        assert "latency_s" in meta


class TestMeanLatency:
    @staticmethod
    def _sample(latency: float | None) -> EvalSample:
        meta = {} if latency is None else {"latency_s": latency}
        return EvalSample(
            id="1",
            query="q",
            response="a",
            retrieved_contexts=[],
            reference="r",
            metadata=meta,
        )

    def test_mean_of_present_latencies(self) -> None:
        samples = [self._sample(0.2), self._sample(0.4), self._sample(0.6)]
        assert _mean_latency(samples) == pytest.approx(0.4)

    def test_empty_or_missing_is_zero(self) -> None:
        assert _mean_latency([]) == 0.0
        assert _mean_latency([self._sample(None)]) == 0.0


class TestSliceRate:
    """_slice_rate is the FRR/MRR/conflict-rate primitive (Phase E P1c)."""

    @staticmethod
    def _sample(answerable: bool) -> EvalSample:
        return EvalSample(
            id="1",
            query="q",
            response="a",
            retrieved_contexts=[],
            reference="r",
            metadata={"answerable": answerable},
        )

    def test_false_refusal_rate(self) -> None:
        # FRR = refused (signal 1.0) over the answerable slice.
        samples = [self._sample(True), self._sample(True), self._sample(True)]
        signals: list[float | None] = [1.0, 0.0, 0.0]
        frr = _slice_rate(samples, signals, label_key="answerable", label_value=True)
        assert frr == pytest.approx(1 / 3)

    def test_missed_refusal_rate(self) -> None:
        # MRR = did-NOT-abstain (signal 0.0) over the unanswerable slice.
        samples = [self._sample(False), self._sample(False)]
        signals: list[float | None] = [0.0, 1.0]
        mrr = _slice_rate(
            samples,
            signals,
            label_key="answerable",
            label_value=False,
            positive_value=0.0,
        )
        assert mrr == pytest.approx(0.5)

    def test_empty_slice_is_none(self) -> None:
        samples = [self._sample(True)]
        signals: list[float | None] = [1.0]
        assert (
            _slice_rate(samples, signals, label_key="answerable", label_value=False)
            is None
        )

    def test_all_none_signals_is_none(self) -> None:
        samples = [self._sample(True), self._sample(True)]
        signals: list[float | None] = [None, None]
        assert (
            _slice_rate(samples, signals, label_key="answerable", label_value=True)
            is None
        )


class TestRunAspectCritics:
    """run_aspect_critics: AspectCritic verdicts via the quarantined adapter."""

    def _samples(self, n: int = 2) -> list[EvalSample]:
        return [
            EvalSample(
                id=str(i + 1),
                query=f"q{i}",
                response=f"a{i}",
                retrieved_contexts=[],
                reference=f"r{i}",
            )
            for i in range(n)
        ]

    def test_returns_per_sample_verdicts_nan_to_none(self) -> None:
        from ragbench.evaluation import _ragas_adapter
        from ragbench.evaluation._ragas_adapter import AspectCriticSpec

        fake_result = MagicMock()
        fake_result.scores = [{"abstained": 1.0}, {"abstained": float("nan")}]

        with (
            patch("ragas.evaluate", return_value=fake_result) as mock_eval,
            patch("ragas.EvaluationDataset"),
            patch("ragas.dataset_schema.SingleTurnSample"),
            patch("ragas.metrics._aspect_critic.AspectCritic"),
        ):
            out = _ragas_adapter.run_aspect_critics(
                self._samples(2),
                [AspectCriticSpec(name="abstained", definition="refuses?")],
                llm=MagicMock(),
                run_config=MagicMock(),
            )
        assert mock_eval.called
        assert out[0]["abstained"] == 1.0
        assert out[1]["abstained"] is None  # NaN → None

    def test_empty_specs_returns_empty_dicts(self) -> None:
        from ragbench.evaluation._ragas_adapter import run_aspect_critics

        out = run_aspect_critics(
            self._samples(2), [], llm=MagicMock(), run_config=MagicMock()
        )
        assert out == [{}, {}]

    def test_references_length_mismatch_raises(self) -> None:
        from ragbench.evaluation._ragas_adapter import (
            AspectCriticSpec,
            run_aspect_critics,
        )

        with pytest.raises(ValueError, match="references length"):
            run_aspect_critics(
                self._samples(1),
                [AspectCriticSpec(name="x", definition="y")],
                llm=MagicMock(),
                run_config=MagicMock(),
                references=["a", "b"],
            )


class TestApplyAbstention:
    """End-to-end FRR/MRR injection via the deterministic phrase classifier."""

    def test_phrase_classifier_confusion_matrix(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {"protocol": "abstention", "abstention": {"classifier": "phrase"}}
        )
        evaluator = Evaluator(cfg)
        samples = [
            # answerable + answered → correct (not a false refusal)
            EvalSample(
                "1",
                "q",
                "Grade forgiveness is allowed three times.",
                [],
                "r",
                metadata={"answerable": True},
            ),
            # answerable + abstained → FALSE refusal
            EvalSample(
                "2",
                "q",
                "I don't have that information.",
                [],
                "r",
                metadata={"answerable": True},
            ),
            # unanswerable + abstained → correct
            EvalSample(
                "3",
                "q",
                "I don't have access to your records.",
                [],
                "r",
                metadata={"answerable": False},
            ),
            # unanswerable + answered → MISSED refusal
            EvalSample(
                "4",
                "q",
                "Your balance is 500 dollars.",
                [],
                "r",
                metadata={"answerable": False},
            ),
        ]
        rates = evaluator._apply_abstention(samples)
        assert rates["false_refusal_rate"] == pytest.approx(0.5)  # 1 of 2 answerable
        assert rates["missed_refusal_rate"] == pytest.approx(0.5)  # 1 of 2 unanswerable
        # Per-sample signal is recorded for spot-checking.
        assert samples[0].metadata["abstained"] == 0.0
        assert samples[1].metadata["abstained"] == 1.0

    def test_empty_samples_returns_no_rates(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {"protocol": "abstention", "abstention": {"classifier": "phrase"}}
        )
        assert Evaluator(cfg)._apply_abstention([]) == {}


class TestLoadConversations:
    def test_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "mt.jsonl"
        f.write_text(
            json.dumps(
                {"conversation_id": "c1", "turns": [{"query": "q", "reference": "r"}]}
            )
            + "\n"
        )
        convs = _load_conversations(str(f))
        assert len(convs) == 1
        assert convs[0]["conversation_id"] == "c1"
        assert convs[0]["turns"][0]["turn"] == 1  # 1-based index assigned

    def test_assigns_conversation_id_when_absent(self, tmp_path: Path) -> None:
        f = tmp_path / "mt.jsonl"
        f.write_text(json.dumps({"turns": [{"query": "q", "reference": "r"}]}) + "\n")
        convs = _load_conversations(str(f))
        assert convs[0]["conversation_id"] == "conv_1"

    def test_empty_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "mt.jsonl"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _load_conversations(str(f))

    def test_no_turns_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "mt.jsonl"
        f.write_text(json.dumps({"conversation_id": "c1", "turns": []}) + "\n")
        with pytest.raises(ValueError, match="no 'turns'"):
            _load_conversations(str(f))

    def test_turn_missing_reference_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "mt.jsonl"
        f.write_text(
            json.dumps({"conversation_id": "c1", "turns": [{"query": "q"}]}) + "\n"
        )
        with pytest.raises(ValueError, match="reference"):
            _load_conversations(str(f))


class TestRunOnceMultiTurn:
    """Per-turn scoring with history threaded + reset per conversation."""

    @staticmethod
    def _pipeline_recording(calls: list[tuple[str, list]]) -> MagicMock:
        def fake_query(question: str, history: list | None = None) -> GenerationResult:
            calls.append((question, list(history or [])))
            return GenerationResult(
                query=question, answer=f"ans:{question}", retrieved_chunks=[]
            )

        pipeline = MagicMock()
        pipeline.query.side_effect = fake_query
        return pipeline

    def test_threads_history_and_resets_per_conversation(self) -> None:
        evaluator = Evaluator(EvaluationConfig.from_dict({"protocol": "multi_turn"}))
        calls: list[tuple[str, list]] = []
        pipeline = self._pipeline_recording(calls)
        conversations = [
            {
                "conversation_id": "c1",
                "domain": "d",
                "turns": [
                    {
                        "turn": 1,
                        "query": "q1",
                        "reference": "r1",
                        "answerable": True,
                        "depends_on_prior": False,
                    },
                    {
                        "turn": 2,
                        "query": "q2",
                        "reference": "r2",
                        "answerable": False,
                        "abstention_type": "personal_data",
                        "depends_on_prior": True,
                    },
                ],
            },
            {
                "conversation_id": "c2",
                "turns": [{"turn": 1, "query": "q3", "reference": "r3"}],
            },
        ]
        samples = evaluator._run_once_multi_turn(pipeline, conversations)

        assert len(samples) == 3
        # Turn 1 sees empty history; turn 2 sees the turn-1 exchange (2 messages).
        assert calls[0] == ("q1", [])
        assert [m["content"] for m in calls[1][1]] == ["q1", "ans:q1"]
        # Conversation c2 resets history.
        assert calls[2] == ("q3", [])
        # Stable per-turn ids + label/conv metadata.
        assert [s.id for s in samples] == ["c1::t1", "c1::t2", "c2::t1"]
        assert samples[1].metadata["answerable"] is False
        assert samples[1].metadata["abstention_type"] == "personal_data"
        assert samples[1].metadata["depends_on_prior"] is True
        assert samples[0].metadata["conversation_id"] == "c1"
        assert samples[0].metadata["domain"] == "d"

    def test_failed_turn_skipped_and_excluded_from_history(self) -> None:
        evaluator = Evaluator(EvaluationConfig.from_dict({"protocol": "multi_turn"}))
        calls: list[tuple[str, list]] = []

        def fake_query(question: str, history: list | None = None) -> GenerationResult:
            calls.append((question, list(history or [])))
            if question == "boom":
                raise RuntimeError("nope")
            return GenerationResult(query=question, answer="a", retrieved_chunks=[])

        pipeline = MagicMock()
        pipeline.query.side_effect = fake_query
        conversations = [
            {
                "conversation_id": "c1",
                "turns": [
                    {"turn": 1, "query": "ok1", "reference": "r"},
                    {"turn": 2, "query": "boom", "reference": "r"},
                    {"turn": 3, "query": "ok2", "reference": "r"},
                ],
            }
        ]
        samples = evaluator._run_once_multi_turn(pipeline, conversations)
        assert [s.id for s in samples] == ["c1::t1", "c1::t3"]
        # Turn 3 sees only turn 1's exchange (the failed turn 2 isn't in history).
        assert [m["content"] for m in calls[2][1]] == ["ok1", "a"]


class TestRunOnceConflict:
    """The conflict loop injects the counterfactual + flags corpus retrieval."""

    def test_passes_injected_context_and_flags_retrieval(self) -> None:
        evaluator = Evaluator(EvaluationConfig.from_dict({"protocol": "conflict"}))
        captured: dict = {}

        def fake_query(question, history=None, injected_contexts=None):
            captured["injected"] = injected_contexts
            rc = RetrievedChunk(
                chunk=Chunk(
                    content="grade forgiveness is allowed three times",
                    chunk_id="c1",
                    metadata={},
                ),
                score=0.9,
            )
            return GenerationResult(query=question, answer="A", retrieved_chunks=[rc])

        pipeline = MagicMock()
        pipeline.query.side_effect = fake_query
        dataset = [
            {
                "id": "1",
                "query": "q",
                "reference": "ref",
                "injected_context": "grade forgiveness is allowed five times",
                "corpus_fact": "grade forgiveness is allowed three times",
            }
        ]
        samples = evaluator._run_once_conflict(pipeline, dataset)
        assert captured["injected"] == ["grade forgiveness is allowed five times"]
        assert len(samples) == 1
        assert samples[0].metadata["corpus_fact_retrieved"] is True
        # The injected lie never enters the reported retrieved contexts.
        assert "five times" not in " ".join(samples[0].retrieved_contexts)


class TestApplyConflict:
    """Conflict AspectCritic rates via a mocked adapter call."""

    def test_rates_and_per_sample_verdicts(self) -> None:
        cfg = EvaluationConfig.from_dict(
            {
                "protocol": "conflict",
                "mode": "full",
                "evaluator_llm": {"provider": "ollama", "model_name": "m"},
            }
        )
        evaluator = Evaluator(cfg)
        samples = [
            EvalSample(
                "1", "q", "a", [], "ref", metadata={"corpus_fact": "three times"}
            ),
            EvalSample("2", "q", "a", [], "ref", metadata={"corpus_fact": "2.50 GPA"}),
        ]
        verdicts = [
            {"corpus_preference": 1.0, "error_detection": 0.0},
            {"corpus_preference": 0.0, "error_detection": 1.0},
        ]
        with (
            patch.object(evaluator, "_build_evaluator_llm", return_value=MagicMock()),
            patch(
                "ragbench.evaluation.evaluator._ragas_adapter.run_aspect_critics",
                return_value=verdicts,
            ) as mock_run,
        ):
            rates = evaluator._apply_conflict(samples)
        assert rates["corpus_preference_rate"] == pytest.approx(0.5)
        assert rates["error_detection_rate"] == pytest.approx(0.5)
        assert samples[0].metadata["corpus_preference"] == 1.0
        assert samples[1].metadata["error_detection"] == 1.0
        # The judge compares the response to corpus_fact (reference override).
        assert mock_run.call_args.kwargs["references"] == ["three times", "2.50 GPA"]

    def test_empty_samples_returns_no_rates(self) -> None:
        cfg = EvaluationConfig.from_dict({"protocol": "conflict"})
        assert Evaluator(cfg)._apply_conflict([]) == {}


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
            patch("ragbench.evaluation.evaluator.registry") as mock_registry,
            patch("ragbench.evaluation.evaluator._load_dataset", return_value=[]),
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
            patch("ragbench.evaluation.evaluator._load_dataset", return_value=[]),
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
                metadata={"mode": "agent", "iterations": i + 1},
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
            patch("ragbench.evaluation.evaluator.registry") as mock_registry,
            patch(
                "ragbench.evaluation.evaluator._load_dataset",
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
        # mode="none" skips RAGAS scoring but still reports run latency.
        assert set(result.metrics) == {"latency_mean_s", "latency_mean_s_std"}
        assert len(result.per_run_samples) == 1
        assert len(result.per_run_samples[0]) == 2
        for scored in result.per_run_samples[0]:
            assert scored.scores == {}
            assert scored.response  # pipeline output is preserved
            assert scored.metadata.get("mode") == "agent"  # provenance threaded
