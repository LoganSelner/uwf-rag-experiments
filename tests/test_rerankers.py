"""Tests for src/components/rerankers.py — cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uwf_rag.components.rerankers import CrossEncoderReranker
from uwf_rag.core.types import Chunk, RetrievedChunk


def _make_rc(chunk_id: str, score: float, content: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content=content, chunk_id=chunk_id, metadata={}),
        score=score,
        retrieval_method="dense",
    )


@patch("uwf_rag.components.rerankers.CrossEncoder")
class TestCrossEncoderReranker:
    """All tests patch CrossEncoder to avoid downloading model weights."""

    def test_sorts_by_cross_encoder_score(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([0.1, 0.9, 0.5])
        reranker = CrossEncoderReranker()

        chunks = [_make_rc("a", 0.9), _make_rc("b", 0.1), _make_rc("c", 0.5)]
        result = reranker.rerank("Q?", chunks, top_k=3)

        assert [r.chunk.chunk_id for r in result] == ["b", "c", "a"]

    def test_rejects_classification_cross_encoder(self, mock_ce_cls: MagicMock) -> None:
        """A num_labels>1 (classification) model is rejected at construction."""
        mock_ce_cls.return_value.config.num_labels = 2
        with pytest.raises(ValueError, match="regression reranker"):
            CrossEncoderReranker()

    def test_replaces_scores_with_cross_encoder_scores(
        self, mock_ce_cls: MagicMock
    ) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([0.2, 0.8])
        reranker = CrossEncoderReranker()

        chunks = [_make_rc("a", 0.99), _make_rc("b", 0.01)]
        result = reranker.rerank("Q?", chunks, top_k=2)

        assert result[0].score == 0.8
        assert result[1].score == 0.2

    def test_truncates_to_top_k(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array(
            [0.1, 0.5, 0.9, 0.3, 0.7]
        )
        reranker = CrossEncoderReranker()

        chunks = [_make_rc(f"c{i}", 0.5) for i in range(5)]
        result = reranker.rerank("Q?", chunks, top_k=2)

        assert len(result) == 2
        assert result[0].chunk.chunk_id == "c2"  # score 0.9
        assert result[1].chunk.chunk_id == "c4"  # score 0.7

    def test_top_k_exceeds_total(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([0.5])
        reranker = CrossEncoderReranker()

        chunks = [_make_rc("a", 0.9)]
        result = reranker.rerank("Q?", chunks, top_k=10)

        assert len(result) == 1

    def test_empty_chunks(self, mock_ce_cls: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("Q?", [], top_k=5)

        assert result == []
        mock_ce_cls.return_value.predict.assert_not_called()

    def test_preserves_chunk_objects(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([0.8])
        reranker = CrossEncoderReranker()

        original_chunk = Chunk(
            content="important", chunk_id="x", metadata={"src": "handbook"}
        )
        chunks = [
            RetrievedChunk(chunk=original_chunk, score=0.1, retrieval_method="dense")
        ]
        result = reranker.rerank("Q?", chunks, top_k=1)

        assert result[0].chunk is original_chunk
        assert result[0].retrieval_method == "dense"

    def test_passes_correct_pairs_to_model(self, mock_ce_cls: MagicMock) -> None:
        mock_model = mock_ce_cls.return_value
        mock_model.predict.return_value = np.array([0.5, 0.5])
        reranker = CrossEncoderReranker()

        chunks = [
            _make_rc("a", 0.9, content="first chunk"),
            _make_rc("b", 0.8, content="second chunk"),
        ]
        reranker.rerank("my query", chunks, top_k=2)

        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("my query", "first chunk"), ("my query", "second chunk")]

    def test_respects_batch_size_config(self, mock_ce_cls: MagicMock) -> None:
        mock_model = mock_ce_cls.return_value
        mock_model.predict.return_value = np.array([0.5])
        reranker = CrossEncoderReranker(config={"batch_size": 16})

        chunks = [_make_rc("a", 0.9)]
        reranker.rerank("Q?", chunks, top_k=1)

        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_default_model_name(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([])
        CrossEncoderReranker()

        mock_ce_cls.assert_called_once_with(
            "Alibaba-NLP/gte-reranker-modernbert-base", device=None
        )

    def test_custom_model_and_device(self, mock_ce_cls: MagicMock) -> None:
        mock_ce_cls.return_value.predict.return_value = np.array([])
        CrossEncoderReranker(
            config={
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
            }
        )

        mock_ce_cls.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
