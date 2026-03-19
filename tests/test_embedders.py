"""Tests for src/components/embedders.py — HuggingFace embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from components.embedders import HuggingFaceEmbedder
from core.types import Chunk, EmbeddedChunk


def _make_chunks(n: int = 3) -> list[Chunk]:
    """Create N simple chunks for testing."""
    return [
        Chunk(
            content=f"Text {i}",
            chunk_id=f"id_{i}",
            metadata={"source_name": "test"},
        )
        for i in range(n)
    ]


@patch("components.embedders.SentenceTransformer")
class TestHuggingFaceEmbedder:
    def test_embed_query_returns_list_of_floats(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"model_name": "test-model"})
        result = embedder.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    def test_embed_query_passes_normalize_flag(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"normalize": False})
        embedder.embed_query("test")

        mock_model.encode.assert_called_once_with(
            "test", normalize_embeddings=False, show_progress_bar=False
        )

    def test_embed_chunks_returns_embedded_chunks(self, mock_st_cls: MagicMock) -> None:
        chunks = _make_chunks(2)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].chunk is chunks[0]
        assert result[1].chunk is chunks[1]
        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]

    def test_embed_chunks_passes_batch_size(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0]])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"batch_size": 16})
        embedder.embed_chunks(_make_chunks(1))

        _, kwargs = mock_model.encode.call_args
        assert kwargs["batch_size"] == 16

    def test_embed_chunks_empty_list(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([]).reshape(0, 8)
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        result = embedder.embed_chunks([])

        assert result == []

    def test_default_model_name(self, mock_st_cls: MagicMock) -> None:
        mock_st_cls.return_value = MagicMock()
        HuggingFaceEmbedder()
        mock_st_cls.assert_called_once_with("BAAI/bge-m3")

    def test_custom_model_name(self, mock_st_cls: MagicMock) -> None:
        mock_st_cls.return_value = MagicMock()
        HuggingFaceEmbedder({"model_name": "custom/model"})
        mock_st_cls.assert_called_once_with("custom/model")

    def test_default_config_values(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        embedder.embed_query("test")

        mock_model.encode.assert_called_once_with(
            "test", normalize_embeddings=True, show_progress_bar=False
        )
