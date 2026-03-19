"""Tests for src/components/retrievers.py — DenseRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from components.retrievers import DenseRetriever
from core.types import Chunk, RetrievedChunk


def _make_rc(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content="text", chunk_id=chunk_id, metadata={}),
        score=score,
    )


class TestDenseRetriever:
    def _wired_retriever(
        self, *, default_filters: dict | None = None
    ) -> tuple[DenseRetriever, MagicMock, MagicMock]:
        """Return a DenseRetriever with mocked embedder and vectorstore."""
        retriever = DenseRetriever()
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vs = MagicMock()
        mock_vs.search.return_value = [_make_rc("c1", 0.9), _make_rc("c2", 0.8)]

        retriever.set_embedder(mock_embedder)
        retriever.set_vectorstore(mock_vs)
        if default_filters:
            retriever.set_default_filters(default_filters)
        return retriever, mock_embedder, mock_vs

    def test_basic_retrieve(self) -> None:
        retriever, mock_emb, mock_vs = self._wired_retriever()
        results = retriever.retrieve("What is X?", top_k=5)
        assert len(results) == 2
        mock_emb.embed_query.assert_called_once_with("What is X?")
        mock_vs.search.assert_called_once()

    def test_top_k_forwarded(self) -> None:
        retriever, _, mock_vs = self._wired_retriever()
        retriever.retrieve("Q?", top_k=10)
        call_kwargs = mock_vs.search.call_args
        assert (
            call_kwargs.kwargs.get("top_k") == 10 or call_kwargs[1].get("top_k") == 10
        )

    def test_merges_default_and_call_filters(self) -> None:
        retriever, _, mock_vs = self._wired_retriever(
            default_filters={"source_name": "handbook"}
        )
        retriever.retrieve("Q?", top_k=5, filters={"page_number": 1})
        call_kwargs = mock_vs.search.call_args
        passed_filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get(
            "filters"
        )
        assert passed_filters["source_name"] == "handbook"
        assert passed_filters["page_number"] == 1

    def test_empty_filters_passes_none(self) -> None:
        retriever, _, mock_vs = self._wired_retriever()
        retriever.retrieve("Q?", top_k=5)
        call_kwargs = mock_vs.search.call_args
        passed_filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get(
            "filters"
        )
        assert passed_filters is None

    def test_no_embedder_raises(self) -> None:
        retriever = DenseRetriever()
        retriever.set_vectorstore(MagicMock())
        with pytest.raises(RuntimeError, match="Embedder not set"):
            retriever.retrieve("Q?", top_k=5)

    def test_no_vectorstore_raises(self) -> None:
        retriever = DenseRetriever()
        retriever.set_embedder(MagicMock())
        with pytest.raises(RuntimeError, match="VectorStore not set"):
            retriever.retrieve("Q?", top_k=5)

    def test_call_filters_override_defaults(self) -> None:
        retriever, _, mock_vs = self._wired_retriever(
            default_filters={"source_name": "old"}
        )
        retriever.retrieve("Q?", top_k=5, filters={"source_name": "new"})
        call_kwargs = mock_vs.search.call_args
        passed_filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get(
            "filters"
        )
        assert passed_filters["source_name"] == "new"
