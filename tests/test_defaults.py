"""Tests for src/components/defaults.py — passthrough/no-op components."""

from __future__ import annotations

from components.defaults import (
    BufferWindowMemory,
    NoMemory,
    NoOpReranker,
    PassthroughQueryTransformer,
)
from core.types import Chunk, RetrievedChunk


def _make_rc(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content="text", chunk_id=chunk_id, metadata={}),
        score=score,
    )


# -----------------------------------------------------------------------
# PassthroughQueryTransformer
# -----------------------------------------------------------------------


class TestPassthroughQueryTransformer:
    def test_returns_same_query(self) -> None:
        qt = PassthroughQueryTransformer()
        result = qt.transform("What is X?")
        assert result == ["What is X?"]

    def test_ignores_history(self) -> None:
        qt = PassthroughQueryTransformer()
        history = [{"role": "user", "content": "prior"}]
        result = qt.transform("Q?", history=history)
        assert result == ["Q?"]


# -----------------------------------------------------------------------
# NoOpReranker
# -----------------------------------------------------------------------


class TestNoOpReranker:
    def test_truncates_to_top_k(self) -> None:
        reranker = NoOpReranker()
        chunks = [_make_rc(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
        result = reranker.rerank("Q?", chunks, top_k=3)
        assert len(result) == 3

    def test_preserves_order(self) -> None:
        reranker = NoOpReranker()
        chunks = [_make_rc("a", 0.9), _make_rc("b", 0.8), _make_rc("c", 0.7)]
        result = reranker.rerank("Q?", chunks, top_k=3)
        assert [r.chunk.chunk_id for r in result] == ["a", "b", "c"]

    def test_top_k_exceeds_total(self) -> None:
        reranker = NoOpReranker()
        chunks = [_make_rc("a", 0.9)]
        result = reranker.rerank("Q?", chunks, top_k=10)
        assert len(result) == 1


# -----------------------------------------------------------------------
# NoMemory
# -----------------------------------------------------------------------


class TestNoMemory:
    def test_add_turn_noop(self) -> None:
        mem = NoMemory()
        mem.add_turn("user", "Hi")  # should not crash

    def test_get_history_empty(self) -> None:
        mem = NoMemory()
        mem.add_turn("user", "Hi")
        assert mem.get_history() == []

    def test_clear_noop(self) -> None:
        mem = NoMemory()
        mem.clear()  # should not crash


# -----------------------------------------------------------------------
# BufferWindowMemory
# -----------------------------------------------------------------------


class TestBufferWindowMemory:
    def test_basic_add_and_get(self) -> None:
        mem = BufferWindowMemory({"window_size": 5})
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_truncates_at_window_size(self) -> None:
        mem = BufferWindowMemory({"window_size": 2})
        mem.add_turn("user", "msg1")
        mem.add_turn("assistant", "msg2")
        mem.add_turn("user", "msg3")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["content"] == "msg2"
        assert history[1]["content"] == "msg3"

    def test_clear(self) -> None:
        mem = BufferWindowMemory({"window_size": 5})
        mem.add_turn("user", "Hi")
        mem.clear()
        assert mem.get_history() == []

    def test_default_window_size(self) -> None:
        mem = BufferWindowMemory()
        assert mem._window_size == 5
