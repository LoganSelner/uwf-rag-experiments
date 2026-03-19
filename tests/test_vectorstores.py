"""Tests for src/components/vectorstores.py — FAISS vector store."""

from __future__ import annotations

import numpy as np

from components.vectorstores import FAISSVectorStore
from core.types import Chunk, EmbeddedChunk, RetrievedChunk


def _make_embedded_chunks(n: int, dim: int = 8) -> list[EmbeddedChunk]:
    """Create N embedded chunks with deterministic vectors."""
    rng = np.random.default_rng(42)
    chunks = []
    for i in range(n):
        c = Chunk(
            content=f"Chunk {i}",
            chunk_id=f"id_{i}",
            metadata={"source_name": "test", "page_number": i + 1},
        )
        vec = rng.standard_normal(dim).tolist()
        chunks.append(EmbeddedChunk(chunk=c, embedding=vec))
    return chunks


class TestFAISSVectorStore:
    def test_add_and_search_basic(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        chunks = _make_embedded_chunks(5)
        store.add(chunks)
        query_vec = chunks[0].embedding  # search for something similar to chunk 0
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_search_empty_store(self) -> None:
        store = FAISSVectorStore()
        results = store.search([0.1] * 8, top_k=3)
        assert results == []

    def test_search_top_k_limits_results(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(10))
        results = store.search([0.1] * 8, top_k=2)
        assert len(results) == 2

    def test_search_top_k_exceeds_total(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(3))
        results = store.search([0.1] * 8, top_k=10)
        assert len(results) == 3

    def test_cosine_metric_scores(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(5))
        results = store.search([0.1] * 8, top_k=5)
        for r in results:
            assert isinstance(r.score, float)

    def test_l2_metric_constructor(self) -> None:
        store = FAISSVectorStore({"metric": "l2"})
        store.add(_make_embedded_chunks(3))
        results = store.search([0.1] * 8, top_k=2)
        assert len(results) == 2

    def test_filtered_search_matches(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(5))
        results = store.search([0.1] * 8, top_k=5, filters={"page_number": 1})
        assert all(r.chunk.metadata["page_number"] == 1 for r in results)

    def test_filtered_search_no_match(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(5))
        results = store.search([0.1] * 8, top_k=5, filters={"page_number": 999})
        assert results == []

    def test_add_empty_list_noop(self) -> None:
        store = FAISSVectorStore()
        store.add([])  # should not crash

    def test_save_and_load_roundtrip(self, tmp_path: str) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        chunks = _make_embedded_chunks(5)
        store.add(chunks)

        store.save(str(tmp_path))
        loaded = FAISSVectorStore({"metric": "cosine"})
        loaded.load(str(tmp_path))

        results = loaded.search(chunks[0].embedding, top_k=3)
        assert len(results) == 3
        assert results[0].chunk.chunk_id == chunks[0].chunk.chunk_id

    def test_add_incremental(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        batch1 = _make_embedded_chunks(3)
        batch2 = _make_embedded_chunks(2)  # different seed won't matter, same rng
        store.add(batch1)
        store.add(batch2)
        results = store.search([0.1] * 8, top_k=10)
        assert len(results) == 5

    def test_score_ordering(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        store.add(_make_embedded_chunks(10))
        results = store.search([0.1] * 8, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_l2_scores_are_negative(self) -> None:
        store = FAISSVectorStore({"metric": "l2"})
        chunks = _make_embedded_chunks(5)
        store.add(chunks)
        results = store.search(chunks[0].embedding, top_k=5)
        # Self-match should score 0.0 (negated distance of 0)
        assert results[0].score == 0.0
        # All others should be strictly negative
        for r in results[1:]:
            assert r.score < 0.0

    def test_l2_score_ordering(self) -> None:
        store = FAISSVectorStore({"metric": "l2"})
        store.add(_make_embedded_chunks(10))
        results = store.search([0.1] * 8, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_l2_filtered_scores_are_negative(self) -> None:
        store = FAISSVectorStore({"metric": "l2"})
        store.add(_make_embedded_chunks(5))
        results = store.search([0.1] * 8, top_k=5, filters={"source_name": "test"})
        for r in results:
            assert r.score <= 0.0

    def test_cosine_scores_are_not_negated(self) -> None:
        store = FAISSVectorStore({"metric": "cosine"})
        chunks = _make_embedded_chunks(5)
        store.add(chunks)
        results = store.search(chunks[0].embedding, top_k=5)
        # Self-match with inner product should be positive
        assert results[0].score > 0.0

    def test_matches_filters_helper(self) -> None:
        chunk = Chunk(content="x", chunk_id="1", metadata={"k": "v", "n": 5})
        assert FAISSVectorStore._matches_filters(chunk, {"k": "v"}) is True
        assert FAISSVectorStore._matches_filters(chunk, {"k": "other"}) is False
        assert FAISSVectorStore._matches_filters(chunk, {"missing": 1}) is False
