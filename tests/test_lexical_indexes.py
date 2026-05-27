"""Tests for src/components/lexical_indexes.py — BM25LexicalIndex."""

from __future__ import annotations

from pathlib import Path

import pytest

from components.lexical_indexes import BM25LexicalIndex
from core.types import Chunk, RetrievedChunk


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            content="Grade forgiveness allows students to retake a course.",
            chunk_id="c1",
            metadata={"source_name": "handbook", "page_number": 1},
        ),
        Chunk(
            content="The policy applies to undergraduate students only.",
            chunk_id="c2",
            metadata={"source_name": "handbook", "page_number": 1},
        ),
        Chunk(
            content="Financial aid applications are due by March 1.",
            chunk_id="c3",
            metadata={"source_name": "handbook", "page_number": 3},
        ),
        Chunk(
            content="The university police department is open 24 hours.",
            chunk_id="c4",
            metadata={"source_name": "handbook", "page_number": 2},
        ),
        Chunk(
            content="Course retake forgives a previous failing grade.",
            chunk_id="c5",
            metadata={"source_name": "policies", "page_number": 7},
        ),
    ]


class TestBM25LexicalIndex:
    def test_add_and_search_lexical_match(self) -> None:
        idx = BM25LexicalIndex()
        idx.add(_make_chunks())
        # "financial aid" is uniquely present in c3 — unambiguous match.
        results = idx.search("financial aid", top_k=3)
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "c3"
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].retrieval_method == "bm25"

    def test_search_empty_store(self) -> None:
        idx = BM25LexicalIndex()
        assert idx.search("anything", top_k=3) == []

    def test_search_top_k_limits_results(self) -> None:
        idx = BM25LexicalIndex()
        idx.add(_make_chunks())
        results = idx.search("students grade course", top_k=2)
        assert len(results) <= 2

    def test_score_ordering(self) -> None:
        idx = BM25LexicalIndex()
        idx.add(_make_chunks())
        results = idx.search("students grade course", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filtered_search_matches(self) -> None:
        idx = BM25LexicalIndex()
        idx.add(_make_chunks())
        results = idx.search(
            "course grade", top_k=5, filters={"source_name": "handbook"}
        )
        assert results
        assert all(r.chunk.metadata["source_name"] == "handbook" for r in results)

    def test_filtered_search_no_match(self) -> None:
        idx = BM25LexicalIndex()
        idx.add(_make_chunks())
        results = idx.search("anything", top_k=5, filters={"source_name": "missing"})
        assert results == []

    def test_add_empty_list_noop(self) -> None:
        idx = BM25LexicalIndex()
        idx.add([])  # should not crash

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        idx = BM25LexicalIndex({"k1": 1.2, "b": 0.7, "method": "lucene"})
        idx.add(_make_chunks())
        save_dir = tmp_path / "bm25"
        idx.save(str(save_dir))

        reloaded = BM25LexicalIndex()
        reloaded.load(str(save_dir))
        # "financial aid" is uniquely in c3 — must survive the roundtrip.
        results = reloaded.search("financial aid", top_k=3)
        assert results
        assert results[0].chunk.chunk_id == "c3"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            BM25LexicalIndex({"method": "not_a_real_method"})

    def test_stemmer_disabled(self) -> None:
        # Without stemming, query "forgives" should not match the
        # stem-equivalent "forgiveness" — but the corpus has chunks
        # containing both, so we just verify the index builds + serves.
        idx = BM25LexicalIndex({"stemmer": None})
        idx.add(_make_chunks())
        results = idx.search("forgiveness", top_k=3)
        assert len(results) > 0

    def test_custom_filter_overretrieve(self) -> None:
        idx = BM25LexicalIndex({"filter_overretrieve": 1})
        idx.add(_make_chunks())
        # Just verify it still functions with a custom multiplier.
        results = idx.search("students", top_k=2, filters={"source_name": "handbook"})
        assert all(r.chunk.metadata["source_name"] == "handbook" for r in results)
