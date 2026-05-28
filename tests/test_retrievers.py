"""Tests for src/components/retrievers.py — Dense, BM25, and Hybrid retrievers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from components.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from core.types import Chunk, RetrievedChunk, TransformedQuery


def _make_rc(chunk_id: str, score: float, content: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(content=content, chunk_id=chunk_id, metadata={}),
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


# -----------------------------------------------------------------------
# BM25Retriever
# -----------------------------------------------------------------------


class TestBM25Retriever:
    def _wired_retriever(
        self, *, default_filters: dict | None = None
    ) -> tuple[BM25Retriever, MagicMock]:
        retriever = BM25Retriever()
        mock_idx = MagicMock()
        mock_idx.search.return_value = [_make_rc("c1", 12.0), _make_rc("c2", 7.0)]
        retriever.set_sparse_index(mock_idx)
        if default_filters:
            retriever.set_default_filters(default_filters)
        return retriever, mock_idx

    def test_basic_retrieve_delegates_to_index(self) -> None:
        retriever, mock_idx = self._wired_retriever()
        results = retriever.retrieve("What is X?", top_k=5)
        assert len(results) == 2
        mock_idx.search.assert_called_once()
        kwargs = mock_idx.search.call_args.kwargs
        assert kwargs["query"] == "What is X?"
        assert kwargs["top_k"] == 5

    def test_no_sparse_index_raises(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="SparseIndex not set"):
            retriever.retrieve("Q?", top_k=5)

    def test_default_filters_merged(self) -> None:
        retriever, mock_idx = self._wired_retriever(
            default_filters={"source_name": "handbook"}
        )
        retriever.retrieve("Q?", top_k=5, filters={"page_number": 1})
        passed = mock_idx.search.call_args.kwargs["filters"]
        assert passed["source_name"] == "handbook"
        assert passed["page_number"] == 1

    def test_empty_filters_passes_none(self) -> None:
        retriever, mock_idx = self._wired_retriever()
        retriever.retrieve("Q?", top_k=5)
        assert mock_idx.search.call_args.kwargs["filters"] is None


# -----------------------------------------------------------------------
# HybridRetriever
# -----------------------------------------------------------------------


class TestHybridRetriever:
    def _make_child(
        self, name: str, results: list[RetrievedChunk]
    ) -> tuple[str, MagicMock, int]:
        mock = MagicMock()
        mock.retrieve.return_value = results
        return name, mock, 10

    def test_rrf_fusion_basic(self) -> None:
        # Dense and BM25 both rank c1 highest; c2 and c3 appear in one each.
        dense = self._make_child("dense", [_make_rc("c1", 0.95), _make_rc("c2", 0.7)])
        sparse = self._make_child("bm25", [_make_rc("c1", 18.0), _make_rc("c3", 4.5)])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf", rrf_k=60)
        results = hybrid.retrieve("Q?", top_k=5)
        ids = [rc.chunk.chunk_id for rc in results]
        assert ids[0] == "c1"  # in both lists → highest RRF score
        assert set(ids) == {"c1", "c2", "c3"}
        assert all(rc.retrieval_method == "hybrid_rrf" for rc in results)

    def test_weighted_fusion_basic(self) -> None:
        dense = self._make_child("dense", [_make_rc("a", 1.0), _make_rc("b", 0.5)])
        sparse = self._make_child("bm25", [_make_rc("c", 10.0), _make_rc("a", 5.0)])
        hybrid = HybridRetriever(
            children=[dense, sparse],
            fusion="weighted",
            weights={"dense": 0.5, "bm25": 0.5},
            normalize="min_max",
        )
        results = hybrid.retrieve("Q?", top_k=5)
        assert all(rc.retrieval_method == "hybrid_weighted" for rc in results)
        # 'a' is in both lists → should rank top.
        assert results[0].chunk.chunk_id == "a"

    def test_top_k_truncates_fused_list(self) -> None:
        dense = self._make_child(
            "dense", [_make_rc(f"d{i}", 1.0 - i * 0.1) for i in range(5)]
        )
        sparse = self._make_child(
            "bm25", [_make_rc(f"s{i}", 10.0 - i) for i in range(5)]
        )
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")
        results = hybrid.retrieve("Q?", top_k=3)
        assert len(results) == 3

    def test_filters_pushed_to_all_children(self) -> None:
        dense = self._make_child("dense", [_make_rc("c1", 0.9)])
        sparse = self._make_child("bm25", [_make_rc("c1", 5.0)])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")
        hybrid.set_default_filters({"source_name": "handbook"})
        # set_default_filters should have propagated to children
        assert dense[1].set_default_filters.called
        assert sparse[1].set_default_filters.called

    def test_per_call_filters_forwarded_to_children(self) -> None:
        dense = self._make_child("dense", [_make_rc("c1", 0.9)])
        sparse = self._make_child("bm25", [_make_rc("c1", 5.0)])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")
        hybrid.retrieve("Q?", top_k=3, filters={"page_number": 1})
        dense_filters = dense[1].retrieve.call_args.args[2]
        sparse_filters = sparse[1].retrieve.call_args.args[2]
        assert dense_filters == {"page_number": 1}
        assert sparse_filters == {"page_number": 1}

    def test_no_children_raises(self) -> None:
        hybrid = HybridRetriever(children=[], fusion="rrf")
        with pytest.raises(RuntimeError, match="no children"):
            hybrid.retrieve("Q?", top_k=3)

    def test_unknown_fusion_raises(self) -> None:
        dense = self._make_child("dense", [_make_rc("c1", 0.9)])
        sparse = self._make_child("bm25", [_make_rc("c1", 5.0)])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="bogus")
        with pytest.raises(ValueError, match="unknown fusion mode"):
            hybrid.retrieve("Q?", top_k=3)

    def test_child_per_branch_top_k_forwarded(self) -> None:
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [_make_rc("c1", 0.9)]
        sparse_mock = MagicMock()
        sparse_mock.retrieve.return_value = [_make_rc("c2", 5.0)]
        hybrid = HybridRetriever(
            children=[("dense", dense_mock, 20), ("bm25", sparse_mock, 30)],
            fusion="rrf",
        )
        hybrid.retrieve("Q?", top_k=5)
        # First positional arg after query is top_k for each child.
        assert dense_mock.retrieve.call_args.args[1] == 20
        assert sparse_mock.retrieve.call_args.args[1] == 30


# -----------------------------------------------------------------------
# BaseRetriever.retrieve_multi default impl (exercised via DenseRetriever)
# -----------------------------------------------------------------------


class TestBaseRetrieverRetrieveMulti:
    def _wired(self) -> tuple[DenseRetriever, MagicMock]:
        retriever = DenseRetriever()
        retriever.set_embedder(MagicMock(embed_query=lambda q: [0.1]))
        mock_vs = MagicMock()
        retriever.set_vectorstore(mock_vs)
        return retriever, mock_vs

    def test_single_query_returns_results_as_is(self) -> None:
        retriever, mock_vs = self._wired()
        mock_vs.search.return_value = [_make_rc("a", 0.9), _make_rc("b", 0.7)]
        results = retriever.retrieve_multi(
            [TransformedQuery(text="Q?")], top_k_per_query=5
        )
        assert [rc.chunk.chunk_id for rc in results] == ["a", "b"]
        # Single query: no fusion overhead, raw scores preserved.
        assert results[0].score == 0.9

    def test_rrf_fusion_two_queries(self) -> None:
        retriever, mock_vs = self._wired()
        mock_vs.search.side_effect = [
            [_make_rc("shared", 0.9), _make_rc("only_a", 0.5)],
            [_make_rc("shared", 0.6), _make_rc("only_b", 0.4)],
        ]
        results = retriever.retrieve_multi(
            [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
            top_k_per_query=10,
            fusion="rrf",
        )
        ids = [rc.chunk.chunk_id for rc in results]
        assert ids[0] == "shared"  # Appears in both lists → highest RRF score.
        assert set(ids) == {"shared", "only_a", "only_b"}

    def test_max_fusion_keeps_highest_score(self) -> None:
        retriever, mock_vs = self._wired()
        mock_vs.search.side_effect = [
            [_make_rc("c1", 0.5)],
            [_make_rc("c1", 0.9)],
        ]
        results = retriever.retrieve_multi(
            [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
            top_k_per_query=10,
            fusion="max",
        )
        assert results[0].score == 0.9

    def test_branch_hints_silently_ignored(self) -> None:
        retriever, mock_vs = self._wired()
        mock_vs.search.return_value = [_make_rc("a", 0.9)]
        # Branch hints don't apply outside HybridRetriever.
        results = retriever.retrieve_multi(
            [TransformedQuery(text="q", branch="dense")], top_k_per_query=5
        )
        assert len(results) == 1
        mock_vs.search.assert_called_once()

    def test_empty_queries_returns_empty(self) -> None:
        retriever, _ = self._wired()
        assert retriever.retrieve_multi([], top_k_per_query=5) == []

    def test_unknown_fusion_raises(self) -> None:
        retriever, mock_vs = self._wired()
        mock_vs.search.return_value = [_make_rc("a", 0.9)]
        with pytest.raises(ValueError, match="unknown fusion mode"):
            retriever.retrieve_multi(
                [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
                top_k_per_query=5,
                fusion="bogus",
            )

    def test_none_fusion_rejected_for_multiple_queries(self) -> None:
        # Regression: "none" used to silently return only the first
        # query's results, dropping the rest. It's now an unknown mode
        # and must raise rather than lose data.
        retriever, mock_vs = self._wired()
        mock_vs.search.side_effect = [
            [_make_rc("a", 0.9)],
            [_make_rc("b", 0.8)],
        ]
        with pytest.raises(ValueError, match="unknown fusion mode"):
            retriever.retrieve_multi(
                [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
                top_k_per_query=5,
                fusion="none",
            )


# -----------------------------------------------------------------------
# HybridRetriever.retrieve_multi — branch routing + two-level fusion
# -----------------------------------------------------------------------


class TestHybridRetrieverRetrieveMulti:
    def _make_child(
        self, name: str, results_per_call: list[list[RetrievedChunk]]
    ) -> tuple[str, MagicMock, int]:
        """Return (name, mock_child, top_k). Child's retrieve_multi returns
        ``results_per_call[0]`` (the first list — children own their own
        intra-branch fusion via the default base impl)."""
        mock = MagicMock()
        # HybridRetriever calls child.retrieve_multi (not child.retrieve)
        # — the child is responsible for its own intra-branch fusion.
        # For testing we collapse to one canonical fused list per child.
        mock.retrieve_multi.return_value = results_per_call[0]
        return name, mock, 10

    def test_all_unrouted_broadcasts_to_all_children(self) -> None:
        dense = self._make_child("dense", [[_make_rc("c1", 0.95)]])
        sparse = self._make_child("bm25", [[_make_rc("c1", 18.0)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        hybrid.retrieve_multi(
            [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
            top_k_per_query=5,
        )

        # Both children received both queries.
        for _, child, _ in [dense, sparse]:
            received = child.retrieve_multi.call_args.args[0]
            assert len(received) == 2

    def test_branch_routing_exclusive_dispatch(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        hybrid.retrieve_multi(
            [
                TransformedQuery(text="hyp", branch="dense"),
                TransformedQuery(text="orig", branch="bm25"),
            ],
            top_k_per_query=5,
        )

        dense_qs = [q.text for q in dense[1].retrieve_multi.call_args.args[0]]
        sparse_qs = [q.text for q in sparse[1].retrieve_multi.call_args.args[0]]
        assert dense_qs == ["hyp"]
        assert sparse_qs == ["orig"]

    def test_mixed_routed_and_unrouted(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        hybrid.retrieve_multi(
            [
                TransformedQuery(text="hyp", branch="dense"),
                TransformedQuery(text="broadcast"),
            ],
            top_k_per_query=5,
        )

        dense_qs = [q.text for q in dense[1].retrieve_multi.call_args.args[0]]
        sparse_qs = [q.text for q in sparse[1].retrieve_multi.call_args.args[0]]
        # Dense gets the routed query + the unrouted broadcast.
        assert set(dense_qs) == {"hyp", "broadcast"}
        # Sparse gets only the unrouted broadcast (no explicit branch="bm25").
        assert sparse_qs == ["broadcast"]

    def test_unknown_branch_raises(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        with pytest.raises(ValueError, match="unknown branch"):
            hybrid.retrieve_multi(
                [TransformedQuery(text="q", branch="not_a_child")],
                top_k_per_query=5,
            )

    def test_qt_fusion_forwarded_to_children(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        hybrid.retrieve_multi(
            [TransformedQuery(text="q1"), TransformedQuery(text="q2")],
            top_k_per_query=5,
            fusion="max",
        )

        # Each child's retrieve_multi was called with fusion="max" — that's
        # the pipeline-level (intra-branch) fusion. Hybrid's own fusion
        # (cross-branch) is independent.
        for _, child, _ in [dense, sparse]:
            assert child.retrieve_multi.call_args.kwargs["fusion"] == "max"

    def test_per_branch_top_k_forwarded(self) -> None:
        dense_mock = MagicMock()
        dense_mock.retrieve_multi.return_value = [_make_rc("d1", 0.9)]
        sparse_mock = MagicMock()
        sparse_mock.retrieve_multi.return_value = [_make_rc("s1", 0.7)]
        hybrid = HybridRetriever(
            children=[("dense", dense_mock, 20), ("bm25", sparse_mock, 30)],
            fusion="rrf",
        )

        hybrid.retrieve_multi([TransformedQuery(text="q")], top_k_per_query=5)

        assert dense_mock.retrieve_multi.call_args.kwargs["top_k_per_query"] == 20
        assert sparse_mock.retrieve_multi.call_args.kwargs["top_k_per_query"] == 30

    def test_filters_forwarded_to_children(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")
        hybrid.set_default_filters({"source_name": "handbook"})

        hybrid.retrieve_multi(
            [TransformedQuery(text="q")],
            top_k_per_query=5,
            filters={"page_number": 1},
        )

        for _, child, _ in [dense, sparse]:
            forwarded = child.retrieve_multi.call_args.kwargs["filters"]
            # Default filters merge with per-call filters.
            assert forwarded == {"source_name": "handbook", "page_number": 1}

    def test_cross_branch_fusion_label(self) -> None:
        dense = self._make_child("dense", [[_make_rc("c1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("c1", 18.0)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        results = hybrid.retrieve_multi([TransformedQuery(text="q")], top_k_per_query=5)
        assert all(rc.retrieval_method == "hybrid_rrf" for rc in results)

    def test_skips_empty_branches(self) -> None:
        dense = self._make_child("dense", [[_make_rc("d1", 0.9)]])
        sparse = self._make_child("bm25", [[_make_rc("s1", 0.7)]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")

        # Only routes to dense; sparse should be skipped (empty group).
        hybrid.retrieve_multi(
            [TransformedQuery(text="hyp", branch="dense")], top_k_per_query=5
        )
        dense[1].retrieve_multi.assert_called_once()
        sparse[1].retrieve_multi.assert_not_called()

    def test_empty_queries_returns_empty(self) -> None:
        dense = self._make_child("dense", [[]])
        sparse = self._make_child("bm25", [[]])
        hybrid = HybridRetriever(children=[dense, sparse], fusion="rrf")
        assert hybrid.retrieve_multi([], top_k_per_query=5) == []
