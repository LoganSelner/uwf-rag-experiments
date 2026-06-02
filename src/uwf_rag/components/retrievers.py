"""Retriever implementations.

Each retriever uses an embedder + vectorstore (dense), a sparse
index (lexical), or composes child retrievers (hybrid) to find
relevant chunks for a query.
"""

from __future__ import annotations

import logging
from typing import Any

from uwf_rag.components.base import (
    BaseEmbedder,
    BaseLexicalIndex,
    BaseRetriever,
    BaseVectorStore,
)
from uwf_rag.core.fusion import reciprocal_rank_fusion, weighted_score_fusion
from uwf_rag.core.registry import registry
from uwf_rag.core.types import RetrievedChunk, TransformedQuery

logger = logging.getLogger(__name__)


@registry.register("retrieval", "dense")
class DenseRetriever(BaseRetriever):
    """Dense (embedding-based) retriever.

    Embeds the query, searches the vectorstore, returns top_k results.
    Supports metadata filtering via default_filters (from config) merged
    with per-call filters. The vectorstore + embedder are injected at
    construction.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        vectorstore: BaseVectorStore,
        embedder: BaseEmbedder,
        default_filters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, default_filters=default_filters)
        self._vectorstore = vectorstore
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        merged_filters = {**self._default_filters, **(filters or {})}
        query_vector = self._embedder.embed_query(query)
        return self._vectorstore.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=merged_filters if merged_filters else None,
        )


@registry.register("retrieval", "bm25")
class BM25Retriever(BaseRetriever):
    """Lexical retriever delegating to a sparse index (e.g. BM25).

    The index — owned by the indexing pipeline and living in
    ``IndexArtifact.auxiliary_stores`` — is injected at construction. No
    embedder is needed.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        sparse_index: BaseLexicalIndex,
        default_filters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, default_filters=default_filters)
        self._sparse_index = sparse_index

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        merged_filters = {**self._default_filters, **(filters or {})}
        return self._sparse_index.search(
            query=query,
            top_k=top_k,
            filters=merged_filters if merged_filters else None,
        )


@registry.register("retrieval", "hybrid")
class HybridRetriever(BaseRetriever):
    """Composes child retrievers and fuses their results.

    Holds an ordered list of named child retrievers and a fusion
    strategy. The default fusion is Reciprocal Rank Fusion (RRF),
    which sidesteps the cross-retriever score-normalization problem
    by ranking on positions only. Weighted score fusion is available
    as an opt-in variant for ablation studies.

    Filter semantics: the hybrid's ``default_filters`` (from
    ``query.retrieval.filters``) are merged with per-call filters and passed
    identically to every child at retrieve time, so per-branch over-retrieve +
    post-filter behavior stays consistent with the dense/BM25 paths.

    Children are constructed externally (in ``QueryPipeline.from_config``)
    and injected via the constructor along with their per-branch
    ``top_k`` depths and (for weighted fusion) names mapped to weights.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        children: list[tuple[str, BaseRetriever, int]] | None = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        normalize: str = "min_max",
        default_filters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, default_filters=default_filters)
        self._children: list[tuple[str, BaseRetriever, int]] = children or []
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._weights = weights or {}
        self._normalize = normalize
        self._method_label = f"hybrid_{fusion}"

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._children:
            raise RuntimeError(
                "HybridRetriever has no children. "
                "Construct with children=[(name, retriever, top_k), ...]."
            )

        merged_filters = {**self._default_filters, **(filters or {})}
        active_filters = merged_filters if merged_filters else None

        # Gather per-branch results. Maintain a chunk_id -> RetrievedChunk
        # map across branches so fused ids can be mapped back to chunks
        # without re-querying.
        per_branch_ids: list[list[str]] = []
        per_branch_scored: list[list[tuple[str, float]]] = []
        per_branch_names: list[str] = []
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for name, child, branch_top_k in self._children:
            results = child.retrieve(query, branch_top_k, active_filters)
            ids: list[str] = []
            scored: list[tuple[str, float]] = []
            for rc in results:
                cid = rc.chunk.chunk_id
                ids.append(cid)
                scored.append((cid, rc.score))
                # First occurrence wins for chunk lookup; identical
                # chunk objects across branches differ only in score,
                # which fusion is replacing anyway.
                chunk_lookup.setdefault(cid, rc)
            per_branch_ids.append(ids)
            per_branch_scored.append(scored)
            per_branch_names.append(name)

        # Fuse.
        if self._fusion == "rrf":
            fused = reciprocal_rank_fusion(per_branch_ids, k=self._rrf_k)
        elif self._fusion == "weighted":
            ordered_weights = [self._weights[n] for n in per_branch_names]
            fused = weighted_score_fusion(
                per_branch_scored,
                ordered_weights,
                normalize=self._normalize,
            )
        else:
            raise ValueError(
                f"HybridRetriever: unknown fusion mode '{self._fusion}' "
                f"(expected 'rrf' or 'weighted')"
            )

        # Rebind to RetrievedChunk objects with fused scores and a
        # method label that names the fusion, not whatever a child set.
        out: list[RetrievedChunk] = []
        for cid, fused_score in fused[:top_k]:
            base = chunk_lookup[cid]
            out.append(
                RetrievedChunk(
                    chunk=base.chunk,
                    score=fused_score,
                    retrieval_method=self._method_label,
                )
            )
        return out

    def retrieve_multi(
        self,
        transformed_queries: list[TransformedQuery],
        top_k_per_query: int,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Route transformed queries to children by branch, fuse two levels.

        Dispatch rules:
        - ``branch=None`` queries broadcast to every child (multi-query
          fan-out across all retrieval methods).
        - ``branch=<child name>`` queries go only to the matching child
          (HyDE's hypothetical doc to dense, original to BM25).
        - ``branch=<unknown name>`` raises ``ValueError`` — silently
          dropping would hide configuration mistakes.

        Two-level fusion:
        - **Intra-branch**: each child's queries are fused via ``fusion``
          (the pipeline-level setting). This is the multi-query fusion
          when a child sees more than one query.
        - **Cross-branch**: per-branch fused lists are combined via the
          hybrid's own ``self._fusion`` (``rrf`` or ``weighted``) — the
          retrieval-method fusion configured on the hybrid.

        Two-level is *not* equivalent to a single flat RRF over all
        (query, retriever) pairs: flat RRF would silently bias toward
        whichever branch had more queries; two-level preserves the
        hybrid's intended branch weighting regardless of how many
        reformulations were issued.
        """
        if not self._children:
            raise RuntimeError(
                "HybridRetriever has no children. "
                "Construct with children=[(name, retriever, top_k), ...]."
            )
        if not transformed_queries:
            return []

        merged_filters = {**self._default_filters, **(filters or {})}
        active_filters = merged_filters if merged_filters else None

        known_branches = {name for name, _, _ in self._children}
        per_child_queries: dict[str, list[TransformedQuery]] = {
            name: [] for name in known_branches
        }
        unrouted: list[TransformedQuery] = []
        unknown_branches: set[str] = set()

        for tq in transformed_queries:
            if tq.branch is None:
                unrouted.append(tq)
            elif tq.branch in known_branches:
                per_child_queries[tq.branch].append(tq)
            else:
                unknown_branches.add(tq.branch)

        if unknown_branches:
            raise ValueError(
                f"HybridRetriever: TransformedQuery references unknown "
                f"branch(es) {sorted(unknown_branches)}; known children "
                f"are {sorted(known_branches)}"
            )

        # Broadcast unrouted queries to every child.
        for name in known_branches:
            per_child_queries[name].extend(unrouted)

        per_branch_ids: list[list[str]] = []
        per_branch_scored: list[list[tuple[str, float]]] = []
        per_branch_names: list[str] = []
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for name, child, branch_top_k in self._children:
            queries_for_child = per_child_queries[name]
            if not queries_for_child:
                continue
            results = child.retrieve_multi(
                queries_for_child,
                top_k_per_query=branch_top_k,
                fusion=fusion,
                filters=active_filters,
            )
            ids: list[str] = []
            scored: list[tuple[str, float]] = []
            for rc in results:
                cid = rc.chunk.chunk_id
                ids.append(cid)
                scored.append((cid, rc.score))
                chunk_lookup.setdefault(cid, rc)
            per_branch_ids.append(ids)
            per_branch_scored.append(scored)
            per_branch_names.append(name)

        if not per_branch_ids:
            return []

        if self._fusion == "rrf":
            fused = reciprocal_rank_fusion(per_branch_ids, k=self._rrf_k)
        elif self._fusion == "weighted":
            ordered_weights = [self._weights[n] for n in per_branch_names]
            fused = weighted_score_fusion(
                per_branch_scored,
                ordered_weights,
                normalize=self._normalize,
            )
        else:
            raise ValueError(
                f"HybridRetriever: unknown fusion mode '{self._fusion}' "
                f"(expected 'rrf' or 'weighted')"
            )

        out: list[RetrievedChunk] = []
        for cid, fused_score in fused[:top_k_per_query]:
            base = chunk_lookup[cid]
            out.append(
                RetrievedChunk(
                    chunk=base.chunk,
                    score=fused_score,
                    retrieval_method=self._method_label,
                )
            )
        return out
