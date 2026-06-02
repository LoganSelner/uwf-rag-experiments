"""Retriever implementations.

Each retriever uses an embedder + vectorstore (dense), a sparse
index (lexical), or composes child retrievers (hybrid) to find
relevant chunks for a query.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uwf_rag.components.base import (
    BaseEmbedder,
    BaseLexicalIndex,
    BaseRetriever,
    BaseVectorStore,
)
from uwf_rag.components.build import BM25_AUX_KEY
from uwf_rag.core.fusion import reciprocal_rank_fusion, weighted_score_fusion
from uwf_rag.core.registry import registry
from uwf_rag.core.types import RetrievedChunk, TransformedQuery

if TYPE_CHECKING:
    from uwf_rag.components.build import BuildContext
    from uwf_rag.core.config import ValidateContext

logger = logging.getLogger(__name__)

_SUPPORTED_FUSION_MODES: frozenset[str] = frozenset({"rrf", "weighted"})
_SUPPORTED_NORMALIZE_MODES: frozenset[str] = frozenset({"min_max", "none"})


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

    @classmethod
    def build(
        cls,
        *,
        params: dict[str, Any],
        default_filters: dict[str, Any] | None,
        ctx: BuildContext,
    ) -> DenseRetriever:
        if ctx.index is None:
            raise RuntimeError(
                "DenseRetriever.build requires a populated index in the "
                "BuildContext (ctx.index is None)."
            )
        return cls(
            config=params,
            vectorstore=ctx.index.vectorstore,
            embedder=ctx.index.embedder,
            default_filters=default_filters,
        )

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

    @classmethod
    def build(
        cls,
        *,
        params: dict[str, Any],
        default_filters: dict[str, Any] | None,
        ctx: BuildContext,
    ) -> BM25Retriever:
        sparse = ctx.index.auxiliary_stores.get(BM25_AUX_KEY) if ctx.index else None
        if sparse is None:
            raise RuntimeError(
                "BM25 retriever requires a sparse index, but the index has none "
                f"under auxiliary_stores['{BM25_AUX_KEY}']. Ensure "
                "indexing.sparse_index is configured."
            )
        if not isinstance(sparse, BaseLexicalIndex):
            raise RuntimeError(
                f"auxiliary_stores['{BM25_AUX_KEY}'] is not a BaseLexicalIndex"
            )
        return cls(
            config=params,
            sparse_index=sparse,
            default_filters=default_filters,
        )

    @classmethod
    def validate_params(
        cls,
        params: dict[str, Any],
        ctx: ValidateContext,
    ) -> list[str]:
        if not ctx.sparse_index_type:
            return [
                "query.retrieval.type is 'bm25' but indexing.sparse_index "
                "is not configured"
            ]
        return []

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

    @classmethod
    def build(
        cls,
        *,
        params: dict[str, Any],
        default_filters: dict[str, Any] | None,
        ctx: BuildContext,
    ) -> HybridRetriever:
        """Build the hybrid + its child sub-retrievers, recursing through ``ctx``.

        Each child spec is ``{name, type, top_k, params}``; the child is built
        via ``ctx.registry.get("retrieval", type).build(...)`` so the dispatch
        lives in the registry, not here. The hybrid's ``default_filters`` are
        injected into every child too, keeping per-branch filter semantics
        consistent with the standalone retrievers.
        """
        children: list[tuple[str, BaseRetriever, int]] = []
        for spec in params.get("retrievers", []):
            sub_cls = ctx.registry.get("retrieval", spec["type"])
            child = sub_cls.build(
                params=spec.get("params", {}),
                default_filters=default_filters,
                ctx=ctx,
            )
            children.append((spec.get("name", spec["type"]), child, int(spec["top_k"])))

        return cls(
            config=params,
            children=children,
            fusion=params.get("fusion", "rrf"),
            rrf_k=int(params.get("rrf_k", 60)),
            weights={k: float(v) for k, v in params.get("weights", {}).items()},
            normalize=params.get("normalize", "min_max"),
            default_filters=default_filters,
        )

    @classmethod
    def validate_params(
        cls,
        params: dict[str, Any],
        ctx: ValidateContext,
    ) -> list[str]:
        """Validate a hybrid's raw ``params`` (child roster + fusion settings).

        Owns the structural checks that used to live in core's
        ``validate_config``: the child roster shape, no nested hybrid, unique
        names, per-child positive top_k, the bm25-child-needs-sparse-index
        rule, fusion mode + rrf_k, and the weighted-fusion weights. The
        cross-config context (registry, configured sparse-index type, retrieve
        budget) arrives via ``ctx``.
        """
        registry = ctx.registry
        sparse_index_type = ctx.sparse_index_type
        top_k_retrieve = ctx.top_k_retrieve
        errors: list[str] = []
        params = params or {}
        children = params.get("retrievers")
        if not isinstance(children, list):
            errors.append(
                "query.retrieval.params.retrievers must be a list of "
                "sub-retriever specs for hybrid retrieval"
            )
            return errors
        if len(children) < 2:
            errors.append(
                f"query.retrieval.params.retrievers must have at least 2 "
                f"entries for hybrid (got {len(children)})"
            )

        seen_names: set[str] = set()
        seen_types: dict[str, int] = {}
        has_bm25_child = False
        child_top_k_sum = 0

        for i, child in enumerate(children):
            path = f"query.retrieval.params.retrievers[{i}]"
            if not isinstance(child, dict):
                errors.append(f"{path} must be a dict")
                continue
            sub_type = child.get("type", "")
            if sub_type == "hybrid":
                errors.append(f"{path}.type: nested 'hybrid' is not allowed")
                continue
            if not sub_type:
                errors.append(f"{path}.type is empty")
            else:
                if not registry.is_registered("retrieval", sub_type):
                    available = registry.list_category("retrieval")
                    errors.append(
                        f"{path}.type: '{sub_type}' is not registered in "
                        f"category 'retrieval'. Available: {available}"
                    )
                seen_types[sub_type] = seen_types.get(sub_type, 0) + 1
                if sub_type == "bm25":
                    has_bm25_child = True

            sub_name = str(child.get("name", sub_type))
            if sub_name in seen_names:
                errors.append(
                    f"{path}.name: '{sub_name}' is duplicated in this hybrid "
                    "(child names must be unique)"
                )
            else:
                seen_names.add(sub_name)

            sub_top_k = child.get("top_k")
            if not isinstance(sub_top_k, int) or sub_top_k <= 0:
                errors.append(
                    f"{path}.top_k must be a positive int (got {sub_top_k!r})"
                )
            else:
                child_top_k_sum += sub_top_k

        # Duplicate types without explicit names are disallowed — the
        # default name (the type) would collide.
        for sub_type, count in seen_types.items():
            if count > 1:
                explicit = sum(
                    1
                    for c in children
                    if isinstance(c, dict)
                    and c.get("type") == sub_type
                    and c.get("name")
                )
                if explicit < count:
                    errors.append(
                        f"query.retrieval.params.retrievers: type '{sub_type}' "
                        f"appears {count} times — each duplicate child must "
                        "supply an explicit unique 'name'"
                    )

        if child_top_k_sum and top_k_retrieve > 0 and child_top_k_sum < top_k_retrieve:
            errors.append(
                f"query.retrieval.params.retrievers: sum of child top_k "
                f"({child_top_k_sum}) < query.retrieval.top_k_retrieve "
                f"({top_k_retrieve}) — fusion cannot fill the slate"
            )

        if has_bm25_child and sparse_index_type != "bm25":
            errors.append(
                "hybrid has a 'bm25' child but indexing.sparse_index.type "
                f"is '{sparse_index_type}' (expected 'bm25')"
            )

        fusion = params.get("fusion", "rrf")
        if fusion not in _SUPPORTED_FUSION_MODES:
            errors.append(
                f"query.retrieval.params.fusion: '{fusion}' is not supported. "
                f"Supported: {sorted(_SUPPORTED_FUSION_MODES)}"
            )

        if fusion == "rrf":
            rrf_k = params.get("rrf_k", 60)
            if not isinstance(rrf_k, int) or rrf_k <= 0:
                errors.append(
                    f"query.retrieval.params.rrf_k must be a positive int "
                    f"(got {rrf_k!r})"
                )

        if fusion == "weighted":
            normalize = params.get("normalize", "min_max")
            if normalize not in _SUPPORTED_NORMALIZE_MODES:
                errors.append(
                    f"query.retrieval.params.normalize: '{normalize}' is not "
                    f"supported. Supported: {sorted(_SUPPORTED_NORMALIZE_MODES)}"
                )

            weights = params.get("weights")
            if not isinstance(weights, dict):
                errors.append(
                    "query.retrieval.params.weights is required for weighted "
                    "fusion and must be a dict keyed by sub-retriever name"
                )
            else:
                if set(weights.keys()) != seen_names:
                    errors.append(
                        f"query.retrieval.params.weights keys "
                        f"{sorted(weights.keys())} must match sub-retriever "
                        f"names {sorted(seen_names)}"
                    )
                weight_values = list(weights.values())
                if any(not isinstance(w, (int, float)) or w < 0 for w in weight_values):
                    errors.append(
                        "query.retrieval.params.weights values must be "
                        f"non-negative numbers (got {weight_values})"
                    )
                elif sum(weight_values) <= 0:
                    errors.append(
                        "query.retrieval.params.weights must sum to > 0 "
                        f"(got {sum(weight_values)})"
                    )

        return errors

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

        # Gather per-branch results, then fuse + rebind via the shared helper.
        per_branch_ids: list[list[str]] = []
        per_branch_scored: list[list[tuple[str, float]]] = []
        per_branch_names: list[str] = []
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for name, child, branch_top_k in self._children:
            results = child.retrieve(query, branch_top_k, active_filters)
            self._accumulate_branch(
                name,
                results,
                per_branch_ids,
                per_branch_scored,
                per_branch_names,
                chunk_lookup,
            )

        return self._fuse_branches(
            per_branch_ids, per_branch_scored, per_branch_names, chunk_lookup, top_k
        )

    @staticmethod
    def _accumulate_branch(
        name: str,
        results: list[RetrievedChunk],
        per_branch_ids: list[list[str]],
        per_branch_scored: list[list[tuple[str, float]]],
        per_branch_names: list[str],
        chunk_lookup: dict[str, RetrievedChunk],
    ) -> None:
        """Record one branch's rank list into the parallel fusion inputs.

        First occurrence wins for chunk lookup; identical chunk objects across
        branches differ only in score, which fusion is replacing anyway.
        """
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

    def _fuse_branches(
        self,
        per_branch_ids: list[list[str]],
        per_branch_scored: list[list[tuple[str, float]]],
        per_branch_names: list[str],
        chunk_lookup: dict[str, RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Cross-branch fusion + rebind shared by ``retrieve`` / ``retrieve_multi``.

        Combines the per-branch rank lists via the hybrid's own ``self._fusion``
        (RRF or weighted), then rebinds the fused ids to ``RetrievedChunk``
        objects carrying the fused score and the hybrid's method label (not
        whatever a child set), truncated to ``top_k``.
        """
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
            self._accumulate_branch(
                name,
                results,
                per_branch_ids,
                per_branch_scored,
                per_branch_names,
                chunk_lookup,
            )

        return self._fuse_branches(
            per_branch_ids,
            per_branch_scored,
            per_branch_names,
            chunk_lookup,
            top_k_per_query,
        )
