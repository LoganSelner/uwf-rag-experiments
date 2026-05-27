"""Rank-list fusion helpers.

Pure data manipulation — no domain types, no I/O. Used by the hybrid
retriever in Phase A; will be reused by multi-query fusion in Phase B.

Two strategies are supported:

- **Reciprocal Rank Fusion (RRF)**: rank-only, scale-agnostic.
  ``score(d) = Σ 1 / (k + rank_i(d))`` across all lists that contain
  ``d``. The production default across OpenSearch, Elasticsearch,
  Azure AI Search, MongoDB Atlas, Weaviate, and Chroma; works well
  zero-shot with ``k=60`` because it sidesteps the cross-retriever
  score-normalization problem entirely.

- **Weighted score fusion**: combines per-retriever scores into a
  single weighted sum. Each list is independently normalized first
  (min-max by default) to bring scores onto a comparable scale.
  Useful for ablating against RRF, but more brittle in practice
  because optimal weights are query- and corpus-dependent.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple rank-ordered id lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a sequence of ids in rank
            order (best first). Lists may have different lengths and
            need not contain the same ids. Empty inner lists are
            allowed.
        k: The RRF constant. ``k=60`` is the cross-vendor default;
            values in ``[40, 80]`` perform comparably in benchmarks.
            Must be positive.

    Returns:
        A list of ``(id, fused_score)`` tuples sorted by score
        descending. The returned scores are small (e.g. a top-1 hit
        in two lists with ``k=60`` scores ~0.033) — do not compare
        them across different ``k`` values or against raw retriever
        scores.

    Raises:
        ValueError: If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be > 0, got {k}")

    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def weighted_score_fusion(
    scored_lists: list[list[tuple[str, float]]],
    weights: list[float],
    normalize: str = "min_max",
) -> list[tuple[str, float]]:
    """Fuse multiple scored id lists via a weighted sum after normalization.

    Each list is independently rescaled to ``[0, 1]`` (when
    ``normalize == "min_max"``) before the weighted sum, so retrievers
    with incompatible score ranges (cosine in ``[-1, 1]``, BM25
    unbounded) can be combined sanely. Weights need not sum to 1;
    they are normalized internally so the relative ratios are what
    matter.

    Args:
        scored_lists: Each inner list is ``(id, score)`` tuples in any
            order. Lists may have different lengths and overlapping
            ids. Empty inner lists are allowed.
        weights: One non-negative weight per list. Must be the same
            length as ``scored_lists`` and have ``sum(weights) > 0``.
        normalize: Per-list normalization strategy. ``"min_max"``
            rescales each list to ``[0, 1]``. ``"none"`` uses raw
            scores (only safe when all retrievers produce comparable
            scales — rarely true in practice).

    Returns:
        ``(id, fused_score)`` tuples sorted by score descending. An id
        missing from a given list contributes 0 from that branch.

    Raises:
        ValueError: If lengths mismatch, weights are negative, weight
            sum is non-positive, or ``normalize`` is unknown.
    """
    if len(scored_lists) != len(weights):
        raise ValueError(
            f"weighted_score_fusion: got {len(scored_lists)} lists but "
            f"{len(weights)} weights — they must match"
        )
    if any(w < 0 for w in weights):
        raise ValueError(f"weighted_score_fusion: weights must be >= 0, got {weights}")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError(
            f"weighted_score_fusion: sum(weights) must be > 0, got {weight_sum}"
        )
    if normalize not in {"min_max", "none"}:
        raise ValueError(
            f"weighted_score_fusion: normalize must be 'min_max' or 'none', "
            f"got {normalize!r}"
        )

    normalized_weights = [w / weight_sum for w in weights]
    fused: dict[str, float] = {}

    for scored, weight in zip(scored_lists, normalized_weights, strict=True):
        if not scored or weight == 0.0:
            continue
        rescaled = _rescale(scored, normalize)
        for doc_id, score in rescaled:
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * score

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def max_score_dedup(
    scored_lists: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Deduplicate (id, score) pairs across lists, keeping the highest score.

    Used by ``BaseRetriever.retrieve_multi`` when the pipeline-level
    ``fusion="max"`` is selected. The semantics match the legacy
    ``QueryPipeline._retrieve_and_deduplicate`` behavior preserved for
    backward compatibility and as an ablation against RRF.

    Scores are only meaningful when all lists come from the *same*
    underlying retriever (same scoring function) — true by construction
    inside an intra-branch fan-out where each child runs the same N
    transformed queries through one retriever. For cross-retriever
    score combination, use ``weighted_score_fusion`` (with
    normalization) or ``reciprocal_rank_fusion`` (rank-only) instead.

    Args:
        scored_lists: Each inner list is ``(id, score)`` tuples in any
            order. Lists may have different lengths and overlapping ids.
            Empty inner lists are allowed.

    Returns:
        ``(id, fused_score)`` tuples sorted by score descending. Ties
        keep the first-encountered ordering.
    """
    seen: dict[str, float] = {}
    for scored in scored_lists:
        for doc_id, score in scored:
            if doc_id not in seen or score > seen[doc_id]:
                seen[doc_id] = score
    return sorted(seen.items(), key=lambda kv: kv[1], reverse=True)


def _rescale(
    scored: list[tuple[str, float]],
    normalize: str,
) -> list[tuple[str, float]]:
    """Apply per-list score normalization."""
    if normalize == "none":
        return scored

    # min_max
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi == lo:
        # All scores equal — collapse to 1.0 so they each contribute
        # their weight share rather than zero.
        return [(d, 1.0) for d, _ in scored]
    span = hi - lo
    return [(d, (s - lo) / span) for d, s in scored]
