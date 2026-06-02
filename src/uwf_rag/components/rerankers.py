"""Reranker implementations.

Each reranker re-scores retrieved chunks and truncates to top_k.
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from uwf_rag.components.base import BaseReranker
from uwf_rag.core.registry import registry
from uwf_rag.core.types import RetrievedChunk

logger = logging.getLogger(__name__)


@registry.register("reranking", "cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Re-scores chunks using a cross-encoder model.

    Encodes (query, chunk) pairs jointly and scores relevance.
    More accurate than bi-encoder similarity but slower — intended
    as a second-stage reranker over a small candidate set.

    Requires a regression-style reranker model (``num_labels=1``)
    that returns a single scalar relevance score per (query, chunk)
    pair.  Classification cross-encoders (``num_labels > 1``) are
    NOT supported — ``predict`` returns a per-class array in that
    case and ``float(s)`` below will raise ``TypeError`` at query
    time.  When swapping ``model_name``, verify the model is a
    regression reranker (e.g. its HF model card / ``config.json``
    has ``num_labels: 1``) before running an experiment.

    Config params:
        model_name: HuggingFace model ID for a regression
            cross-encoder (default:
            "Alibaba-NLP/gte-reranker-modernbert-base")
        device: "cuda", "cpu", or null for auto-detect (default: null)
        batch_size: Pairs per forward pass (default: 32)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        model_name = self.config.get(
            "model_name", "Alibaba-NLP/gte-reranker-modernbert-base"
        )
        device = self.config.get("device")
        self._batch_size: int = self.config.get("batch_size", 32)

        logger.info("Loading cross-encoder model: %s", model_name)
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Score each chunk against the query and return the top_k highest."""
        if not chunks:
            return []

        pairs = [(query, rc.chunk.content) for rc in chunks]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        scored = [
            RetrievedChunk(
                chunk=rc.chunk,
                score=float(s),
                retrieval_method=rc.retrieval_method,
            )
            for rc, s in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)

        logger.info(
            "Cross-encoder reranked %d chunks, returning top %d",
            len(scored),
            min(top_k, len(scored)),
        )
        return scored[:top_k]
