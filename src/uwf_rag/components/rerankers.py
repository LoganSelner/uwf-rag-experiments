"""Reranker implementations.

Each reranker re-scores retrieved chunks and truncates to top_k.
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from uwf_rag.components.base import BaseReranker, ComponentParams
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

    class Params(ComponentParams):
        model_name: str = "Alibaba-NLP/gte-reranker-modernbert-base"
        device: str | None = None
        batch_size: int = 32

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._batch_size = self.p.batch_size

        logger.info("Loading cross-encoder model: %s", self.p.model_name)
        self._model = CrossEncoder(self.p.model_name, device=self.p.device)

        # Fail fast on a classification cross-encoder: predict() returns a
        # per-class array there, so float(score) below would blow up mid-run.
        raw_labels = getattr(getattr(self._model, "config", None), "num_labels", 1)
        try:
            num_labels = int(raw_labels)
        except (TypeError, ValueError):
            num_labels = 1  # unreadable (e.g. a test double) → assume regression
        if num_labels > 1:
            raise ValueError(
                f"CrossEncoderReranker requires a regression reranker "
                f"(num_labels=1), but '{self.p.model_name}' reports "
                f"num_labels={num_labels}. A classification cross-encoder returns "
                "a per-class score array, not a single relevance score. Use a "
                "regression reranker (e.g. Alibaba-NLP/gte-reranker-modernbert-base)."
            )

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
