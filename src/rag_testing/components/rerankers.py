"""Reranker implementations.

Each class satisfies the ``Reranker`` Protocol defined in ``.base``.

Implementations
---------------
NoReranker            — identity pass-through; returns candidates unchanged (baseline)
CrossEncoderReranker  — reorders candidates using a sentence-transformers CrossEncoder

To add a new reranker (e.g. CohereReranker):
  1. Add a class here satisfying the ``Reranker`` Protocol (``rerank`` method) and
     add a ``from_settings(cls, settings: Settings) -> <class>`` classmethod.
  2. Export it from ``__init__.py``.
  3. Add a branch in ``pipeline._build_reranker()`` for the new ``reranker_type``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..config import Settings


@dataclass(frozen=True)
class NoReranker:
    """Identity reranker: returns the candidate list unchanged."""

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        return docs

    @classmethod
    def from_settings(cls, settings: Settings) -> NoReranker:
        return cls()


@dataclass(frozen=True, eq=False)  # eq=False: CrossEncoder is not hashable
class CrossEncoderReranker:
    """Reranks documents using a sentence-transformers CrossEncoder model.

    Scores each ``(question, doc.page_content)`` pair and returns the full
    candidate list sorted by descending relevance score.  Truncation to
    ``top_k`` is handled by the pipeline, not by this component.
    """

    model: CrossEncoder

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(
            zip(scores, docs, strict=True),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        return [doc for _, doc in scored]

    @classmethod
    def from_settings(cls, settings: Settings) -> CrossEncoderReranker:
        model_name = settings.eval.reranker_model
        if not model_name:
            raise ValueError(
                "reranker_model must be set when reranker_type is 'cross_encoder'"
            )
        return cls(model=CrossEncoder(model_name))
