"""Reranker implementations.

Each class satisfies the ``Reranker`` Protocol defined in ``.base``.

Implementations
---------------
NoReranker — identity pass-through; returns candidates unchanged (current baseline)

To add a new reranker (e.g. CrossEncoderReranker, CohereReranker):
  1. Add a class here satisfying the ``Reranker`` Protocol (``rerank`` method) and
     add a ``from_settings(cls, settings: Settings) -> <class>`` classmethod.
  2. Export it from ``__init__.py``.
  3. Add a branch in ``pipeline._build_reranker()`` for the new ``reranker_type``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from ..config import Settings


@dataclass(frozen=True)
class NoReranker:
    """Identity reranker: returns the candidate list unchanged."""

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        return docs

    @classmethod
    def from_settings(cls, settings: Settings) -> NoReranker:
        return cls()
