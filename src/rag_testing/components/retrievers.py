"""Retriever implementations.

Each class satisfies the ``Retriever`` Protocol defined in ``.base``.

Implementations
---------------
SimpleRetriever — dense semantic similarity search via Chroma (current baseline)

To add a new retriever (e.g. HybridRetriever, MMRRetriever):
  1. Add a class here satisfying the ``Retriever`` Protocol (``retrieve`` method) and
     add a ``from_settings(cls, settings: Settings, store: Chroma) -> <class>``
     classmethod.
  2. Export it from ``__init__.py``.
  3. Add a branch in ``pipeline._build_retriever()`` for the new ``retriever_type``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import Settings


@dataclass(frozen=True, eq=False)  # eq=False: Chroma is not hashable
class SimpleRetriever:
    """Dense semantic similarity search against a Chroma vector store."""

    store: Chroma
    top_k: int

    def retrieve(self, question: str) -> list[Document]:
        return self.store.similarity_search(question, k=self.top_k)

    @classmethod
    def from_settings(cls, settings: Settings, store: Chroma) -> SimpleRetriever:
        return cls(store=store, top_k=settings.eval.retrieval_k)
