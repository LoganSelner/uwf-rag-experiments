"""Retriever implementations.

Each class satisfies the ``Retriever`` Protocol defined in ``.base``.

Implementations
---------------
SimpleRetriever — dense semantic similarity search via Chroma (baseline)
MMRRetriever    — maximal marginal relevance search via Chroma (relevance + diversity)

To add a new retriever (e.g. HybridRetriever):
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


@dataclass(frozen=True, eq=False)  # eq=False: Chroma is not hashable
class MMRRetriever:
    """Maximal marginal relevance search against a Chroma vector store.

    Balances relevance to the query with diversity among returned documents.
    ``lambda_mult`` controls the tradeoff: ``1.0`` = pure relevance (equivalent
    to dense), ``0.0`` = maximum diversity.  ``fetch_k`` is the initial candidate
    pool size that MMR selects from; it is computed automatically by
    ``from_settings`` as ``max(20, 4 * retrieval_k)``.
    """

    store: Chroma
    top_k: int
    fetch_k: int
    lambda_mult: float

    def retrieve(self, question: str) -> list[Document]:
        return self.store.max_marginal_relevance_search(
            question,
            k=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

    @classmethod
    def from_settings(cls, settings: Settings, store: Chroma) -> MMRRetriever:
        k = settings.eval.retrieval_k
        return cls(
            store=store,
            top_k=k,
            fetch_k=max(20, 4 * k),
            lambda_mult=settings.eval.mmr_lambda,
        )
