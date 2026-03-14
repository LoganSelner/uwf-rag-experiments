"""Retriever implementations.

Each class satisfies the ``Retriever`` Protocol defined in ``.base``.

Implementations
---------------
SimpleRetriever  — dense semantic similarity search via Chroma (baseline)
MMRRetriever     — maximal marginal relevance search via Chroma (relevance + diversity)
HybridRetriever  — BM25 + dense search fused via weighted Reciprocal Rank Fusion

To add a new retriever:
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
from rank_bm25 import BM25Okapi

from ..config import Settings


@dataclass(frozen=True, eq=False)  # eq=False: Chroma is not hashable
class SimpleRetriever:
    """Dense semantic similarity search against a Chroma vector store."""

    store: Chroma
    k: int

    def retrieve(self, question: str) -> list[Document]:
        return self.store.similarity_search(question, k=self.k)

    @classmethod
    def from_settings(cls, settings: Settings, store: Chroma) -> SimpleRetriever:
        return cls(store=store, k=settings.eval.retrieval_k)


@dataclass(frozen=True, eq=False)  # eq=False: Chroma is not hashable
class MMRRetriever:
    """Maximal marginal relevance search against a Chroma vector store.

    Balances relevance to the query with diversity among returned documents.
    ``lambda_mult`` controls the tradeoff: ``1.0`` = pure relevance (equivalent
    to dense), ``0.0`` = maximum diversity.  ``fetch_k`` is the initial candidate
    pool size that MMR selects from; configurable via ``mmr_fetch_k`` in
    eval settings (default: 20, matching LangChain's default).
    """

    store: Chroma
    k: int
    fetch_k: int
    lambda_mult: float

    def retrieve(self, question: str) -> list[Document]:
        return self.store.max_marginal_relevance_search(
            question,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

    @classmethod
    def from_settings(cls, settings: Settings, store: Chroma) -> MMRRetriever:
        return cls(
            store=store,
            k=settings.eval.retrieval_k,
            fetch_k=settings.eval.mmr_fetch_k,
            lambda_mult=settings.eval.mmr_lambda,
        )


# ---------------------------------------------------------------------------
# Hybrid helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer for BM25."""
    return text.lower().split()


def _reciprocal_rank_fusion(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    alpha: float,
    k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse two ranked lists using weighted Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_docs:
        Ranked results from dense retriever (position 0 = best).
    bm25_docs:
        Ranked results from BM25 (position 0 = best).
    alpha:
        Blend weight.  ``1.0`` = dense only, ``0.0`` = BM25 only.
    k:
        Number of results to return.
    rrf_k:
        RRF constant (default 60, per Cormack et al.).
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + alpha / (rrf_k + rank + 1)
        doc_lookup[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + (1 - alpha) / (rrf_k + rank + 1)
        if key not in doc_lookup:
            doc_lookup[key] = doc

    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [doc_lookup[c] for c in ranked[:k]]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HybridRetriever:
    """Hybrid retrieval combining dense vector search with BM25 keyword search.

    Results from both retrievers are fused using weighted Reciprocal Rank Fusion
    (RRF).  ``alpha`` controls the blend: ``1.0`` = dense only, ``0.0`` = BM25
    only, ``0.5`` = equal weight (default).

    The BM25 index is built at construction time from all documents stored in the
    Chroma collection (retrieved via ``store.get()``).
    """

    store: Chroma
    bm25: BM25Okapi
    corpus_docs: tuple[Document, ...]
    k: int
    alpha: float

    def retrieve(self, question: str) -> list[Document]:
        dense_results = self.store.similarity_search(question, k=self.k)

        tokenized_query = _tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[::-1][: self.k]
        bm25_results = [self.corpus_docs[i] for i in top_indices]

        return _reciprocal_rank_fusion(
            dense_results, bm25_results, alpha=self.alpha, k=self.k
        )

    @classmethod
    def from_settings(cls, settings: Settings, store: Chroma) -> HybridRetriever:
        stored = store.get()
        contents: list[str] = stored["documents"]
        metadatas: list[dict[str, str]] = stored.get("metadatas") or [{}] * len(
            contents
        )
        corpus_docs = tuple(
            Document(page_content=c, metadata=m or {})
            for c, m in zip(contents, metadatas, strict=True)
        )
        tokenized_corpus = [_tokenize(doc.page_content) for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(
            store=store,
            bm25=bm25,
            corpus_docs=corpus_docs,
            k=settings.eval.retrieval_k,
            alpha=settings.eval.hybrid_alpha,
        )
