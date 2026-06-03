"""Lexical (sparse) index implementations.

Each lexical index stores chunks and provides term-frequency-based
search. Parallel to the vectorstore layer but for sparse retrievers
that operate on raw text rather than dense embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pickle
from typing import Any

import bm25s

from ragbench.components.base import BaseLexicalIndex, ComponentParams
from ragbench.core.registry import registry
from ragbench.core.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_VALID_BM25_METHODS = frozenset({"lucene", "robertson", "atire", "bm25l", "bm25+"})
_META_VERSION = 1


@registry.register("sparse_index", "bm25")
class BM25LexicalIndex(BaseLexicalIndex):
    """BM25 lexical index backed by ``bm25s``.

    Owns its own tokenizer + stemmer + parameters. Query-time
    tokenization mirrors index-time tokenization automatically.

    Metadata filtering is post-retrieval: over-retrieve (configurable
    multiplier), filter, truncate. BM25 has no native filter pushdown,
    and lexical relevance correlates with metadata less than semantic
    relevance does, so the multiplier may need tuning per corpus.

    Config params:
        k1: Saturation parameter (default 1.5).
        b: Length-normalization parameter (default 0.75).
        delta: Used by ``bm25l`` and ``bm25+`` (default 0.5).
        method: One of "lucene" (default), "robertson", "atire",
            "bm25l", "bm25+".
        stopwords: ``"en"``/``"english"``, a custom list, or ``null``
            to disable (default ``"en"``).
        stemmer: PyStemmer language name (default ``"english"``), or
            ``null`` to disable stemming.
        filter_overretrieve: Multiplier on ``top_k`` when filtering
            (default 4). Floor of 50 matches the FAISS pattern.
    """

    class Params(ComponentParams):
        k1: float = 1.5
        b: float = 0.75
        delta: float = 0.5
        method: str = "lucene"
        stopwords: str | list[str] | None = "en"
        stemmer: str | None = "english"
        filter_overretrieve: int = 4

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)

        if self.p.method not in _VALID_BM25_METHODS:
            raise ValueError(
                f"BM25LexicalIndex method must be one of "
                f"{sorted(_VALID_BM25_METHODS)}, got '{self.p.method}'"
            )

        self._k1 = self.p.k1
        self._b = self.p.b
        self._delta = self.p.delta
        self._method = self.p.method
        self._stopwords = self.p.stopwords
        self._stemmer_lang = self.p.stemmer
        self._filter_overretrieve = self.p.filter_overretrieve

        self._stemmer = self._build_stemmer(self._stemmer_lang)
        self._retriever: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []

    @staticmethod
    def _build_stemmer(lang: str | None) -> Any:
        if not lang:
            return None
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer(lang)

    def _tokenize(self, texts: list[str]) -> Any:
        """Tokenize using bm25s with the index's configured stopwords + stemmer."""
        return bm25s.tokenize(
            texts,
            stopwords=self._stopwords,
            stemmer=self._stemmer,
            return_ids=True,
            show_progress=False,
            leave=False,
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Tokenize chunk content and build the BM25 index.

        Builds the index in one shot — bm25s does not support
        incremental indexing. Subsequent calls re-index from scratch
        with the union of stored and new chunks.
        """
        if not chunks:
            if self._retriever is None and not self._chunks:
                return
            chunks = []

        # bm25s does not support add-without-rebuild; accumulate and
        # always re-index from the full corpus.
        self._chunks.extend(chunks)
        if not self._chunks:
            return

        texts = [c.text_for_index for c in self._chunks]
        tokenized = self._tokenize(texts)

        retriever = bm25s.BM25(
            k1=self._k1,
            b=self._b,
            delta=self._delta,
            method=self._method,
        )
        retriever.index(tokenized, show_progress=False, leave_progress=False)
        self._retriever = retriever

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._retriever is None or not self._chunks:
            return []

        if filters:
            return self._filtered_search(query, top_k, filters)

        query_tokens = self._tokenize([query])
        n_results = min(top_k, len(self._chunks))
        docs, scores = self._retriever.retrieve(
            query_tokens,
            k=n_results,
            show_progress=False,
            leave_progress=False,
        )

        results: list[RetrievedChunk] = []
        for idx, score in zip(docs[0], scores[0], strict=True):
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(idx)],
                    score=float(score),
                    retrieval_method="bm25",
                )
            )
        return results

    def _filtered_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """Over-retrieve, apply metadata filters, truncate to top_k.

        Mirrors ``FAISSVectorStore._filtered_search`` — BM25 has no
        native metadata filtering. Multiplier defaults to 4x (FAISS's
        value); ``filter_overretrieve`` config raises it when lexical-
        metadata correlation is poor on a given corpus.
        """
        if self._retriever is None or not self._chunks:
            return []

        fetch_k = min(
            max(top_k * self._filter_overretrieve, 50),
            len(self._chunks),
        )
        query_tokens = self._tokenize([query])
        docs, scores = self._retriever.retrieve(
            query_tokens,
            k=fetch_k,
            show_progress=False,
            leave_progress=False,
        )

        results: list[RetrievedChunk] = []
        for idx, score in zip(docs[0], scores[0], strict=True):
            chunk = self._chunks[int(idx)]
            if self._matches_filters(chunk, filters):
                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=float(score),
                        retrieval_method="bm25",
                    )
                )
                if len(results) >= top_k:
                    break
        return results

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    def save(self, directory: str) -> None:
        if self._retriever is None:
            raise RuntimeError("Cannot save — BM25 index is not initialized")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self._retriever.save(str(path / "bm25s"), show_progress=False)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        meta = {
            "version": _META_VERSION,
            "k1": self._k1,
            "b": self._b,
            "delta": self._delta,
            "method": self._method,
            "stopwords": self._stopwords,
            "stemmer": self._stemmer_lang,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, directory: str) -> None:
        path = Path(directory)

        self._retriever = bm25s.BM25.load(str(path / "bm25s"), mmap=True)
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        with open(path / "meta.pkl", "rb") as f:
            meta: dict[str, Any] = pickle.load(f)

        # Restore tokenizer state to whatever the index was built with.
        # Critical for query-time tokenization parity.
        self._k1 = meta["k1"]
        self._b = meta["b"]
        self._delta = meta["delta"]
        self._method = meta["method"]
        self._stopwords = meta["stopwords"]
        self._stemmer_lang = meta["stemmer"]
        self._stemmer = self._build_stemmer(self._stemmer_lang)
