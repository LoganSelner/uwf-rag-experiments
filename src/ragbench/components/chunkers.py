"""Chunker implementations.

Each chunker splits Documents into Chunks with unique IDs and
preserved metadata.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import re
from typing import Any

import numpy as np

from ragbench.components.base import BaseChunker, ComponentParams
from ragbench.core.registry import registry
from ragbench.core.types import Chunk, Document


def _make_chunk_id(
    content: str,
    metadata: dict[str, Any],
    index: int,
) -> str:
    """Generate a deterministic chunk ID from content and position."""
    source = metadata.get("source_path", "")
    page = metadata.get("page_number", 0)
    key = f"{source}:{page}:{index}:{content[:64]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@registry.register("chunking", "recursive")
class LangChainRecursiveChunker(BaseChunker):
    """Recursive character text splitter (the default chunker).

    Wraps ``langchain_text_splitters.RecursiveCharacterTextSplitter`` and emits
    the project's own ``Chunk`` type, so nothing downstream touches LangChain.
    Registered as ``recursive``; the class keeps the ``LangChain`` prefix because
    the *implementation* is library-backed, but the config-facing name does not
    leak that.

    Config params:
        chunk_size: Target chunk size in characters (default: 512).
        chunk_overlap: Overlap between chunks (default: 50).
        separators: List of separator strings (default: LangChain's default).
    """

    class Params(ComponentParams):
        chunk_size: int = 512
        chunk_overlap: int = 50
        separators: list[str] | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs: dict[str, Any] = {
            "chunk_size": self.p.chunk_size,
            "chunk_overlap": self.p.chunk_overlap,
        }
        if self.p.separators is not None:
            kwargs["separators"] = self.p.separators

        self._splitter = RecursiveCharacterTextSplitter(
            keep_separator=False,
            **kwargs,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            lc_docs = self._splitter.create_documents(
                [doc.content],
                metadatas=[doc.metadata],
            )
            for i, lc_doc in enumerate(lc_docs):
                chunk_id = _make_chunk_id(lc_doc.page_content, lc_doc.metadata, i)
                chunks.append(
                    Chunk(
                        content=lc_doc.page_content,
                        chunk_id=chunk_id,
                        metadata={**lc_doc.metadata, "chunk_index": i},
                    )
                )
        return chunks


_DEFAULT_SENTENCE_REGEX = r"(?<=[.?!])\s+"

# Default statistic amount per threshold type (matches the de-facto
# defaults popularized by Kamradt / LangChain's SemanticChunker).
_DEFAULT_THRESHOLD_AMOUNTS: dict[str, float] = {
    "percentile": 95.0,
    "standard_deviation": 3.0,
    "interquartile": 1.5,
    "gradient": 95.0,
}


@registry.register("chunking", "semantic")
class SemanticChunker(BaseChunker):
    """Semantic chunker: splits where embedding similarity drops.

    A direct implementation of the standard embedding-breakpoint algorithm
    (Kamradt; popularized by LangChain's ``SemanticChunker``), owned here
    rather than wrapped so it carries no ``langchain-experimental``
    dependency and reuses the project's own embedder:

      1. Split each document into sentences.
      2. Combine each sentence with ``buffer_size`` neighbors per side.
      3. Embed the combined groups.
      4. Take cosine distance between consecutive group embeddings.
      5. Split at the gaps whose distance exceeds a threshold derived from
         the chosen statistic.

    The breakpoint embedder is independent of the indexing embedder and
    defaults to a small local sentence-transformers model: it embeds every
    sentence group, so it is kept cheap and local.

    Config params:
        embedding_model: sentence-transformers model id for breakpoint
            detection (default: "BAAI/bge-small-en-v1.5").
        breakpoint_threshold_type: "percentile" (default),
            "standard_deviation", "interquartile", or "gradient".
        breakpoint_threshold_amount: Statistic amount for the chosen type
            (defaults: percentile 95, standard_deviation 3,
            interquartile 1.5, gradient 95).
        buffer_size: Sentences of context combined per group (default 1).
        sentence_split_regex: Sentence-boundary regex
            (default: ``(?<=[.?!])\\s+``).
        min_chunk_size: If set, a chunk shorter than this many characters
            is merged into the next one (default: None).
    """

    _VALID_THRESHOLD_TYPES = frozenset(_DEFAULT_THRESHOLD_AMOUNTS)

    class Params(ComponentParams):
        embedding_model: str = "BAAI/bge-small-en-v1.5"
        breakpoint_threshold_type: str = "percentile"
        # ``None`` → the per-type default in _DEFAULT_THRESHOLD_AMOUNTS.
        breakpoint_threshold_amount: float | None = None
        buffer_size: int = 1
        sentence_split_regex: str = _DEFAULT_SENTENCE_REGEX
        min_chunk_size: int | None = None

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)

        self._threshold_type: str = self.p.breakpoint_threshold_type
        if self._threshold_type not in self._VALID_THRESHOLD_TYPES:
            raise ValueError(
                f"SemanticChunker breakpoint_threshold_type must be one of "
                f"{sorted(self._VALID_THRESHOLD_TYPES)}, got "
                f"'{self._threshold_type}'"
            )
        self._threshold_amount: float = (
            self.p.breakpoint_threshold_amount
            if self.p.breakpoint_threshold_amount is not None
            else _DEFAULT_THRESHOLD_AMOUNTS[self._threshold_type]
        )
        self._buffer_size: int = self.p.buffer_size
        self._sentence_regex: str = self.p.sentence_split_regex
        self._min_chunk_size: int | None = self.p.min_chunk_size

        # Injectable embedding seam (tests pass a deterministic fn). Built
        # lazily on first chunk() so constructing the chunker on the
        # cache-load path never loads the breakpoint model — a cached
        # semantic index loads without needing the model present.
        self._embed_fn: Callable[[list[str]], list[list[float]]] | None = embed_fn

    def _get_embed_fn(self) -> Callable[[list[str]], list[list[float]]]:
        if self._embed_fn is None:
            self._embed_fn = self._build_embed_fn()
        return self._embed_fn

    def _build_embed_fn(self) -> Callable[[list[str]], list[list[float]]]:
        from ragbench.components.embedders import HuggingFaceEmbedder

        # auto_prompt=False: breakpoint detection embeds sentence groups, not
        # queries/documents, so the family's retrieval prompts must not leak in.
        embedder = HuggingFaceEmbedder(
            {
                "model_name": self.p.embedding_model,
                "normalize": True,
                "auto_prompt": False,
            }
        )

        def embed(texts: list[str]) -> list[list[float]]:
            tmp = [
                Chunk(content=t, chunk_id=str(i), metadata={})
                for i, t in enumerate(texts)
            ]
            return [ec.embedding for ec in embedder.embed_chunks(tmp)]

        return embed

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, text in enumerate(self._split_document(doc.content)):
                chunk_id = _make_chunk_id(text, doc.metadata, i)
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_id=chunk_id,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks

    def _split_document(self, content: str) -> list[str]:
        if not content.strip():
            return []
        sentences = [s for s in re.split(self._sentence_regex, content) if s.strip()]
        # Too few sentences to compute breakpoints → one chunk.
        if len(sentences) <= 2:
            return [" ".join(sentences)]

        combined = self._combine_with_buffer(sentences)
        embeddings = np.asarray(self._get_embed_fn()(combined), dtype=np.float64)
        distances = self._cosine_distances(embeddings)
        breakpoints = self._find_breakpoints(distances)
        groups = self._group_sentences(sentences, breakpoints)
        if self._min_chunk_size is not None:
            groups = self._merge_small(groups)
        return groups

    def _combine_with_buffer(self, sentences: list[str]) -> list[str]:
        """Combine each sentence with ``buffer_size`` neighbors per side."""
        n = len(sentences)
        combined: list[str] = []
        for i in range(n):
            lo = max(0, i - self._buffer_size)
            hi = min(n, i + self._buffer_size + 1)
            combined.append(" ".join(sentences[lo:hi]))
        return combined

    @staticmethod
    def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
        """Cosine distance between each consecutive pair of group vectors."""
        a = embeddings[:-1]
        b = embeddings[1:]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0.0, 1e-10, denom)
        sims = np.sum(a * b, axis=1) / denom
        return 1.0 - sims

    def _find_breakpoints(self, distances: np.ndarray) -> list[int]:
        """Indices i where the gap after sentence i is a split point."""
        if distances.size < 2:
            return []

        if self._threshold_type == "gradient":
            gradients = np.gradient(distances)
            threshold = float(np.percentile(gradients, self._threshold_amount))
            return [i for i, g in enumerate(gradients) if g > threshold]

        if self._threshold_type == "percentile":
            threshold = float(np.percentile(distances, self._threshold_amount))
        elif self._threshold_type == "standard_deviation":
            threshold = float(
                np.mean(distances) + self._threshold_amount * np.std(distances)
            )
        else:  # interquartile
            q1, q3 = np.percentile(distances, [25, 75])
            threshold = float(np.mean(distances) + self._threshold_amount * (q3 - q1))

        return [i for i, d in enumerate(distances) if d > threshold]

    @staticmethod
    def _group_sentences(sentences: list[str], breakpoints: list[int]) -> list[str]:
        """Join sentences into chunks, cutting after each breakpoint index."""
        groups: list[str] = []
        start = 0
        for bp in breakpoints:
            end = bp + 1  # gap after sentence bp → boundary before bp+1
            groups.append(" ".join(sentences[start:end]))
            start = end
        if start < len(sentences):
            groups.append(" ".join(sentences[start:]))
        return [g for g in groups if g.strip()]

    def _merge_small(self, groups: list[str]) -> list[str]:
        """Merge any chunk shorter than ``min_chunk_size`` into the next."""
        min_size = self._min_chunk_size
        if min_size is None:  # only called when set, but keep the type honest
            return groups
        merged: list[str] = []
        carry = ""
        for g in groups:
            candidate = f"{carry} {g}".strip() if carry else g
            if len(candidate) < min_size:
                carry = candidate
            else:
                merged.append(candidate)
                carry = ""
        if carry:
            if merged:
                merged[-1] = f"{merged[-1]} {carry}"
            else:
                merged.append(carry)
        return merged
