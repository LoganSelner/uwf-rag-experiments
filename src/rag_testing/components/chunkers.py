"""Chunker implementations.

Each class satisfies the ``Chunker`` Protocol defined in ``.base``.

Implementations
---------------
RecursiveChunker — recursive character splitting (current baseline)

To add a new chunker (e.g. SentenceChunker, TokenChunker):
  1. Add a class here satisfying the ``Chunker`` Protocol (``chunk`` method) and
     add a ``from_settings(cls, settings: Settings) -> <class>`` classmethod.
  2. Export it from ``__init__.py``.
  3. Add a branch in ``ingest.build_chunker()`` for the new ``chunker_type`` value.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings


@dataclass(frozen=True)
class RecursiveChunker:
    """Splits documents with LangChain's RecursiveCharacterTextSplitter."""

    chunk_size: int
    chunk_overlap: int

    def chunk(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(docs)

    @classmethod
    def from_settings(cls, settings: Settings) -> RecursiveChunker:
        return cls(
            chunk_size=settings.index.chunk_size,
            chunk_overlap=settings.index.chunk_overlap,
        )
