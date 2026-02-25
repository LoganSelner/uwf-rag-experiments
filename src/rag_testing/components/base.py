"""Protocol definitions for all RAG pipeline stages.

These are the interfaces every component implementation must satisfy.  They use
Python's structural typing (``Protocol``) so any class with the correct method
signature automatically conforms — no inheritance required.

Pipeline stages
---------------
Chunker    → docs → list[Document]  (index-time only; not part of answer())
Retriever  → question → list[Document]
Reranker   → question + docs → list[Document]  (filtered/reordered subset)
Generator  → question + context strings → answer string
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Chunker(Protocol):
    def chunk(self, docs: list[Document]) -> list[Document]:
        """Split documents into chunks suitable for embedding."""
        ...


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[Document]:
        """Return candidate documents for the given question."""
        ...


class Reranker(Protocol):
    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        """Return a reordered / filtered subset of *docs* for the given question."""
        ...


class Generator(Protocol):
    def generate(self, question: str, contexts: list[str]) -> str:
        """Return a generated answer string given a question and context strings."""
        ...
