"""Swappable pipeline components for RAG methods.

Structure
---------
base.py        — Chunker, Retriever, Reranker, Generator Protocols (interfaces)
chunkers.py    — RecursiveChunker  (add SentenceChunker, TokenChunker, etc. here)
retrievers.py  — SimpleRetriever   (add HybridRetriever, MMRRetriever, etc. here)
rerankers.py   — NoReranker        (add CrossEncoderReranker, CohereReranker, etc. here)
generators.py  — StuffGenerator    (add MapReduceGenerator, RefineGenerator, etc. here)

All public names are re-exported here so callers can import from the package root:

    from rag_testing.components import SimpleRetriever, NoReranker, StuffGenerator
    from rag_testing.components import Retriever, Reranker, Generator
    from rag_testing.components import RecursiveChunker, Chunker
"""

from .base import Chunker, Generator, Reranker, Retriever
from .chunkers import RecursiveChunker
from .generators import StuffGenerator
from .rerankers import NoReranker
from .retrievers import SimpleRetriever

__all__ = [
    "Chunker",
    "Generator",
    "NoReranker",
    "RecursiveChunker",
    "Reranker",
    "Retriever",
    "SimpleRetriever",
    "StuffGenerator",
]
