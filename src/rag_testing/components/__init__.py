"""Swappable pipeline components for RAG methods.

Structure
---------
base.py        — Chunker, Retriever, Reranker, Generator Protocols (interfaces)
chunkers.py    — RecursiveChunker  (add SentenceChunker, TokenChunker, etc. here)
retrievers.py  — SimpleRetriever, MMRRetriever  (add HybridRetriever, etc. here)
rerankers.py   — NoReranker, CrossEncoderReranker  (add CohereReranker, etc. here)
generators.py  — StuffGenerator    (add MapReduceGenerator, RefineGenerator, etc. here)

All public names are re-exported here so callers can import from the package root:

    from rag_testing.components import SimpleRetriever, NoReranker, StuffGenerator
    from rag_testing.components import Retriever, Reranker, Generator
    from rag_testing.components import RecursiveChunker, Chunker
"""

from .base import Chunker, Generator, Reranker, Retriever
from .chunkers import RecursiveChunker
from .generators import StuffGenerator
from .rerankers import CrossEncoderReranker, NoReranker
from .retrievers import HybridRetriever, MMRRetriever, SimpleRetriever

__all__ = [
    "Chunker",
    "CrossEncoderReranker",
    "Generator",
    "HybridRetriever",
    "MMRRetriever",
    "NoReranker",
    "RecursiveChunker",
    "Reranker",
    "Retriever",
    "SimpleRetriever",
    "StuffGenerator",
]
