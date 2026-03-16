"""Abstract base classes for all pipeline components.

This file IS the architectural contract. Every pipeline stage has an
ABC here that defines exactly what goes in and what comes out.
Implementations are swappable because they honor these interfaces.

DO NOT add implementation logic here. This file should rarely change
after the initial design is finalized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import (
    Chunk,
    Document,
    EmbeddedChunk,
    GenerationResult,
    RetrievedChunk,
    ToolResult,
)


class BaseIngestor(ABC):
    """Reads a source file and produces Documents."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def ingest(self, path: str) -> list[Document]:
        """Ingest a file and return a list of Documents.

        Each Document should have metadata including at minimum the
        source file path. The ``source_name`` metadata is added by the
        indexing pipeline, not by the ingestor.
        """


class BaseChunker(ABC):
    """Splits Documents into Chunks."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks.

        Each Chunk must have a unique ``chunk_id`` and carry forward
        relevant metadata from the source Document.
        """


class BaseEmbedder(ABC):
    """Produces embedding vectors for chunks and queries."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of chunks, returning EmbeddedChunks."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string, returning a vector."""


class BaseVectorStore(ABC):
    """Stores and searches embedded chunks."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the store."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for the top_k most similar chunks.

        Implementations that don't support native filtering (e.g. FAISS)
        must implement post-retrieval filtering internally: over-retrieve,
        filter, return top_k.
        """

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist the store to disk."""

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load a previously saved store from disk."""


class BaseRetriever(ABC):
    """Retrieves relevant chunks for a query.

    Holds references to a vectorstore and embedder, injected after
    construction via set_vectorstore() / set_embedder(). This allows
    the pipeline to construct components independently from config
    and wire them together afterward.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._vectorstore: BaseVectorStore | None = None
        self._embedder: BaseEmbedder | None = None
        self._default_filters: dict[str, Any] = {}

    def set_vectorstore(self, vectorstore: BaseVectorStore) -> None:
        self._vectorstore = vectorstore

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def set_default_filters(self, filters: dict[str, Any]) -> None:
        self._default_filters = filters

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top_k most relevant chunks for a query.

        Uses self._default_filters merged with any explicit filters.
        Embeds the query using self._embedder, searches self._vectorstore.
        """


class BaseQueryTransformer(ABC):
    """Transforms a user query before retrieval.

    Always returns a list of queries, even for single-query transforms.
    This naturally supports passthrough (1 query), HyDE (1 query),
    multi-query (N queries), and sub-question decomposition (N queries).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def transform(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Transform the query into one or more search queries."""


class BaseReranker(ABC):
    """Re-scores retrieved chunks and truncates to top_k."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Re-score chunks and return the top_k most relevant."""


class BaseGenerator(ABC):
    """Pure LLM wrapper. Takes a formatted prompt, returns an answer.

    The generator never sees raw chunks — the pipeline calls
    PromptTemplate.format() first, then passes the result here.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str | list[dict[str, str]]) -> GenerationResult:
        """Generate a response from a formatted prompt.

        Returns a GenerationResult with answer and metadata populated.
        The pipeline is responsible for attaching query and chunks.
        """


class BasePromptTemplate(ABC):
    """Formats query + chunks + history into a prompt for the generator."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def format(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> str | list[dict[str, str]]:
        """Format components into a prompt string or message list."""


class BaseTool(ABC):
    """An external tool usable by agent pipelines."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """The tool's name, used in agent routing."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the agent to decide when to use it."""

    @abstractmethod
    def execute(self, query: str) -> ToolResult:
        """Execute the tool and return a result."""


class BaseMemory(ABC):
    """Conversation memory for agent pipelines."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def add_turn(self, role: str, content: str) -> None:
        """Record a conversation turn."""

    @abstractmethod
    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history (windowed by implementation)."""

    @abstractmethod
    def clear(self) -> None:
        """Reset the memory."""
