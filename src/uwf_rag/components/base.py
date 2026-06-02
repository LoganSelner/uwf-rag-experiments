"""Abstract base classes for all pipeline components.

This file IS the architectural contract. Every pipeline stage has an
ABC here that defines exactly what goes in and what comes out.
Implementations are swappable because they honor these interfaces.

DO NOT add implementation logic here. This file should rarely change
after the initial design is finalized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from uwf_rag.core.fusion import max_score_dedup, reciprocal_rank_fusion
from uwf_rag.core.types import (
    Chunk,
    Document,
    EmbeddedChunk,
    GenerationResult,
    Message,
    RetrievedChunk,
    ToolResult,
    ToolSpec,
    TransformedQuery,
)

if TYPE_CHECKING:
    from uwf_rag.components.build import BuildContext
    from uwf_rag.core.config import AgentDefinitionConfig, ValidateContext


class ComponentParams(BaseModel):
    """Base for a component's typed parameter model.

    A component declares a nested ``class Params(ComponentParams)`` with typed
    fields + defaults; its ``__init__`` validates the raw ``config`` dict into
    ``self.p = self.Params.model_validate(self.config)``. This is the single
    home for a component's defaults and param validation, replacing scattered
    ``self.config.get(key, default)`` reads. Unknown keys are rejected so a
    typo'd YAML param fails loudly rather than being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


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


class BaseChunkEnricher(ABC):
    """Transforms chunks between chunking and embedding/indexing.

    A build-time-only stage that may set each chunk's ``index_text`` (the
    text that gets embedded + lexically indexed) without touching
    ``content`` (what is stored, generated from, and scored). The default
    implementation is a no-op; the contextual enricher prepends a
    model-generated, document-situating blurb (the Anthropic contextual-
    retrieval recipe).

    Receives the source ``documents`` as well as the ``chunks`` so an
    implementation can situate a chunk within its parent document/page via
    provenance metadata (``source_path``, ``page_number``).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def enrich(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Return chunks, optionally with ``index_text`` populated.

        Must preserve each chunk's ``content`` and ``chunk_id``. May return
        the same list (mutated in place) or a new list of the same length.
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


class BaseLexicalIndex(ABC):
    """Stores and searches chunks via lexical (term-frequency) matching.

    Parallel to BaseVectorStore but for sparse/lexical retrievers like
    BM25 that operate on raw text rather than dense embeddings. The
    index owns its own tokenizer; query-time tokenization mirrors
    index-time tokenization.

    Named "lexical" rather than "sparse" to leave room for learned
    sparse retrievers (e.g. SPLADE) that produce sparse *vectors* and
    naturally fit BaseVectorStore.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Tokenize and index a batch of chunks."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for the top_k most lexically relevant chunks.

        Takes a raw query string — the index handles its own tokenization.
        Implementations without native filtering must over-retrieve and
        post-filter, mirroring the FAISS pattern.
        """

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""


class BaseRetriever(ABC):
    """Retrieves relevant chunks for a query.

    Dependencies (vectorstore + embedder for dense, a lexical index for BM25,
    composed children for hybrid) and the default metadata filters (from
    ``query.retrieval.filters``) are injected through each concrete retriever's
    constructor by ``QueryPipeline.from_config``. There is no post-construction
    wiring, so a retriever is always fully formed and usable once built.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        default_filters: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self._default_filters: dict[str, Any] = default_filters or {}

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        params: dict[str, Any],
        default_filters: dict[str, Any] | None,
        ctx: BuildContext,
    ) -> BaseRetriever:
        """Construct this retriever, pulling its index dependencies from ``ctx``.

        Each retriever owns its own wiring: dense pulls the vector store +
        embedder from ``ctx.index``; BM25 pulls its lexical index; hybrid
        recurses through ``ctx.registry`` to build its children. The pipeline
        calls ``registry.get("retrieval", type).build(...)`` and never branches
        on a concrete retriever type.
        """

    @classmethod
    def validate_params(
        cls,
        params: dict[str, Any],
        ctx: ValidateContext,
    ) -> list[str]:
        """Return config-validation error strings for this retriever's params.

        Default: no retriever-specific checks. Overridden by retrievers whose
        params have structure to validate (BM25 needs a sparse index; hybrid
        has a child roster + fusion settings). Keeps that knowledge in the
        component rather than in ``core.validate_config``. The cross-config
        context (registry, configured sparse-index type, retrieve budget)
        arrives via ``ctx`` so the component never reaches into global config.
        """
        return []

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top_k most relevant chunks for a query.

        Uses ``self._default_filters`` merged with any explicit filters.
        """

    def retrieve_multi(
        self,
        transformed_queries: list[TransformedQuery],
        top_k_per_query: int,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse across multiple transformed queries.

        Default implementation: ignores ``branch`` hints, calls
        :meth:`retrieve` once per query, fuses the per-query rank lists
        via the chosen strategy. Branch hints are meaningful only inside
        :class:`HybridRetriever` (which overrides this method); non-hybrid
        retrievers silently drop the hint, by design — this lets the same
        HyDE/multi-query config work in both dense and hybrid setups.

        Args:
            transformed_queries: One or more queries from a
                :class:`BaseQueryTransformer`. Empty list returns ``[]``.
            top_k_per_query: Per-query retrieval depth before fusion.
                The fused output is also truncated to this size.
            fusion: ``"rrf"`` (Reciprocal Rank Fusion across queries) or
                ``"max"`` (highest score per chunk across queries). A
                single query short-circuits both (fusion is a no-op).
            filters: Optional per-call filters forwarded to each
                ``retrieve`` call. Merged with ``self._default_filters``
                inside ``retrieve``.

        Returns:
            Top-``top_k_per_query`` fused chunks with scores reflecting
            the fusion strategy. ``retrieval_method`` carries through
            from the first occurrence of each chunk.
        """
        if not transformed_queries:
            return []

        ranked_id_lists: list[list[str]] = []
        scored_lists: list[list[tuple[str, float]]] = []
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for tq in transformed_queries:
            results = self.retrieve(tq.text, top_k_per_query, filters)
            ids: list[str] = []
            scored: list[tuple[str, float]] = []
            for rc in results:
                cid = rc.chunk.chunk_id
                ids.append(cid)
                scored.append((cid, rc.score))
                chunk_lookup.setdefault(cid, rc)
            ranked_id_lists.append(ids)
            scored_lists.append(scored)

        if len(transformed_queries) == 1:
            # Single query: fusion is a no-op, preserve the retrieval order.
            return [chunk_lookup[cid] for cid in ranked_id_lists[0]][:top_k_per_query]

        if fusion == "rrf":
            fused = reciprocal_rank_fusion(ranked_id_lists)
        elif fusion == "max":
            fused = max_score_dedup(scored_lists)
        else:
            raise ValueError(
                f"retrieve_multi: unknown fusion mode '{fusion}' "
                "(expected 'rrf' or 'max')"
            )

        out: list[RetrievedChunk] = []
        for cid, fused_score in fused[:top_k_per_query]:
            base = chunk_lookup[cid]
            out.append(
                RetrievedChunk(
                    chunk=base.chunk,
                    score=fused_score,
                    retrieval_method=base.retrieval_method,
                )
            )
        return out


class BaseQueryTransformer(ABC):
    """Transforms a user query before retrieval.

    Always returns a list of TransformedQuery instances, even for
    single-query transforms. This naturally supports passthrough (1
    query), HyDE (1 query, optionally branch-tagged for hybrid routing),
    multi-query (N queries), and sub-question decomposition (N queries).

    LLM-backed transformers receive their reasoning ``generator`` via
    constructor injection (built by the pipeline through
    ``build_generator``); a transformer never constructs its own generator
    or touches the registry. Passthrough ignores it.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        generator: BaseGenerator | None = None,
    ) -> None:
        self.config = config or {}
        self._generator = generator

    @property
    def _llm(self) -> BaseGenerator:
        """The injected reasoning generator (LLM-backed transformers only).

        LLM-backed transformers require a generator and enforce it in their
        ``__init__``; this accessor narrows the optional for callers and fails
        loudly if one is reached without an injected generator.
        """
        if self._generator is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a generator but none was injected."
            )
        return self._generator

    @abstractmethod
    def transform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[TransformedQuery]:
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
    def generate(
        self,
        prompt: str | list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> GenerationResult:
        """Generate a response from a formatted prompt.

        Returns a GenerationResult with answer and metadata populated.
        The pipeline is responsible for attaching query and chunks.

        When ``tools`` is ``None`` (the linear pipeline's only call), this is
        a plain free-text completion and the result's ``tool_calls`` /
        ``finish_reason`` stay empty — behavior is identical to before tool
        support existed. When ``tools`` is provided, the model may instead
        request tool calls: the result carries them in ``tool_calls`` (with
        ``answer`` possibly empty) and the caller (the agent loop) executes
        them and feeds results back via the neutral message schema. ``prompt``
        may then contain assistant turns with ``tool_calls`` and ``tool``-role
        turns with ``tool_call_id`` (see :data:`core.types.Message`).
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
        history: list[Message] | None = None,
    ) -> str | list[Message]:
        """Format components into a prompt string or message list."""


class BaseTool(ABC):
    """An external tool usable by agent pipelines.

    Tools are constructed through :meth:`build` (resolved from the ``tool``
    registry category and given a :class:`BuildContext`), not via a generic
    ``config`` dict — a tool typically wraps live dependencies (a retrieval
    sub-pipeline, an API client) rather than a flat param bag.
    """

    @classmethod
    @abstractmethod
    def build(cls, cfg: AgentDefinitionConfig, ctx: BuildContext) -> BaseTool:
        """Construct this tool from its roster entry + the build context."""

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
    def get_history(self) -> list[Message]:
        """Return conversation history (windowed by implementation)."""

    @abstractmethod
    def clear(self) -> None:
        """Reset the memory."""
