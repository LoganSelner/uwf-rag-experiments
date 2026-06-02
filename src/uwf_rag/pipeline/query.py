"""Query pipeline.

Processes queries through the linear RAG pipeline:
  query → transform → retrieve → rerank → (prompt → generate)

Supports retrieval-only mode (skips generation) for fast
retriever evaluation without LLM cost.
"""

from __future__ import annotations

import logging
from typing import Any

from uwf_rag.components.base import (
    BaseGenerator,
    BaseLexicalIndex,
    BasePromptTemplate,
    BaseQueryTransformer,
    BaseReranker,
    BaseRetriever,
)
from uwf_rag.components.generators import build_generator
from uwf_rag.components.retrievers import HybridRetriever
from uwf_rag.core.config import QueryConfig, RetrievalConfig
from uwf_rag.core.registry import registry
from uwf_rag.core.types import GenerationResult, Message
from uwf_rag.pipeline.indexing import BM25_AUX_KEY, IndexArtifact

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Orchestrates query processing through the linear RAG pipeline."""

    def __init__(
        self,
        query_transformer: BaseQueryTransformer,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        top_k_retrieve: int,
        top_k_final: int,
        generator: BaseGenerator | None = None,
        prompt_template: BasePromptTemplate | None = None,
        retrieval_only: bool = False,
        qt_fusion: str = "rrf",
    ) -> None:
        self._query_transformer = query_transformer
        self._retriever = retriever
        self._reranker = reranker
        self._top_k_retrieve = top_k_retrieve
        self._top_k_final = top_k_final
        self._generator = generator
        self._prompt_template = prompt_template
        self._retrieval_only = retrieval_only
        self._qt_fusion = qt_fusion

    @classmethod
    def from_config(
        cls,
        config: QueryConfig,
        index_artifact: IndexArtifact,
        retrieval_only: bool = False,
    ) -> QueryPipeline:
        """Build a QueryPipeline from config and a pre-built index.

        Args:
            config: The query section of the experiment config.
            index_artifact: The IndexArtifact from the indexing pipeline.
            retrieval_only: If True, skip generator/prompt construction.
        """
        # Query transformer — its reasoning LLM (if any) is built here via the
        # shared factory and injected; the transformer never touches the registry.
        qt_type = config.query_transform.type or "passthrough"
        qt_cls = registry.get("query_transform", qt_type)
        qt_generator = (
            build_generator(config.query_transform.generator)
            if config.query_transform.generator.provider
            else None
        )
        query_transformer: BaseQueryTransformer = qt_cls(
            config=config.query_transform.params,
            generator=qt_generator,
        )

        # Retriever — built with its index sources (and default filters)
        # injected at construction. Hybrid retrievers compose child
        # retrievers, each built the same way.
        retriever = cls._build_retriever(config.retrieval, index_artifact)

        # Reranker
        rerank_type = config.reranking.type or "none"
        reranker_cls = registry.get("reranking", rerank_type)
        reranker: BaseReranker = reranker_cls(config=config.reranking.params)

        # Generator + prompt (only if not retrieval-only)
        generator: BaseGenerator | None = None
        prompt_template: BasePromptTemplate | None = None
        if not retrieval_only:
            if config.generator.provider:
                generator = build_generator(config.generator)

            if config.prompt.type:
                prompt_cls = registry.get("prompts", config.prompt.type)
                prompt_config: dict[str, Any] = {
                    "system_template": config.prompt.system_template,
                    "context_format": config.prompt.context_format,
                    "use_chain_of_thought": config.prompt.use_chain_of_thought,
                    "citation_style": config.prompt.citation_style,
                    "few_shot_examples": config.prompt.few_shot_examples,
                    "max_context_tokens": config.prompt.max_context_tokens,
                }
                prompt_template = prompt_cls(config=prompt_config)

        return cls(
            query_transformer=query_transformer,
            retriever=retriever,
            reranker=reranker,
            top_k_retrieve=config.retrieval.top_k_retrieve,
            top_k_final=config.retrieval.top_k_final,
            generator=generator,
            prompt_template=prompt_template,
            retrieval_only=retrieval_only,
            qt_fusion=config.query_transform.fusion,
        )

    def run(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> GenerationResult:
        """Run the query pipeline end-to-end.

        Args:
            query: The user's question.
            history: Optional conversation history for context-aware transforms.

        Returns:
            GenerationResult with retrieved chunks and (if not
            retrieval-only) a generated answer.
        """
        # 1. Transform query
        transformed = self._query_transformer.transform(query, history)
        logger.info("Query transformed into %d search queries", len(transformed))

        # 2. Retrieve and fuse across all transformed queries. The retriever
        #    owns the per-branch routing and intra/cross-branch fusion; the
        #    pipeline only supplies the strategy.
        all_chunks = self._retriever.retrieve_multi(
            transformed,
            top_k_per_query=self._top_k_retrieve,
            fusion=self._qt_fusion,
            filters=None,
        )
        logger.info("Retrieved %d unique candidates", len(all_chunks))

        # 3. Rerank and truncate to top_k_final
        reranked = self._reranker.rerank(query, all_chunks, self._top_k_final)
        logger.info("Reranked to %d chunks", len(reranked))

        # 4. If retrieval-only, return without generation
        if self._retrieval_only:
            return GenerationResult(
                query=query,
                answer="",
                retrieved_chunks=reranked,
                metadata={"retrieval_only": True},
            )

        # 5. Format prompt and generate
        if self._prompt_template is None or self._generator is None:
            raise RuntimeError(
                "Generator and prompt template are required "
                "for non-retrieval-only mode. Set "
                "retrieval_only=True or configure generation."
            )

        prompt = self._prompt_template.format(query, reranked, history)
        result = self._generator.generate(prompt)
        result.query = query
        result.retrieved_chunks = reranked
        return result

    @classmethod
    def _build_retriever(
        cls,
        retrieval: RetrievalConfig,
        index_artifact: IndexArtifact,
    ) -> BaseRetriever:
        """Build the retriever (single or hybrid) with its sources injected."""
        default_filters = retrieval.filters or None
        if retrieval.type == "hybrid":
            return cls._build_hybrid_retriever(
                retrieval.params, index_artifact, default_filters
            )
        return cls._build_single_retriever(
            retrieval.type, retrieval.params, index_artifact, default_filters
        )

    @staticmethod
    def _build_single_retriever(
        retrieval_type: str,
        params: dict[str, Any],
        index_artifact: IndexArtifact,
        default_filters: dict[str, Any] | None,
    ) -> BaseRetriever:
        """Construct one retriever with its index sources injected.

        Dense retrievers receive vectorstore + embedder; BM25 receives the
        sparse index from ``auxiliary_stores``. Other future retrievers can
        extend this dispatch.
        """
        retriever_cls = registry.get("retrieval", retrieval_type)
        if retrieval_type == "bm25":
            sparse_index = index_artifact.auxiliary_stores.get(BM25_AUX_KEY)
            if sparse_index is None:
                raise RuntimeError(
                    "BM25 retriever requires a sparse index, but the "
                    "IndexArtifact has none under "
                    f"auxiliary_stores['{BM25_AUX_KEY}']. Ensure "
                    "indexing.sparse_index is configured."
                )
            if not isinstance(sparse_index, BaseLexicalIndex):
                raise RuntimeError(
                    f"auxiliary_stores['{BM25_AUX_KEY}'] is not a BaseLexicalIndex"
                )
            return retriever_cls(
                config=params,
                sparse_index=sparse_index,
                default_filters=default_filters,
            )
        # Default: dense-style (vectorstore + embedder).
        return retriever_cls(
            config=params,
            vectorstore=index_artifact.vectorstore,
            embedder=index_artifact.embedder,
            default_filters=default_filters,
        )

    @classmethod
    def _build_hybrid_retriever(
        cls,
        params: dict[str, Any],
        index_artifact: IndexArtifact,
        default_filters: dict[str, Any] | None,
    ) -> HybridRetriever:
        """Construct a HybridRetriever and its child sub-retrievers.

        Each sub-retriever entry: ``{name, type, top_k, params}``.
        ``name`` defaults to ``type`` and must be unique within the
        hybrid (validated upstream in ``validate_config``). The hybrid's
        ``default_filters`` are injected into each child too, so a child's
        own retrieve path filters identically to the standalone case.
        """
        sub_specs = params.get("retrievers", [])
        children: list[tuple[str, BaseRetriever, int]] = []
        for spec in sub_specs:
            sub_type = spec["type"]
            sub_name = spec.get("name", sub_type)
            sub_top_k = int(spec["top_k"])
            sub_params = spec.get("params", {})
            child = cls._build_single_retriever(
                sub_type, sub_params, index_artifact, default_filters
            )
            children.append((sub_name, child, sub_top_k))

        fusion = params.get("fusion", "rrf")
        rrf_k = int(params.get("rrf_k", 60))
        weights_dict: dict[str, float] = {
            k: float(v) for k, v in params.get("weights", {}).items()
        }
        normalize = params.get("normalize", "min_max")

        return HybridRetriever(
            config=params,
            children=children,
            fusion=fusion,
            rrf_k=rrf_k,
            weights=weights_dict,
            normalize=normalize,
            default_filters=default_filters,
        )
