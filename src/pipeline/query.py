"""Query pipeline.

Processes queries through the linear RAG pipeline:
  query → transform → retrieve → rerank → (prompt → generate)

Supports retrieval-only mode (skips generation) for fast
retriever evaluation without LLM cost.
"""

from __future__ import annotations

import logging
from typing import Any

from components.base import (
    BaseGenerator,
    BasePromptTemplate,
    BaseQueryTransformer,
    BaseReranker,
    BaseRetriever,
)
from core.config import QueryConfig
from core.registry import registry
from core.types import GenerationResult, RetrievedChunk
from pipeline.indexing import IndexArtifact

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
    ) -> None:
        self._query_transformer = query_transformer
        self._retriever = retriever
        self._reranker = reranker
        self._top_k_retrieve = top_k_retrieve
        self._top_k_final = top_k_final
        self._generator = generator
        self._prompt_template = prompt_template
        self._retrieval_only = retrieval_only

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
        # Query transformer
        qt_type = config.query_transform.type or "passthrough"
        qt_cls = registry.get("query_transform", qt_type)
        query_transformer: BaseQueryTransformer = qt_cls(
            config=config.query_transform.params
        )

        # Retriever — wire to vectorstore and embedder from index
        retriever_cls = registry.get("retrieval", config.retrieval.type)
        retriever: BaseRetriever = retriever_cls(config=config.retrieval.params)
        retriever.set_vectorstore(index_artifact.vectorstore)
        retriever.set_embedder(index_artifact.embedder)
        if config.retrieval.filters:
            retriever.set_default_filters(config.retrieval.filters)

        # Reranker
        rerank_type = config.reranking.type or "none"
        reranker_cls = registry.get("reranking", rerank_type)
        reranker: BaseReranker = reranker_cls(config=config.reranking.params)

        # Generator + prompt (only if not retrieval-only)
        generator: BaseGenerator | None = None
        prompt_template: BasePromptTemplate | None = None
        if not retrieval_only:
            if config.generation.type:
                gen_cls = registry.get("generation", config.generation.type)
                gen_config: dict[str, Any] = {
                    **config.generation.params,
                    "llm": {
                        "provider": config.generation_llm.provider,
                        "model_name": config.generation_llm.model_name,
                        "temperature": config.generation_llm.temperature,
                        "max_tokens": config.generation_llm.max_tokens,
                    },
                }
                generator = gen_cls(config=gen_config)

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
        )

    def run(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
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
        queries = self._query_transformer.transform(query, history)
        logger.info("Query transformed into %d search queries", len(queries))

        # 2. Retrieve for each transformed query, deduplicate
        all_chunks = self._retrieve_and_deduplicate(queries)
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

    def _retrieve_and_deduplicate(self, queries: list[str]) -> list[RetrievedChunk]:
        """Retrieve for each query, deduplicate by chunk_id."""
        seen: dict[str, RetrievedChunk] = {}

        for q in queries:
            results = self._retriever.retrieve(q, self._top_k_retrieve)
            for rc in results:
                cid = rc.chunk.chunk_id
                if cid not in seen or rc.score > seen[cid].score:
                    seen[cid] = rc

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)
