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
    BaseLexicalIndex,
    BasePromptTemplate,
    BaseQueryTransformer,
    BaseReranker,
    BaseRetriever,
)
from components.retrievers import HybridRetriever
from core.config import QueryConfig
from core.registry import registry
from core.types import GenerationResult, RetrievedChunk
from pipeline.indexing import BM25_AUX_KEY, IndexArtifact

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

        # Retriever — wire to vectorstore / embedder / sparse index
        # from the IndexArtifact. Hybrid retrievers compose child
        # retrievers, each independently wired.
        retriever: BaseRetriever
        if config.retrieval.type == "hybrid":
            retriever = cls._build_hybrid_retriever(
                config.retrieval.params, index_artifact
            )
        else:
            retriever_cls = registry.get("retrieval", config.retrieval.type)
            retriever = retriever_cls(config=config.retrieval.params)
            cls._wire_retriever(retriever, config.retrieval.type, index_artifact)
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
        """Retrieve for each query, deduplicate by chunk_id.

        All retrievals here come from the same retriever instance, so
        scores are comparable within this call — multi-query (Phase B)
        will still produce per-query lists from one underlying scoring
        function. Cross-retriever score merging (e.g. cosine vs. BM25)
        is *not* this method's job; ``HybridRetriever`` owns that via
        ``core.fusion``. We keep the highest score per chunk_id here
        and sort descending; ties keep first occurrence.
        """
        seen: dict[str, RetrievedChunk] = {}

        for q in queries:
            results = self._retriever.retrieve(q, self._top_k_retrieve)
            for rc in results:
                cid = rc.chunk.chunk_id
                if cid not in seen or rc.score > seen[cid].score:
                    seen[cid] = rc

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def _wire_retriever(
        retriever: BaseRetriever,
        retrieval_type: str,
        index_artifact: IndexArtifact,
    ) -> None:
        """Inject the appropriate index sources into a single retriever.

        Dense retrievers need vectorstore + embedder; BM25 retrievers
        need the sparse index from ``auxiliary_stores``. Other future
        retrievers can extend this dispatch.
        """
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
            retriever.set_sparse_index(sparse_index)
        else:
            # Default: dense-style wiring (vectorstore + embedder).
            retriever.set_vectorstore(index_artifact.vectorstore)
            retriever.set_embedder(index_artifact.embedder)

    @classmethod
    def _build_hybrid_retriever(
        cls,
        params: dict[str, Any],
        index_artifact: IndexArtifact,
    ) -> HybridRetriever:
        """Construct a HybridRetriever and its wired sub-retrievers.

        Each sub-retriever entry: ``{name, type, top_k, params}``.
        ``name`` defaults to ``type`` and must be unique within the
        hybrid (validated upstream in ``validate_config``).
        """
        sub_specs = params.get("retrievers", [])
        children: list[tuple[str, BaseRetriever, int]] = []
        for spec in sub_specs:
            sub_type = spec["type"]
            sub_name = spec.get("name", sub_type)
            sub_top_k = int(spec["top_k"])
            sub_params = spec.get("params", {})
            sub_cls = registry.get("retrieval", sub_type)
            sub_retriever: BaseRetriever = sub_cls(config=sub_params)
            cls._wire_retriever(sub_retriever, sub_type, index_artifact)
            children.append((sub_name, sub_retriever, sub_top_k))

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
        )
