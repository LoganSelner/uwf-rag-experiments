"""Query pipeline.

Processes queries through the linear RAG pipeline:
  query → transform → retrieve → rerank → (prompt → generate)

Supports retrieval-only mode (skips generation) for fast
retriever evaluation without LLM cost.
"""

from __future__ import annotations

import logging

from ragbench.components.base import (
    BaseGenerator,
    BasePromptTemplate,
    BaseQueryTransformer,
    BaseReranker,
    BaseRetriever,
)
from ragbench.components.build import BuildContext, IndexHandle
from ragbench.components.generators import build_generator
from ragbench.core.config import QueryConfig
from ragbench.core.registry import registry
from ragbench.core.types import Chunk, GenerationResult, Message, RetrievedChunk

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
        index_artifact: IndexHandle,
        retrieval_only: bool = False,
    ) -> QueryPipeline:
        """Build a QueryPipeline from config and a pre-built index.

        Args:
            config: The query section of the experiment config.
            index_artifact: The populated index (an ``IndexArtifact``, accepted
                structurally as an ``IndexHandle``).
            retrieval_only: If True, skip generator/prompt construction.
        """
        # The build context carries the runtime dependencies components may
        # need (the populated index, the generator factory, this query stack)
        # plus the registry for recursive child construction.
        ctx = BuildContext(
            registry=registry,
            index=index_artifact,
            query=config,
            make_generator=build_generator,
        )

        # Query transformer — its reasoning LLM (if any) is built via the
        # shared factory and injected; the transformer never touches the registry.
        qt_type = config.query_transform.type or "passthrough"
        qt_cls = registry.get("query_transform", qt_type)
        qt_generator = (
            ctx.make_generator(config.query_transform.generator)
            if config.query_transform.generator.provider
            else None
        )
        query_transformer: BaseQueryTransformer = qt_cls(
            config=config.query_transform.params,
            generator=qt_generator,
        )

        # Retriever — each retriever's ``build`` pulls its own index
        # dependencies from the context; the pipeline never branches on the
        # concrete retriever type (hybrid recurses to build its children).
        retriever_cls = registry.get("retrieval", config.retrieval.type)
        retriever: BaseRetriever = retriever_cls.build(
            params=config.retrieval.params,
            default_filters=config.retrieval.filters or None,
            ctx=ctx,
        )

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
                # Pass the prompt config straight through (minus the impl
                # selector ``type``) so a new PromptConfig field can't be
                # silently dropped at this seam; ChatPromptTemplate.Params
                # (extra="forbid") owns which keys are valid.
                prompt_template = prompt_cls(
                    config=config.prompt.model_dump(exclude={"type"})
                )

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
        injected_contexts: list[str] | None = None,
    ) -> GenerationResult:
        """Run the query pipeline end-to-end.

        Args:
            query: The user's question.
            history: Optional conversation history for context-aware transforms.
            injected_contexts: Optional passages placed into the generation
                context *alongside* the retrieved chunks (the knowledge-conflict
                probe, item 15). They reach the generator's prompt but never enter
                ``result.retrieved_chunks``, so they cannot pollute retrieval
                metrics. Inert (None) on the normal path.

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

        # 5. Format prompt and generate. The conflict probe (item 15) may inject
        #    a contradicting passage into the prompt context — prepended (the
        #    position most likely to bias a position-sensitive model, the stronger
        #    test) so it sits alongside the real chunks the generator sees. Crucial:
        #    result.retrieved_chunks stays the REAL reranked set, so the injected
        #    passage never enters context-precision/recall (mirrors how web_search
        #    keeps retrieved_chunks empty).
        if self._prompt_template is None or self._generator is None:
            raise RuntimeError(
                "Generator and prompt template are required "
                "for non-retrieval-only mode. Set "
                "retrieval_only=True or configure generation."
            )

        context_chunks = reranked
        if injected_contexts:
            injected = [
                RetrievedChunk(
                    chunk=Chunk(
                        content=text,
                        chunk_id=f"__injected__{i}",
                        metadata={"source_name": "__injected__"},
                    ),
                    score=1.0,
                    retrieval_method="injected",
                )
                for i, text in enumerate(injected_contexts)
            ]
            context_chunks = injected + reranked

        prompt = self._prompt_template.format(query, context_chunks, history)
        result = self._generator.generate(prompt)
        result.query = query
        result.retrieved_chunks = reranked
        return result
