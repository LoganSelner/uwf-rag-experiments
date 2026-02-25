"""Compositional RAG pipeline: config-driven component selection.

The single ``RAGPipeline`` class wires together a Retriever, Reranker, and Generator
based on the corresponding ``*_type`` fields in EvalSettings. Its ``name`` attribute
(e.g. ``"dense_none_stuff"``) encodes all three active components so run-directory
names remain unambiguous across experiments.

To add a new retriever type:
  1. Implement a class in components/retrievers.py satisfying the Retriever Protocol.
  2. Add a ``from_settings(cls, settings, store)`` classmethod.
  3. Register it in ``_RETRIEVERS`` below.

To add a new reranker type:
  1. Implement a class in components/rerankers.py satisfying the Reranker Protocol.
  2. Add a ``from_settings(cls, settings)`` classmethod.
  3. Register it in ``_RERANKERS`` below.

To add a new generator type:
  1. Implement a class in components/generators.py satisfying the Generator Protocol.
  2. Add a ``from_settings(cls, settings, llm)`` classmethod.
  3. Register it in ``_GENERATORS`` below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from .components import (
    Generator,
    NoReranker,
    Reranker,
    Retriever,
    SimpleRetriever,
    StuffGenerator,
)
from .config import Settings
from .providers import build_embeddings, build_llm


@dataclass(frozen=True)
class PipelineResult:
    """Output of a single pipeline invocation."""

    answer: str
    contexts: list[str]


_RETRIEVERS: dict[str, Callable[[Settings, Chroma], Retriever]] = {
    "dense": SimpleRetriever.from_settings,
}

_RERANKERS: dict[str, Callable[[Settings], Reranker]] = {
    "none": NoReranker.from_settings,
}

_GENERATORS: dict[str, Callable[[Settings, BaseChatModel], Generator]] = {
    "stuff": StuffGenerator.from_settings,
}


def _build_retriever(settings: Settings, store: Chroma) -> Retriever:
    """Resolve and construct the configured retriever."""
    key = settings.eval.retriever_type
    if key not in _RETRIEVERS:
        available = ", ".join(_RETRIEVERS)
        raise ValueError(f"Unknown retriever_type: {key!r}. Available: {available}")
    return _RETRIEVERS[key](settings, store)


def _build_reranker(settings: Settings) -> Reranker:
    """Resolve and construct the configured reranker."""
    key = settings.eval.reranker_type
    if key not in _RERANKERS:
        available = ", ".join(_RERANKERS)
        raise ValueError(f"Unknown reranker_type: {key!r}. Available: {available}")
    return _RERANKERS[key](settings)


def _build_generator(settings: Settings, llm: BaseChatModel) -> Generator:
    """Resolve and construct the configured generator."""
    key = settings.eval.generator_type
    if key not in _GENERATORS:
        available = ", ".join(_GENERATORS)
        raise ValueError(f"Unknown generator_type: {key!r}. Available: {available}")
    return _GENERATORS[key](settings, llm)


@dataclass(frozen=True, eq=False)  # eq=False: Chroma/BaseChatModel are not hashable
class RAGPipeline:
    """Config-driven RAG pipeline: Retriever → Reranker → Generator.

    ``name`` encodes the active configuration (e.g. ``"dense_none_stuff"``), and is
    used as the suffix of the run directory created by ``run_pipeline.run_once``.

    ``top_k`` controls how many context chunks are passed to the generator after
    reranking. The retriever fetches ``retrieval_k`` candidates; the reranker reorders
    them; then ``answer()`` truncates to ``top_k`` before generation.
    """

    name: str
    top_k: int
    retriever: Retriever
    reranker: Reranker
    generator: Generator

    def answer(self, question: str) -> PipelineResult:
        """Run retrieval, reranking, and generation for a single question."""
        docs = self.retriever.retrieve(question)
        docs = self.reranker.rerank(question, docs)
        docs = docs[: self.top_k]
        contexts = [doc.page_content for doc in docs]
        answer = self.generator.generate(question, contexts)
        return PipelineResult(answer=answer, contexts=contexts)

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        """Create a pipeline instance from fully loaded settings."""
        embeddings = build_embeddings(settings.models.embeddings)
        store = Chroma(
            collection_name=settings.index.collection_name,
            persist_directory=str(settings.index.persist_dir),
            embedding_function=embeddings,
        )
        retriever = _build_retriever(settings, store)
        reranker = _build_reranker(settings)
        llm = build_llm(settings.models.llm)
        generator = _build_generator(settings, llm)
        name = (
            f"{settings.eval.retriever_type}"
            f"_{settings.eval.reranker_type}"
            f"_{settings.eval.generator_type}"
        )
        return cls(
            name=name,
            top_k=settings.eval.top_k,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )
