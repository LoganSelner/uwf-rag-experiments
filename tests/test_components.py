from __future__ import annotations

import dataclasses
from typing import ClassVar
from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from rag_testing.components import (
    Generator,
    NoReranker,
    RecursiveChunker,
    Reranker,
    Retriever,
    SimpleRetriever,
    StuffGenerator,
)
from rag_testing.components.generators import STUFF_PROMPT
from rag_testing.config import Settings

# ---------------------------------------------------------------------------
# Protocol conformance — structural typing: any class with the right signature works
# ---------------------------------------------------------------------------


def test_retriever_protocol_satisfied_by_conforming_class() -> None:
    class DummyRetriever:
        name: ClassVar[str] = "dummy"

        def retrieve(self, question: str) -> list[Document]:
            return [Document(page_content="chunk")]

    r: Retriever = DummyRetriever()
    docs = r.retrieve("Q?")
    assert len(docs) == 1
    assert docs[0].page_content == "chunk"


def test_reranker_protocol_satisfied_by_conforming_class() -> None:
    class DummyReranker:
        def rerank(self, question: str, docs: list[Document]) -> list[Document]:
            return docs[:1]

    docs = [Document(page_content="a"), Document(page_content="b")]
    r: Reranker = DummyReranker()
    result = r.rerank("Q?", docs)
    assert len(result) == 1
    assert result[0].page_content == "a"


def test_generator_protocol_satisfied_by_conforming_class() -> None:
    class DummyGenerator:
        def generate(self, question: str, contexts: list[str]) -> str:
            return f"answer:{question}"

    g: Generator = DummyGenerator()
    assert g.generate("Q?", ["ctx"]) == "answer:Q?"


# ---------------------------------------------------------------------------
# SimpleRetriever
# ---------------------------------------------------------------------------


def test_simple_retriever_delegates_to_similarity_search() -> None:
    doc = Document(page_content="hello")
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [doc]

    result = SimpleRetriever(store=mock_store, top_k=5).retrieve("test question")

    mock_store.similarity_search.assert_called_once_with("test question", k=5)
    assert result == [doc]


def test_simple_retriever_is_frozen() -> None:
    mock_store = MagicMock()
    r = SimpleRetriever(store=mock_store, top_k=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.top_k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NoReranker
# ---------------------------------------------------------------------------


def test_no_reranker_returns_same_list_object() -> None:
    docs = [Document(page_content="a"), Document(page_content="b")]
    result = NoReranker().rerank("Q?", docs)
    assert result is docs  # identity: no copy, no modification


def test_no_reranker_empty_list() -> None:
    assert NoReranker().rerank("Q?", []) == []


def test_no_reranker_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        NoReranker().x = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# StuffGenerator
# ---------------------------------------------------------------------------


def test_stuff_generator_default_prompt_has_expected_sections() -> None:
    """Default prompt surfaces context and question in expected sections."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="A")

    StuffGenerator(llm=mock_llm).generate("my question", ["my context"])

    prompt = mock_llm.invoke.call_args[0][0]
    assert "my context" in prompt
    assert "my question" in prompt
    assert "CONTEXT:" in prompt
    assert "QUESTION:" in prompt


def test_stuff_generator_joins_contexts_with_separator() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="answer")

    StuffGenerator(llm=mock_llm).generate("Q?", ["chunk one", "chunk two"])

    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "chunk one\n\n---\n\nchunk two" in prompt_arg


def test_stuff_generator_returns_content_attribute() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="My answer")

    result = StuffGenerator(llm=mock_llm).generate("Q?", ["ctx"])
    assert result == "My answer"


def test_stuff_generator_falls_back_to_str_when_no_content_attr() -> None:
    class PlainResponse:
        """No .content attribute — exercises the str() fallback."""

        def __str__(self) -> str:
            return "fallback string"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = PlainResponse()

    result = StuffGenerator(llm=mock_llm).generate("Q?", ["ctx"])
    assert result == "fallback string"


def test_stuff_generator_empty_contexts_produces_empty_stuffed_section() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="A")

    StuffGenerator(llm=mock_llm).generate("Q?", [])

    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "CONTEXT:\n\n" in prompt_arg


def test_stuff_generator_accepts_custom_prompt_template() -> None:
    """A caller can supply a custom template without subclassing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="A")

    template = "CTX: {context} | Q: {question}"
    StuffGenerator(llm=mock_llm, prompt_template=template).generate("hi", ["ctx"])

    prompt = mock_llm.invoke.call_args[0][0]
    assert prompt == "CTX: ctx | Q: hi"


def test_stuff_generator_is_frozen() -> None:
    mock_llm = MagicMock()
    gen = StuffGenerator(llm=mock_llm)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gen.llm = MagicMock()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


def test_recursive_chunker_splits_long_text() -> None:
    long_text = "word " * 300  # ~1 500 chars
    docs = [Document(page_content=long_text)]
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=0).chunk(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200


def test_recursive_chunker_empty_input_returns_empty() -> None:
    assert RecursiveChunker(chunk_size=500, chunk_overlap=50).chunk([]) == []


def test_recursive_chunker_short_text_stays_single() -> None:
    docs = [Document(page_content="Short text.")]
    chunks = RecursiveChunker(chunk_size=500, chunk_overlap=50).chunk(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_recursive_chunker_is_frozen() -> None:
    r = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.chunk_size = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# from_settings() — each component constructs itself from Settings
# ---------------------------------------------------------------------------


def test_simple_retriever_from_settings_uses_retrieval_k(settings: Settings) -> None:
    mock_store = MagicMock()
    r = SimpleRetriever.from_settings(settings, mock_store)
    assert isinstance(r, SimpleRetriever)
    assert r.store is mock_store
    assert r.top_k == settings.eval.retrieval_k


def test_no_reranker_from_settings_returns_instance(settings: Settings) -> None:
    assert isinstance(NoReranker.from_settings(settings), NoReranker)


def test_stuff_generator_from_settings_uses_default_prompt(settings: Settings) -> None:
    mock_llm = MagicMock()
    g = StuffGenerator.from_settings(settings, mock_llm)
    assert isinstance(g, StuffGenerator)
    assert g.llm is mock_llm
    assert g.prompt_template == STUFF_PROMPT  # settings.eval.prompt_template == ""


def test_stuff_generator_from_settings_uses_custom_prompt(settings: Settings) -> None:
    custom = "CTX:{context} Q:{question}"
    new_settings = dataclasses.replace(
        settings,
        eval=dataclasses.replace(settings.eval, prompt_template=custom),
    )
    mock_llm = MagicMock()
    g = StuffGenerator.from_settings(new_settings, mock_llm)
    assert g.prompt_template == custom


def test_recursive_chunker_from_settings_uses_index_params(settings: Settings) -> None:
    c = RecursiveChunker.from_settings(settings)
    assert isinstance(c, RecursiveChunker)
    assert c.chunk_size == settings.index.chunk_size
    assert c.chunk_overlap == settings.index.chunk_overlap
