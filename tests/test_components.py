from __future__ import annotations

import dataclasses
from typing import ClassVar
from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest
from rank_bm25 import BM25Okapi

from rag_testing.components import (
    CrossEncoderReranker,
    Generator,
    HybridRetriever,
    MMRRetriever,
    NoReranker,
    RecursiveChunker,
    Reranker,
    Retriever,
    SimpleRetriever,
    StuffGenerator,
)
from rag_testing.components.generators import STUFF_PROMPT
from rag_testing.components.retrievers import _reciprocal_rank_fusion, _tokenize
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

    result = SimpleRetriever(store=mock_store, k=5).retrieve("test question")

    mock_store.similarity_search.assert_called_once_with("test question", k=5)
    assert result == [doc]


def test_simple_retriever_is_frozen() -> None:
    mock_store = MagicMock()
    r = SimpleRetriever(store=mock_store, k=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MMRRetriever
# ---------------------------------------------------------------------------


def test_mmr_retriever_delegates_to_mmr_search() -> None:
    doc = Document(page_content="hello")
    mock_store = MagicMock()
    mock_store.max_marginal_relevance_search.return_value = [doc]

    result = MMRRetriever(store=mock_store, k=5, fetch_k=20, lambda_mult=0.7).retrieve(
        "test question"
    )

    mock_store.max_marginal_relevance_search.assert_called_once_with(
        "test question", k=5, fetch_k=20, lambda_mult=0.7
    )
    assert result == [doc]


def test_mmr_retriever_is_frozen() -> None:
    mock_store = MagicMock()
    r = MMRRetriever(store=mock_store, k=3, fetch_k=20, lambda_mult=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

_CORPUS = (
    Document(page_content="alpha bravo charlie"),
    Document(page_content="delta echo foxtrot"),
    Document(page_content="golf hotel india"),
)
_TOKENIZED = [_tokenize(d.page_content) for d in _CORPUS]


def _make_hybrid(
    mock_store: MagicMock, *, k: int = 2, alpha: float = 0.5
) -> HybridRetriever:
    bm25 = BM25Okapi(_TOKENIZED)
    return HybridRetriever(
        store=mock_store, bm25=bm25, corpus_docs=_CORPUS, k=k, alpha=alpha
    )


def test_hybrid_retriever_fuses_dense_and_bm25() -> None:
    mock_store = MagicMock()
    # Dense returns first two docs (alpha bravo, delta echo).
    mock_store.similarity_search.return_value = [_CORPUS[0], _CORPUS[1]]

    result = _make_hybrid(mock_store, k=2, alpha=0.5).retrieve("alpha")

    mock_store.similarity_search.assert_called_once_with("alpha", k=2)
    assert len(result) <= 2
    # "alpha bravo charlie" should rank highest: it appears in dense results
    # AND BM25 ranks it first for the query "alpha".
    assert result[0].page_content == "alpha bravo charlie"


def test_hybrid_retriever_alpha_1_favors_dense() -> None:
    mock_store = MagicMock()
    # Dense returns delta, alpha (this order).
    mock_store.similarity_search.return_value = [_CORPUS[1], _CORPUS[0]]

    result = _make_hybrid(mock_store, k=2, alpha=1.0).retrieve("alpha")

    # alpha=1.0 zeroes BM25 weight, so dense order is preserved.
    assert result[0].page_content == "delta echo foxtrot"
    assert result[1].page_content == "alpha bravo charlie"


def test_hybrid_retriever_alpha_0_favors_bm25() -> None:
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [_CORPUS[1], _CORPUS[0]]

    result = _make_hybrid(mock_store, k=2, alpha=0.0).retrieve("alpha")

    # alpha=0.0 zeroes dense weight; BM25 ranks "alpha bravo charlie" first.
    assert result[0].page_content == "alpha bravo charlie"


def test_hybrid_retriever_deduplicates() -> None:
    mock_store = MagicMock()
    # Dense returns the same doc that BM25 will also rank first.
    mock_store.similarity_search.return_value = [_CORPUS[0]]

    result = _make_hybrid(mock_store, k=3, alpha=0.5).retrieve("alpha")

    contents = [d.page_content for d in result]
    assert len(contents) == len(set(contents)), "duplicate documents in results"


def test_hybrid_retriever_is_frozen() -> None:
    mock_store = MagicMock()
    r = _make_hybrid(mock_store)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------


def test_rrf_shared_doc_ranks_highest() -> None:
    shared = Document(page_content="shared")
    dense_only = Document(page_content="dense_only")
    bm25_only = Document(page_content="bm25_only")

    result = _reciprocal_rank_fusion(
        dense_docs=[shared, dense_only],
        bm25_docs=[shared, bm25_only],
        alpha=0.5,
        k=3,
    )
    assert result[0].page_content == "shared"
    assert len(result) == 3


def test_rrf_returns_at_most_k() -> None:
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    result = _reciprocal_rank_fusion(dense_docs=docs, bm25_docs=docs, alpha=0.5, k=2)
    assert len(result) == 2


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
# CrossEncoderReranker
# ---------------------------------------------------------------------------


def test_cross_encoder_reranker_reorders_by_score() -> None:
    docs = [
        Document(page_content="low"),
        Document(page_content="high"),
        Document(page_content="mid"),
    ]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    result = CrossEncoderReranker(model=mock_model).rerank("Q?", docs)

    mock_model.predict.assert_called_once_with(
        [("Q?", "low"), ("Q?", "high"), ("Q?", "mid")]
    )
    assert [d.page_content for d in result] == ["high", "mid", "low"]


def test_cross_encoder_reranker_empty_list() -> None:
    mock_model = MagicMock()
    result = CrossEncoderReranker(model=mock_model).rerank("Q?", [])
    assert result == []
    mock_model.predict.assert_not_called()


def test_cross_encoder_reranker_is_frozen() -> None:
    mock_model = MagicMock()
    r = CrossEncoderReranker(model=mock_model)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.model = MagicMock()  # type: ignore[misc]


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


def test_stuff_generator_retries_on_transient_error() -> None:
    """Transient errors are retried; the final successful response is returned."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        ConnectionError("transient"),
        ConnectionError("transient again"),
        MagicMock(content="recovered answer"),
    ]

    result = StuffGenerator(llm=mock_llm).generate("Q?", ["ctx"])
    assert result == "recovered answer"
    assert mock_llm.invoke.call_count == 3


def test_stuff_generator_no_retry_on_value_error() -> None:
    """Deterministic errors like ValueError are not retried."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ValueError("bad template")

    with pytest.raises(ValueError, match="bad template"):
        StuffGenerator(llm=mock_llm).generate("Q?", ["ctx"])
    assert mock_llm.invoke.call_count == 1


def test_stuff_generator_reraises_after_max_retries() -> None:
    """After exhausting retries, the original exception is reraised."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ConnectionError("persistent failure")

    with pytest.raises(ConnectionError, match="persistent failure"):
        StuffGenerator(llm=mock_llm).generate("Q?", ["ctx"])
    assert mock_llm.invoke.call_count == 4  # 1 initial + 3 retries


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
    assert r.k == settings.eval.retrieval_k


def test_mmr_retriever_from_settings(settings: Settings) -> None:
    mock_store = MagicMock()
    r = MMRRetriever.from_settings(settings, mock_store)
    assert isinstance(r, MMRRetriever)
    assert r.store is mock_store
    assert r.k == settings.eval.retrieval_k
    assert r.lambda_mult == settings.eval.mmr_lambda
    assert r.fetch_k == settings.eval.mmr_fetch_k


def test_hybrid_retriever_from_settings(settings: Settings) -> None:
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "documents": ["chunk one", "chunk two"],
        "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        "ids": ["id1", "id2"],
    }
    s = dataclasses.replace(
        settings,
        eval=dataclasses.replace(
            settings.eval, retriever_type="hybrid", hybrid_alpha=0.7
        ),
    )
    r = HybridRetriever.from_settings(s, mock_store)
    assert isinstance(r, HybridRetriever)
    assert r.store is mock_store
    assert r.k == s.eval.retrieval_k
    assert r.alpha == 0.7
    assert len(r.corpus_docs) == 2
    mock_store.get.assert_called_once()


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


def test_cross_encoder_reranker_from_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_ce = MagicMock()
    monkeypatch.setattr("rag_testing.components.rerankers.CrossEncoder", mock_ce)
    s = dataclasses.replace(
        settings,
        eval=dataclasses.replace(
            settings.eval,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
    )
    r = CrossEncoderReranker.from_settings(s)
    assert isinstance(r, CrossEncoderReranker)
    mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_cross_encoder_reranker_from_settings_empty_model_raises(
    settings: Settings,
) -> None:
    s = dataclasses.replace(
        settings,
        eval=dataclasses.replace(settings.eval, reranker_model=""),
    )
    with pytest.raises(ValueError, match="reranker_model must be set"):
        CrossEncoderReranker.from_settings(s)


def test_recursive_chunker_from_settings_uses_index_params(settings: Settings) -> None:
    c = RecursiveChunker.from_settings(settings)
    assert isinstance(c, RecursiveChunker)
    assert c.chunk_size == settings.index.chunk_size
    assert c.chunk_overlap == settings.index.chunk_overlap
