"""Tests for src/components/chunkers.py — text chunking implementations."""

from __future__ import annotations

import hashlib

from components.chunkers import (
    CustomRecursiveChunker,
    LangChainRecursiveChunker,
    SemanticChunker,
)
from core.types import Document


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic offline embeddings for SemanticChunker tests.

    Returns varied per-text vectors (hash-derived) so the chunker's
    cosine-distance breakpoints are non-degenerate — no model download.
    Matches the ``embed_fn`` seam: ``list[str] -> list[list[float]]``.
    """
    out: list[list[float]] = []
    for t in texts:
        digest = hashlib.sha256(t.encode()).digest()
        out.append([digest[i] / 255.0 for i in range(16)])
    return out


class TestCustomRecursiveChunker:
    def test_single_short_document(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 512})
        docs = [Document(content="Short text.", metadata={})]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_respects_chunk_size(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 100, "chunk_overlap": 0})
        text = "word " * 200  # ~1000 chars
        docs = [Document(content=text, metadata={})]
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 110  # allow separator slop

    def test_overlap_produces_more_chunks(self) -> None:
        """Overlap creates redundancy, so more chunks than zero-overlap."""
        text = "word " * 200  # ~1000 chars
        docs = [Document(content=text.strip(), metadata={})]
        no_overlap = CustomRecursiveChunker({"chunk_size": 100, "chunk_overlap": 0})
        with_overlap = CustomRecursiveChunker({"chunk_size": 100, "chunk_overlap": 40})
        chunks_no = no_overlap.chunk(docs)
        chunks_yes = with_overlap.chunk(docs)
        assert len(chunks_yes) >= len(chunks_no)

    def test_multiple_documents(self, sample_documents: list[Document]) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 512})
        chunks = chunker.chunk(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_metadata_preserved(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 512})
        doc = Document(
            content="Some text.",
            metadata={"source_path": "test.pdf", "page_number": 3},
        )
        chunks = chunker.chunk([doc])
        assert chunks[0].metadata["source_path"] == "test.pdf"
        assert chunks[0].metadata["page_number"] == 3

    def test_chunk_index_in_metadata(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 20, "chunk_overlap": 0})
        doc = Document(content="Hello world. Goodbye world.", metadata={})
        chunks = chunker.chunk([doc])
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_chunk_ids_deterministic(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 512})
        doc = Document(content="Same text.", metadata={"source_path": "a.pdf"})
        chunks_1 = chunker.chunk([doc])
        chunks_2 = chunker.chunk([doc])
        assert chunks_1[0].chunk_id == chunks_2[0].chunk_id

    def test_chunk_ids_unique(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 30, "chunk_overlap": 0})
        text = "First sentence. Second sentence. Third sentence."
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_document(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 512})
        doc = Document(content="", metadata={})
        chunks = chunker.chunk([doc])
        assert chunks == []

    def test_custom_separators(self) -> None:
        chunker = CustomRecursiveChunker(
            {
                "chunk_size": 30,
                "chunk_overlap": 0,
                "separators": ["|||"],
            }
        )
        text = "part one is here|||part two is here|||part three is here"
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 2

    def test_large_document(self) -> None:
        chunker = CustomRecursiveChunker({"chunk_size": 200, "chunk_overlap": 20})
        text = "This is a sentence. " * 500  # ~10K chars
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 10
        # All content should be represented
        total_content = " ".join(c.content for c in chunks)
        assert "sentence" in total_content


class TestLangChainRecursiveChunker:
    def test_single_short_document(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 512})
        docs = [Document(content="Short text.", metadata={})]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_respects_chunk_size(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 100, "chunk_overlap": 0})
        text = "word " * 200
        docs = [Document(content=text, metadata={})]
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 110

    def test_metadata_preserved(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 512})
        doc = Document(
            content="Some text.",
            metadata={"source_path": "test.pdf", "page_number": 3},
        )
        chunks = chunker.chunk([doc])
        assert chunks[0].metadata["source_path"] == "test.pdf"
        assert chunks[0].metadata["page_number"] == 3

    def test_chunk_index_in_metadata(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 20, "chunk_overlap": 0})
        doc = Document(content="Hello world. Goodbye world.", metadata={})
        chunks = chunker.chunk([doc])
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_chunk_ids_deterministic(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 512})
        doc = Document(content="Same text.", metadata={"source_path": "a.pdf"})
        chunks_1 = chunker.chunk([doc])
        chunks_2 = chunker.chunk([doc])
        assert chunks_1[0].chunk_id == chunks_2[0].chunk_id

    def test_empty_document(self) -> None:
        chunker = LangChainRecursiveChunker({"chunk_size": 512})
        doc = Document(content="", metadata={})
        chunks = chunker.chunk([doc])
        assert chunks == []

    def test_custom_separators(self) -> None:
        chunker = LangChainRecursiveChunker(
            {"chunk_size": 30, "chunk_overlap": 0, "separators": ["|||"]}
        )
        text = "part one is here|||part two is here|||part three is here"
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 2

    def test_registered_as_recursive_langchain(self) -> None:
        from core.registry import registry

        cls = registry.get("chunking", "recursive_langchain")
        assert cls is LangChainRecursiveChunker

    def test_custom_registered_as_recursive_custom(self) -> None:
        from core.registry import registry

        cls = registry.get("chunking", "recursive_custom")
        assert cls is CustomRecursiveChunker


class TestSemanticChunker:
    _MULTI = (
        "Grade forgiveness lets students retake a course. "
        "Financial aid applications are due by March 1. "
        "The university police department is open 24 hours. "
        "Parking permits are required on all campus lots. "
        "The library extends its hours during finals week. "
        "Students must maintain a 2.0 GPA for good standing."
    )

    def _chunker(self, **params: object) -> SemanticChunker:
        # Inject deterministic offline embeddings via the test seam.
        return SemanticChunker(config=params, embed_fn=_stub_embed)

    def test_registered_as_semantic(self) -> None:
        from core.registry import registry

        assert registry.get("chunking", "semantic") is SemanticChunker

    def test_produces_chunks(self) -> None:
        chunker = self._chunker()
        chunks = chunker.chunk([Document(content=self._MULTI, metadata={})])
        assert len(chunks) >= 1
        assert all(c.content.strip() for c in chunks)

    def test_splits_at_low_threshold(self) -> None:
        # A low percentile threshold makes most gaps breakpoints → >1 chunk.
        chunker = self._chunker(
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=10,
        )
        chunks = chunker.chunk([Document(content=self._MULTI, metadata={})])
        assert len(chunks) > 1

    def test_metadata_preserved(self) -> None:
        chunker = self._chunker()
        doc = Document(
            content=self._MULTI,
            metadata={"source_path": "test.pdf", "page_number": 3},
        )
        chunks = chunker.chunk([doc])
        for c in chunks:
            assert c.metadata["source_path"] == "test.pdf"
            assert c.metadata["page_number"] == 3

    def test_chunk_index_sequential(self) -> None:
        chunker = self._chunker(
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=10,
        )
        chunks = chunker.chunk([Document(content=self._MULTI, metadata={})])
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_chunk_ids_deterministic(self) -> None:
        doc = Document(content=self._MULTI, metadata={"source_path": "a.pdf"})
        ids_1 = [c.chunk_id for c in self._chunker().chunk([doc])]
        ids_2 = [c.chunk_id for c in self._chunker().chunk([doc])]
        assert ids_1 == ids_2

    def test_empty_document(self) -> None:
        chunker = self._chunker()
        assert chunker.chunk([Document(content="", metadata={})]) == []

    def test_whitespace_only_document(self) -> None:
        chunker = self._chunker()
        assert chunker.chunk([Document(content="   \n\t ", metadata={})]) == []

    def test_single_sentence_falls_back(self) -> None:
        # Single sentence => no breakpoints to compute; fall back to one chunk.
        chunker = self._chunker()
        doc = Document(content="Only one sentence here.", metadata={})
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "Only one sentence here."

    def test_multiple_documents(self) -> None:
        chunker = self._chunker()
        docs = [
            Document(content=self._MULTI, metadata={"page_number": 1}),
            Document(content=self._MULTI, metadata={"page_number": 2}),
        ]
        chunks = chunker.chunk(docs)
        assert {c.metadata["page_number"] for c in chunks} == {1, 2}
