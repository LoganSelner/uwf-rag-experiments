"""Tests for src/components/chunkers.py — recursive text chunking."""

from __future__ import annotations

from components.chunkers import RecursiveChunker
from core.types import Document


class TestRecursiveChunker:
    def test_single_short_document(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 512})
        docs = [Document(content="Short text.", metadata={})]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_respects_chunk_size(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 100, "chunk_overlap": 0})
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
        no_overlap = RecursiveChunker({"chunk_size": 100, "chunk_overlap": 0})
        with_overlap = RecursiveChunker({"chunk_size": 100, "chunk_overlap": 40})
        chunks_no = no_overlap.chunk(docs)
        chunks_yes = with_overlap.chunk(docs)
        assert len(chunks_yes) >= len(chunks_no)

    def test_multiple_documents(self, sample_documents: list[Document]) -> None:
        chunker = RecursiveChunker({"chunk_size": 512})
        chunks = chunker.chunk(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_metadata_preserved(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 512})
        doc = Document(
            content="Some text.",
            metadata={"source_path": "test.pdf", "page_number": 3},
        )
        chunks = chunker.chunk([doc])
        assert chunks[0].metadata["source_path"] == "test.pdf"
        assert chunks[0].metadata["page_number"] == 3

    def test_chunk_index_in_metadata(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 20, "chunk_overlap": 0})
        doc = Document(content="Hello world. Goodbye world.", metadata={})
        chunks = chunker.chunk([doc])
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_chunk_ids_deterministic(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 512})
        doc = Document(content="Same text.", metadata={"source_path": "a.pdf"})
        chunks_1 = chunker.chunk([doc])
        chunks_2 = chunker.chunk([doc])
        assert chunks_1[0].chunk_id == chunks_2[0].chunk_id

    def test_chunk_ids_unique(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 30, "chunk_overlap": 0})
        text = "First sentence. Second sentence. Third sentence."
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_document(self) -> None:
        chunker = RecursiveChunker({"chunk_size": 512})
        doc = Document(content="", metadata={})
        chunks = chunker.chunk([doc])
        assert chunks == []

    def test_custom_separators(self) -> None:
        chunker = RecursiveChunker(
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
        chunker = RecursiveChunker({"chunk_size": 200, "chunk_overlap": 20})
        text = "This is a sentence. " * 500  # ~10K chars
        doc = Document(content=text, metadata={})
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 10
        # All content should be represented
        total_content = " ".join(c.content for c in chunks)
        assert "sentence" in total_content
