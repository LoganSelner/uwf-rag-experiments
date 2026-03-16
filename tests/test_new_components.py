"""Tests for Phase 1 component implementations (new framework).

Tests each component in isolation against its ABC contract.
Uses the small test PDF fixture for ingestor tests.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.types import Chunk, Document, EmbeddedChunk, RetrievedChunk

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample.pdf"


# -----------------------------------------------------------------------
# PDFIngestor
# -----------------------------------------------------------------------


class TestPDFIngestor:
    def test_ingest_returns_documents(self) -> None:
        from components.ingestors import PDFIngestor

        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(SAMPLE_PDF))
        assert len(docs) == 2  # Two pages in our test PDF
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_content(self) -> None:
        from components.ingestors import PDFIngestor

        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(SAMPLE_PDF))
        assert "Grade Forgiveness" in docs[0].content
        assert "Police Department" in docs[1].content

    def test_documents_have_metadata(self) -> None:
        from components.ingestors import PDFIngestor

        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(SAMPLE_PDF))
        assert docs[0].metadata["page_number"] == 1
        assert docs[1].metadata["page_number"] == 2
        assert docs[0].metadata["source_path"] == str(SAMPLE_PDF)


# -----------------------------------------------------------------------
# RecursiveChunker
# -----------------------------------------------------------------------


class TestRecursiveChunker:
    def test_chunks_produced(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 200, "chunk_overlap": 20})
        docs = [Document(content="A" * 500, metadata={"source_path": "test.pdf"})]
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_unique(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 200, "chunk_overlap": 20})
        docs = [Document(content="A" * 500, metadata={"source_path": "test.pdf"})]
        chunks = chunker.chunk(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_carried_forward(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 200, "chunk_overlap": 20})
        docs = [
            Document(
                content="Word " * 100,
                metadata={"source_path": "test.pdf", "page_number": 3},
            )
        ]
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert chunk.metadata["source_path"] == "test.pdf"
            assert chunk.metadata["page_number"] == 3
            assert "chunk_index" in chunk.metadata

    def test_respects_chunk_size(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 100, "chunk_overlap": 10})
        docs = [
            Document(
                content="This is a sentence. " * 50,
                metadata={"source_path": "test.pdf"},
            )
        ]
        chunks = chunker.chunk(docs)
        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_empty_input(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 200, "chunk_overlap": 20})
        assert chunker.chunk([]) == []

    def test_short_text_single_chunk(self) -> None:
        from components.chunkers import RecursiveChunker

        chunker = RecursiveChunker(config={"chunk_size": 500, "chunk_overlap": 50})
        docs = [Document(content="Short text.", metadata={"source_path": "test.pdf"})]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."


# -----------------------------------------------------------------------
# FAISSVectorStore
# -----------------------------------------------------------------------


class TestFAISSVectorStore:
    def _make_chunks(self, n: int, dim: int = 8) -> list[EmbeddedChunk]:
        """Create n embedded chunks with random normalized vectors."""
        import numpy as np

        rng = np.random.default_rng(42)
        chunks = []
        for i in range(n):
            vec = rng.standard_normal(dim).astype(float)
            vec = vec / np.linalg.norm(vec)
            chunk = Chunk(
                content=f"chunk {i}",
                chunk_id=f"c{i}",
                metadata={"source_name": "handbook" if i % 2 == 0 else "kb"},
            )
            chunks.append(EmbeddedChunk(chunk=chunk, embedding=vec.tolist()))
        return chunks

    def test_add_and_search(self) -> None:
        from components.vectorstores import FAISSVectorStore

        store = FAISSVectorStore()
        embedded = self._make_chunks(10)
        store.add(embedded)

        query_vec = embedded[0].embedding
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        # First result should be the exact match
        assert results[0].chunk.chunk_id == "c0"

    def test_filtered_search(self) -> None:
        from components.vectorstores import FAISSVectorStore

        store = FAISSVectorStore()
        embedded = self._make_chunks(20)
        store.add(embedded)

        query_vec = embedded[0].embedding
        results = store.search(query_vec, top_k=5, filters={"source_name": "kb"})
        assert all(r.chunk.metadata["source_name"] == "kb" for r in results)

    def test_save_and_load(self, tmp_path: Path) -> None:
        from components.vectorstores import FAISSVectorStore

        store = FAISSVectorStore()
        embedded = self._make_chunks(10)
        store.add(embedded)

        save_dir = str(tmp_path / "test_index")
        store.save(save_dir)

        loaded = FAISSVectorStore()
        loaded.load(save_dir)

        query_vec = embedded[0].embedding
        original_results = store.search(query_vec, top_k=3)
        loaded_results = loaded.search(query_vec, top_k=3)
        assert [r.chunk.chunk_id for r in original_results] == [
            r.chunk.chunk_id for r in loaded_results
        ]

    def test_empty_search(self) -> None:
        from components.vectorstores import FAISSVectorStore

        store = FAISSVectorStore()
        results = store.search([0.1, 0.2, 0.3], top_k=5)
        assert results == []


# -----------------------------------------------------------------------
# DenseRetriever
# -----------------------------------------------------------------------


class TestDenseRetriever:
    def test_retrieve_requires_embedder(self) -> None:
        from components.retrievers import DenseRetriever

        retriever = DenseRetriever()
        with pytest.raises(RuntimeError, match="Embedder not set"):
            retriever.retrieve("test", top_k=5)

    def test_retrieve_requires_vectorstore(self) -> None:
        from components.retrievers import DenseRetriever

        retriever = DenseRetriever()
        retriever.set_embedder(MagicMock())
        with pytest.raises(RuntimeError, match="VectorStore not set"):
            retriever.retrieve("test", top_k=5)

    def test_retrieve_wires_correctly(self) -> None:
        from components.retrievers import DenseRetriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_chunk = Chunk(content="result", chunk_id="r1", metadata={})
        mock_store = MagicMock()
        mock_store.search.return_value = [RetrievedChunk(chunk=mock_chunk, score=0.9)]

        retriever = DenseRetriever()
        retriever.set_embedder(mock_embedder)
        retriever.set_vectorstore(mock_store)

        results = retriever.retrieve("test query", top_k=5)
        assert len(results) == 1
        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_store.search.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3], top_k=5, filters=None
        )

    def test_default_filters_merged(self) -> None:
        from components.retrievers import DenseRetriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1]
        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = DenseRetriever()
        retriever.set_embedder(mock_embedder)
        retriever.set_vectorstore(mock_store)
        retriever.set_default_filters({"source_name": "handbook"})

        retriever.retrieve("test", top_k=3)
        mock_store.search.assert_called_once_with(
            query_vector=[0.1], top_k=3, filters={"source_name": "handbook"}
        )
