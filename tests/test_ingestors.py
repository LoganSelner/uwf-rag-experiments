"""Tests for src/components/ingestors.py — PDFIngestor with real fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.components.ingestors import PDFIngestor
from ragbench.core.types import Document


class TestPDFIngestor:
    def test_returns_documents(self, fixtures_dir: Path) -> None:
        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(fixtures_dir / "sample.pdf"))
        assert all(isinstance(d, Document) for d in docs)

    def test_page_count(self, fixtures_dir: Path) -> None:
        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(fixtures_dir / "sample.pdf"))
        assert len(docs) >= 1

    def test_content_nonempty(self, fixtures_dir: Path) -> None:
        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(fixtures_dir / "sample.pdf"))
        for doc in docs:
            assert doc.content.strip()

    def test_metadata_has_source_and_page(self, fixtures_dir: Path) -> None:
        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(fixtures_dir / "sample.pdf"))
        for doc in docs:
            assert "source_path" in doc.metadata
            assert "page_number" in doc.metadata

    def test_pages_are_1_based(self, fixtures_dir: Path) -> None:
        ingestor = PDFIngestor()
        docs = ingestor.ingest(str(fixtures_dir / "sample.pdf"))
        assert docs[0].metadata["page_number"] == 1

    def test_nonexistent_file_raises(self) -> None:
        import fitz

        ingestor = PDFIngestor()
        with pytest.raises((FileNotFoundError, fitz.FileNotFoundError)):
            ingestor.ingest("/nonexistent/path.pdf")
