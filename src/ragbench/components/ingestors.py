"""Ingestor implementations.

Each ingestor reads a specific file format and produces Documents.
"""

from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF

from ragbench.components.base import BaseIngestor
from ragbench.core.registry import registry
from ragbench.core.types import Document


@registry.register("ingest", "pdf")
class PDFIngestor(BaseIngestor):
    """Extracts text from PDF files using PyMuPDF, one Document per page."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def ingest(self, path: str) -> list[Document]:
        documents: list[Document] = []
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if not text.strip():
                    continue
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source_path": path,
                            "page_number": page_num + 1,
                        },
                    )
                )
        finally:
            doc.close()
        return documents
