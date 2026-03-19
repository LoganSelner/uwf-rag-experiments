"""Chunker implementations.

Each chunker splits Documents into Chunks with unique IDs and
preserved metadata.
"""

from __future__ import annotations

import hashlib
from typing import Any

from components.base import BaseChunker
from core.registry import registry
from core.types import Chunk, Document


@registry.register("chunking", "recursive")
class RecursiveChunker(BaseChunker):
    """Recursive character text splitter.

    Splits text using a hierarchy of separators, falling back to
    finer-grained splits when chunks exceed the target size.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._chunk_size: int = self.config.get("chunk_size", 512)
        self._chunk_overlap: int = self.config.get("chunk_overlap", 50)
        self._separators: list[str] = self.config.get(
            "separators", ["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            texts = self._split_text(doc.content, self._separators)
            merged = self._merge_splits(texts)
            for i, text in enumerate(merged):
                chunk_id = self._make_chunk_id(text, doc.metadata, i)
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_id=chunk_id,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split as last resort
            splits = list(text)
        else:
            splits = text.split(separator)

        good_splits: list[str] = []
        current = ""

        for piece in splits:
            # Re-attach the separator to the piece (except for empty separator)
            candidate = (current + separator + piece) if current else piece
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                # If this single piece is still too large, split further
                if len(piece) > self._chunk_size and remaining_separators:
                    good_splits.extend(self._split_text(piece, remaining_separators))
                else:
                    current = piece

        if current:
            good_splits.append(current)

        return good_splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits into chunks with overlap."""
        if not splits:
            return []

        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len > self._chunk_size and current_parts:
                merged.append("".join(current_parts))

                # Keep overlap from the end of current_parts
                overlap_parts: list[str] = []
                overlap_len = 0
                for part in reversed(current_parts):
                    if overlap_len + len(part) > self._chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += len(part)

                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(split)
            current_len += split_len

        if current_parts:
            merged.append("".join(current_parts))

        return merged

    @staticmethod
    def _make_chunk_id(
        content: str,
        metadata: dict[str, Any],
        index: int,
    ) -> str:
        """Generate a deterministic chunk ID from content and position."""
        source = metadata.get("source_path", "")
        page = metadata.get("page_number", 0)
        key = f"{source}:{page}:{index}:{content[:64]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
