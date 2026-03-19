"""Embedder implementations.

Each embedder produces vector representations of text chunks and queries.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from components.base import BaseEmbedder
from core.registry import registry
from core.types import Chunk, EmbeddedChunk


@registry.register("embedding", "huggingface")
class HuggingFaceEmbedder(BaseEmbedder):
    """Embeds text using a HuggingFace sentence-transformers model.

    Config params:
        model_name: Model ID (default: "BAAI/bge-m3")
        normalize: Whether to L2-normalize embeddings (default: True)
        batch_size: Batch size for encoding (default: 32)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        model_name = self.config.get("model_name", "BAAI/bge-m3")
        self._normalize: bool = self.config.get("normalize", True)
        self._batch_size: int = self.config.get("batch_size", 32)
        self._model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [c.content for c in chunks]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [
            EmbeddedChunk(chunk=chunk, embedding=vec.tolist())
            for chunk, vec in zip(chunks, embeddings, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        vec = self._model.encode(
            query,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vec.tolist()
