"""Embedder implementations.

Each embedder produces vector representations of text chunks and queries.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sentence_transformers import SentenceTransformer
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from components.base import BaseEmbedder
from core.registry import registry
from core.types import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)

_NON_RETRYABLE = (TypeError, ValueError, KeyError, AttributeError, SyntaxError)

_retry_decorator = retry(
    retry=retry_if_exception(lambda e: not isinstance(e, _NON_RETRYABLE)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


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


# Map legacy model names to current google-genai equivalents
_GOOGLE_MODEL_MAP: dict[str, str] = {
    "models/embedding-001": "gemini-embedding-001",
}


@registry.register("embedding", "google")
class GoogleEmbedder(BaseEmbedder):
    """Embeds text using Google's Gemini embedding API.

    Wraps the ``google-genai`` SDK.  Uses separate task types for
    document embedding (indexing) and query embedding (retrieval) to
    optimise vector quality per the API's recommendations.

    Config params:
        model_name: Embedding model (default: "gemini-embedding-001").
        batch_size: Texts per API call (default: 100).
        task_type_document: Task type for embed_chunks (default: "RETRIEVAL_DOCUMENT").
        task_type_query: Task type for embed_query (default: "RETRIEVAL_QUERY").
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        raw_name = self.config.get("model_name", "gemini-embedding-001")
        self._model_name: str = _GOOGLE_MODEL_MAP.get(raw_name, raw_name)
        self._batch_size: int = self.config.get("batch_size", 100)
        self._task_type_document: str = self.config.get(
            "task_type_document", "RETRIEVAL_DOCUMENT"
        )
        self._task_type_query: str = self.config.get(
            "task_type_query", "RETRIEVAL_QUERY"
        )

        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is "
                "required. Set it in .env or your shell environment."
            )

        from google import genai

        self._client: Any = genai.Client(api_key=api_key)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._embed_with_retry(batch, self._task_type_document)
            all_embeddings.extend(e.values for e in result.embeddings)

        return [
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, all_embeddings, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        result = self._embed_with_retry([query], self._task_type_query)
        return result.embeddings[0].values

    @_retry_decorator
    def _embed_with_retry(self, texts: list[str], task_type: str) -> Any:
        """Call the Google embedding API with automatic retry."""
        return self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config={"task_type": task_type},
        )
