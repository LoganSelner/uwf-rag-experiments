"""Embedder implementations.

Each embedder produces vector representations of text chunks and queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

import requests
from sentence_transformers import SentenceTransformer

from ragbench.components.base import BaseEmbedder, ComponentParams
from ragbench.core.registry import registry
from ragbench.core.retry import retry_transient
from ragbench.core.types import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelPrompts:
    """Canonical encode prompts for an embedding-model family.

    ``query`` is prepended to query text, ``document`` to indexed text. Either
    may be ``None`` (e.g. BGE-v1.5 instructs only the query side).
    """

    query: str | None = None
    document: str | None = None


# Curated, declarative table of the asymmetric query/document prompts that
# popular sentence-transformers retrieval models expect, applied automatically
# (``auto_prompt``) so a model swapped in via config is wired the way it is used
# in practice — not silently mis-prompted. Keyed by a lowercased ``model_name``
# prefix; longest match wins. Models absent from the table (e.g. ``bge-m3``,
# which needs no prefix) get no prompt. Explicit ``query_prompt`` /
# ``document_prompt`` params always override this table.
#
# Sources: intfloat E5 model cards require "query: "/"passage: "; BAAI BGE-v1.5
# English cards recommend the query instruction below.
_BGE_V15_QUERY = "Represent this sentence for searching relevant passages: "
_MODEL_PROMPT_DEFAULTS: dict[str, _ModelPrompts] = {
    "intfloat/e5-": _ModelPrompts(query="query: ", document="passage: "),
    "intfloat/multilingual-e5": _ModelPrompts(query="query: ", document="passage: "),
    "baai/bge-small-en-v1.5": _ModelPrompts(query=_BGE_V15_QUERY),
    "baai/bge-base-en-v1.5": _ModelPrompts(query=_BGE_V15_QUERY),
    "baai/bge-large-en-v1.5": _ModelPrompts(query=_BGE_V15_QUERY),
}


def _match_family(model_name: str) -> _ModelPrompts | None:
    """Case-insensitive longest-prefix match against ``_MODEL_PROMPT_DEFAULTS``."""
    key = model_name.lower()
    best: tuple[int, _ModelPrompts] | None = None
    for prefix, prompts in _MODEL_PROMPT_DEFAULTS.items():
        if key.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), prompts)
    return best[1] if best else None


def resolve_hf_prompts(
    model_name: str,
    query_prompt: str | None,
    document_prompt: str | None,
    auto_prompt: bool,
) -> tuple[str | None, str | None]:
    """Resolve the effective ``(query, document)`` encode prompts.

    Explicit params win; otherwise, when ``auto_prompt`` is set, fall back to the
    known-family default; otherwise ``None`` (no prompt — the historical
    behavior). A pure function of its inputs, so the embedder and any audit agree
    on what a given ``model_name`` resolves to. The result is therefore a
    deterministic function of the (fingerprinted) ``model_name`` plus the
    (git-tracked) table, so an auto-applied prompt stays reproducible without
    living in the config.
    """
    family = _match_family(model_name) if auto_prompt else None
    q = query_prompt if query_prompt is not None else (family.query if family else None)
    d = (
        document_prompt
        if document_prompt is not None
        else (family.document if family else None)
    )
    return q, d


@registry.register("embedding", "huggingface")
class HuggingFaceEmbedder(BaseEmbedder):
    """Embeds text using a HuggingFace sentence-transformers model.

    Config params:
        model_name: Model ID (default: "BAAI/bge-m3")
        normalize: Whether to L2-normalize embeddings (default: True)
        batch_size: Batch size for encoding (default: 32)
    """

    class Params(ComponentParams):
        model_name: str = "BAAI/bge-m3"
        normalize: bool = True
        batch_size: int = 32
        # Asymmetric retrieval prompts. Explicit values win; left None they are
        # auto-resolved from the model family when ``auto_prompt`` is set (e5
        # gets "query: "/"passage: ", bge-v1.5 a query instruction); a model
        # with no known default (e.g. bge-m3) gets no prompt either way.
        query_prompt: str | None = None
        document_prompt: str | None = None
        auto_prompt: bool = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._normalize = self.p.normalize
        self._batch_size = self.p.batch_size
        self._query_prompt, self._document_prompt = resolve_hf_prompts(
            self.p.model_name,
            self.p.query_prompt,
            self.p.document_prompt,
            self.p.auto_prompt,
        )
        if self._query_prompt is not None or self._document_prompt is not None:
            logger.info(
                "HuggingFaceEmbedder '%s' applying prompts (query=%r, document=%r)",
                self.p.model_name,
                self._query_prompt,
                self._document_prompt,
            )
        self._model = SentenceTransformer(self.p.model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [c.text_for_index for c in chunks]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=False,
            prompt=self._document_prompt,
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
            prompt=self._query_prompt,
        )
        return vec.tolist()


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

    class Params(ComponentParams):
        model_name: str = "gemini-embedding-001"
        batch_size: int = 100
        task_type_document: str = "RETRIEVAL_DOCUMENT"
        task_type_query: str = "RETRIEVAL_QUERY"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model_name = self.p.model_name
        self._batch_size = self.p.batch_size
        self._task_type_document = self.p.task_type_document
        self._task_type_query = self.p.task_type_query

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

        texts = [c.text_for_index for c in chunks]
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

    @retry_transient
    def _embed_with_retry(self, texts: list[str], task_type: str) -> Any:
        """Call the Google embedding API with automatic retry."""
        return self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config={"task_type": task_type},
        )


@registry.register("embedding", "openai")
class OpenAIEmbedder(BaseEmbedder):
    """Embeds text using OpenAI's embedding API.

    Uses the ``openai`` SDK directly.

    Config params:
        model_name: Embedding model (default: "text-embedding-ada-002").
        batch_size: Texts per API call (default: 100).
    """

    class Params(ComponentParams):
        model_name: str = "text-embedding-ada-002"
        batch_size: int = 100

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._model_name = self.p.model_name
        self._batch_size = self.p.batch_size

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in .env or your shell environment."
            )

        import openai

        self._client: Any = openai.OpenAI(api_key=api_key)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.text_for_index for c in chunks]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._embed_with_retry(batch)
            batch_embeddings = sorted(result.data, key=lambda e: e.index)
            all_embeddings.extend(e.embedding for e in batch_embeddings)

        return [
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, all_embeddings, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        result = self._embed_with_retry([query])
        return result.data[0].embedding

    @retry_transient
    def _embed_with_retry(self, texts: list[str]) -> Any:
        """Call the OpenAI embedding API with automatic retry."""
        return self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )


@registry.register("embedding", "edenai")
class EdenAIEmbedder(BaseEmbedder):
    """Embeds text using Eden AI's embedding gateway.

    Calls the Eden AI ``/v2/text/embeddings`` endpoint directly.
    Routes to a sub-provider (e.g. ``"openai"``) so a single
    Eden AI key can access multiple embedding backends.

    Config params:
        provider: Sub-provider name (e.g. "openai") — required.
        model_name: Model name forwarded to the sub-provider
            (default: "text-embedding-ada-002").
        batch_size: Texts per API call (default: 100).
    """

    _ENDPOINT = "https://api.edenai.run/v2/text/embeddings"

    class Params(ComponentParams):
        provider: str = ""
        model_name: str = "text-embedding-ada-002"
        batch_size: int = 100

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)
        self._provider = self.p.provider
        if not self._provider:
            raise ValueError(
                "EdenAIEmbedder requires 'provider' in config "
                "(e.g. 'openai'). "
                "Set it via embedding.params.provider in YAML."
            )
        self._model_name = self.p.model_name
        self._batch_size = self.p.batch_size

        self._api_key = os.environ.get("EDENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "EDENAI_API_KEY environment variable is required. "
                "Set it in .env or your shell environment."
            )

        self._session = requests.Session()
        self._session.headers.update(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            }
        )

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST to Eden AI embeddings and return a list of embedding vectors."""
        payload: dict[str, Any] = {
            "texts": texts,
            "providers": self._provider,
            "response_as_dict": False,
        }
        if self._model_name:
            payload["settings"] = {self._provider: self._model_name}

        resp = self._session.post(self._ENDPOINT, json=payload)
        if resp.status_code >= 500:
            raise RuntimeError(f"EdenAI server error {resp.status_code}")
        if resp.status_code >= 400:
            raise ValueError(f"EdenAI invalid payload: {resp.text}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"EdenAI unexpected status {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        # response_as_dict=False returns a list of provider result objects
        provider_result = data[0]
        if provider_result.get("status") == "fail":
            err = provider_result.get("error", {}).get("message", "unknown error")
            raise RuntimeError(f"EdenAI provider error: {err}")

        return [item["embedding"] for item in provider_result["items"]]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.text_for_index for c in chunks]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._embed_with_retry(batch)
            all_embeddings.extend(result)

        return [
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, all_embeddings, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_with_retry([query])[0]

    @retry_transient
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call Eden AI embedding API with retry."""
        return self._call_api(texts)
