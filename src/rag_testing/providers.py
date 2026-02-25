"""Provider adapters for LLM and embeddings.

This keeps EdenAI and direct OpenAI wiring in one place so method code stays clean.
Callers use build_embeddings(cfg) and build_llm(cfg) and receive a LangChain-compatible
object without knowing which backend is in use.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_community.chat_models.edenai import ChatEdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain_community.utilities.requests import Requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import EmbeddingConfig, LLMConfig

# Conservative batch size to stay well under the EdenAI token limit.
_BATCH_SIZE = 500


class PatchedEdenAiEmbeddings(EdenAiEmbeddings):
    """Fixes EdenAI composite provider keys in embedding responses.

    LangChain's EdenAiEmbeddings expects the API to return results keyed by the
    plain provider name (e.g. "openai"), but the actual EdenAI API returns
    composite keys (e.g. "openai/1536__text-embedding-ada-002"). This subclass
    patches the response parsing to match on key prefix instead.
    """

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = "https://api.edenai.run/v2/text/embeddings"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.edenai_api_key.get_secret_value()}",  # type: ignore[union-attr]
            "User-Agent": self.get_user_agent(),
        }
        payload: dict[str, Any] = {"texts": texts, "providers": self.provider}
        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        response = Requests(headers=headers).post(url=url, data=payload)

        if response.status_code >= 500:
            raise RuntimeError(f"EdenAI Server: Error {response.status_code}")
        if response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        if response.status_code != 200:
            raise RuntimeError(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        temp = response.json()
        provider_response = temp.get(self.provider)
        if provider_response is None:
            for key in temp:
                if key.startswith(self.provider):
                    provider_response = temp[key]
                    break

        if provider_response is None:
            raise KeyError(
                f"No provider key starting with '{self.provider}' in response: "
                f"{list(temp.keys())}"
            )
        if provider_response.get("status") == "fail":
            raise RuntimeError(provider_response.get("error", {}).get("message"))

        return [item["embedding"] for item in provider_response["items"]]


def _require_api_key(env_var: str) -> str:
    """Return a required API key from the environment or raise a clear error."""
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"Missing required API key env var: {env_var}")
    return api_key


def build_embeddings(cfg: EmbeddingConfig) -> Embeddings:
    """Build an embeddings client from provider configuration."""
    if cfg.backend == "edenai":
        return PatchedEdenAiEmbeddings(
            provider=cfg.provider,
            model=cfg.model,
            edenai_api_key=_require_api_key(cfg.api_key_env),  # type: ignore[arg-type]
        )
    if cfg.backend == "openai":
        return OpenAIEmbeddings(
            model=cfg.model,
            api_key=_require_api_key(cfg.api_key_env),  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported embeddings backend: {cfg.backend!r}")


def build_llm(cfg: LLMConfig) -> BaseChatModel:
    """Build a chat model client from provider configuration."""
    if cfg.backend == "edenai":
        return ChatEdenAI(
            provider=cfg.provider,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            edenai_api_key=_require_api_key(cfg.api_key_env),  # type: ignore[arg-type]
        )
    if cfg.backend == "openai":
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
            api_key=_require_api_key(cfg.api_key_env),  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported llm backend: {cfg.backend!r}")
