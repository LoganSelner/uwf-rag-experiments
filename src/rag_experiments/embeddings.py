"""Patched Eden AI embeddings.

The upstream ``EdenAiEmbeddings`` class assumes the JSON response is keyed by
the bare provider name (e.g. ``"openai"``), but the current Eden AI API returns
a composite key like ``"openai/1536__text-embedding-ada-002"``.  This subclass
overrides ``_generate_embeddings`` to find the correct key.

It also adds batching to stay within per-request token limits.
"""

from __future__ import annotations

from typing import Any

from langchain_community.embeddings.edenai import EdenAiEmbeddings

# Conservative batch size to stay well under the 300k token limit.
_BATCH_SIZE = 500


class PatchedEdenAiEmbeddings(EdenAiEmbeddings):
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

        from langchain_community.utilities.requests import Requests

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        temp = response.json()

        # Upstream expects temp[self.provider] but the API now returns a
        # composite key like "openai/1536__text-embedding-ada-002".
        provider_response = temp.get(self.provider)
        if provider_response is None:
            for key in temp:
                if key.startswith(self.provider):
                    provider_response = temp[key]
                    break

        if provider_response is None:
            raise KeyError(
                f"No provider key starting with '{self.provider}' "
                f"in response: {list(temp.keys())}"
            )

        if provider_response.get("status") == "fail":
            err_msg = provider_response.get("error", {}).get("message")
            raise Exception(err_msg)

        return [item["embedding"] for item in provider_response["items"]]
