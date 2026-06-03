"""Tests for src/components/embedders.py — HuggingFace and Google embedders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uwf_rag.components.embedders import (
    EdenAIEmbedder,
    GoogleEmbedder,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    resolve_hf_prompts,
)
from uwf_rag.core.types import Chunk, EmbeddedChunk


def _make_chunks(n: int = 3) -> list[Chunk]:
    """Create N simple chunks for testing."""
    return [
        Chunk(
            content=f"Text {i}",
            chunk_id=f"id_{i}",
            metadata={"source_name": "test"},
        )
        for i in range(n)
    ]


@patch("uwf_rag.components.embedders.SentenceTransformer")
class TestHuggingFaceEmbedder:
    def test_embed_query_returns_list_of_floats(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"model_name": "test-model"})
        result = embedder.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    def test_embed_query_passes_normalize_flag(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"normalize": False})
        embedder.embed_query("test")

        mock_model.encode.assert_called_once_with(
            "test", normalize_embeddings=False, show_progress_bar=False, prompt=None
        )

    def test_embed_chunks_returns_embedded_chunks(self, mock_st_cls: MagicMock) -> None:
        chunks = _make_chunks(2)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].chunk is chunks[0]
        assert result[1].chunk is chunks[1]
        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]

    def test_embed_chunks_passes_batch_size(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0]])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"batch_size": 16})
        embedder.embed_chunks(_make_chunks(1))

        _, kwargs = mock_model.encode.call_args
        assert kwargs["batch_size"] == 16

    def test_embed_chunks_empty_list(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([]).reshape(0, 8)
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        result = embedder.embed_chunks([])

        assert result == []

    def test_default_model_name(self, mock_st_cls: MagicMock) -> None:
        mock_st_cls.return_value = MagicMock()
        HuggingFaceEmbedder()
        mock_st_cls.assert_called_once_with("BAAI/bge-m3")

    def test_custom_model_name(self, mock_st_cls: MagicMock) -> None:
        mock_st_cls.return_value = MagicMock()
        HuggingFaceEmbedder({"model_name": "custom/model"})
        mock_st_cls.assert_called_once_with("custom/model")

    def test_default_config_values(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder()
        embedder.embed_query("test")

        mock_model.encode.assert_called_once_with(
            "test", normalize_embeddings=True, show_progress_bar=False, prompt=None
        )

    def test_e5_model_applies_prompts(self, mock_st_cls: MagicMock) -> None:
        """An e5 model auto-applies query:/passage: prompts to encode()."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder({"model_name": "intfloat/e5-base-v2"})
        embedder.embed_query("test")
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] == "query: "

        mock_model.encode.return_value = np.array([[1.0]])
        embedder.embed_chunks(_make_chunks(1))
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] == "passage: "

    def test_auto_prompt_off_sends_no_prompt(self, mock_st_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0])
        mock_st_cls.return_value = mock_model

        embedder = HuggingFaceEmbedder(
            {"model_name": "intfloat/e5-base-v2", "auto_prompt": False}
        )
        embedder.embed_query("test")
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] is None


class TestResolveHfPrompts:
    def test_unknown_model_no_prompt(self) -> None:
        assert resolve_hf_prompts("BAAI/bge-m3", None, None, True) == (None, None)

    def test_e5_family_auto(self) -> None:
        assert resolve_hf_prompts("intfloat/e5-base-v2", None, None, True) == (
            "query: ",
            "passage: ",
        )

    def test_multilingual_e5_family_auto(self) -> None:
        assert resolve_hf_prompts(
            "intfloat/multilingual-e5-large", None, None, True
        ) == ("query: ", "passage: ")

    def test_bge_v15_query_only(self) -> None:
        q, d = resolve_hf_prompts("BAAI/bge-large-en-v1.5", None, None, True)
        assert q is not None and d is None

    def test_explicit_overrides_family(self) -> None:
        assert resolve_hf_prompts("intfloat/e5-base-v2", "Q: ", "D: ", True) == (
            "Q: ",
            "D: ",
        )

    def test_auto_off_disables_family(self) -> None:
        assert resolve_hf_prompts("intfloat/e5-base-v2", None, None, False) == (
            None,
            None,
        )


# -----------------------------------------------------------------------
# GoogleEmbedder
# -----------------------------------------------------------------------


def _mock_embedding(values: list[float]) -> MagicMock:
    """Create a mock embedding object with a .values attribute."""
    emb = MagicMock()
    emb.values = values
    return emb


def _mock_embed_result(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock result from client.models.embed_content()."""
    result = MagicMock()
    result.embeddings = [_mock_embedding(v) for v in embeddings]
    return result


class TestGoogleEmbedder:
    """Tests for GoogleEmbedder — all API calls are mocked."""

    def _make_embedder(self, **config_overrides: object) -> GoogleEmbedder:
        """Create a GoogleEmbedder with a mocked genai.Client."""
        config: dict = {**config_overrides}
        return GoogleEmbedder(config)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_query_returns_list_of_floats(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_result(
            [[0.1, 0.2, 0.3]]
        )

        embedder = self._make_embedder()
        result = embedder.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_query_uses_retrieval_query_task_type(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_result([[1.0]])

        embedder = self._make_embedder()
        embedder.embed_query("test")

        call_kwargs = mock_client.models.embed_content.call_args
        assert call_kwargs.kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_chunks_returns_embedded_chunks(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_result(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        embedder = self._make_embedder()
        chunks = _make_chunks(2)
        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].chunk is chunks[0]
        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_chunks_uses_retrieval_document_task_type(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_result([[1.0]])

        embedder = self._make_embedder()
        embedder.embed_chunks(_make_chunks(1))

        call_kwargs = mock_client.models.embed_content.call_args
        assert call_kwargs.kwargs["config"]["task_type"] == "RETRIEVAL_DOCUMENT"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_default_model_name(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        assert embedder._model_name == "gemini-embedding-001"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_custom_model_name(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder(model_name="text-embedding-004")
        assert embedder._model_name == "text-embedding-004"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_query_uses_configured_model_name(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_result([[1.0]])

        embedder = self._make_embedder(model_name="text-embedding-004")
        embedder.embed_query("test query")

        call_kwargs = mock_client.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-004"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False)
    @patch("google.genai.Client")
    def test_missing_api_key_raises(self, mock_client_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleEmbedder()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}, clear=False)
    @patch("google.genai.Client")
    def test_embed_chunks_empty_list(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        result = embedder.embed_chunks([])

        assert result == []


# -----------------------------------------------------------------------
# OpenAIEmbedder
# -----------------------------------------------------------------------


def _mock_openai_embed_result(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock result from client.embeddings.create()."""
    result = MagicMock()
    data = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.index = i
        item.embedding = emb
        data.append(item)
    result.data = data
    return result


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder — all API calls are mocked."""

    def _make_embedder(self, **config_overrides: object) -> OpenAIEmbedder:
        config: dict = {**config_overrides}
        return OpenAIEmbedder(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_query_returns_list_of_floats(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_embed_result(
            [[0.1, 0.2, 0.3]]
        )

        embedder = self._make_embedder()
        result = embedder.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_chunks_returns_embedded_chunks(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_embed_result(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        embedder = self._make_embedder()
        chunks = _make_chunks(2)
        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].chunk is chunks[0]
        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_chunks_empty_list(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        result = embedder.embed_chunks([])

        assert result == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_default_model_name(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        assert embedder._model_name == "text-embedding-ada-002"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_custom_model_name(self, mock_client_cls: MagicMock) -> None:
        embedder = self._make_embedder(model_name="text-embedding-3-small")
        assert embedder._model_name == "text-embedding-3-small"

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    @patch("openai.OpenAI")
    def test_missing_api_key_raises(self, mock_client_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbedder()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_batching(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        # Two batches: first returns 2 embeddings, second returns 1
        mock_client.embeddings.create.side_effect = [
            _mock_openai_embed_result([[0.1], [0.2]]),
            _mock_openai_embed_result([[0.3]]),
        ]

        embedder = self._make_embedder(batch_size=2)
        chunks = _make_chunks(3)
        result = embedder.embed_chunks(chunks)

        assert len(result) == 3
        assert mock_client.embeddings.create.call_count == 2
        assert result[0].embedding == [0.1]
        assert result[1].embedding == [0.2]
        assert result[2].embedding == [0.3]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_query_uses_configured_model(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_embed_result([[1.0]])

        embedder = self._make_embedder(model_name="text-embedding-3-small")
        embedder.embed_query("test")

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


# -----------------------------------------------------------------------
# EdenAIEmbedder
# -----------------------------------------------------------------------


def _eden_api_response(embeddings: list[list[float]], status: str = "success") -> dict:
    """Build a fake Eden AI list-format response item."""
    return {
        "status": status,
        "provider": "openai",
        "items": [{"embedding": emb} for emb in embeddings],
    }


class TestEdenAIEmbedder:
    """Tests for EdenAIEmbedder — all API calls are mocked."""

    def _make_embedder(self, **config_overrides: object) -> EdenAIEmbedder:
        config: dict = {"provider": "openai", **config_overrides}
        return EdenAIEmbedder(config)

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_embed_query_returns_list_of_floats(self, mock_sess_cls: MagicMock) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = [_eden_api_response([[0.1, 0.2, 0.3]])]
        mock_sess_cls.return_value.post.return_value = mock_resp

        embedder = self._make_embedder()
        result = embedder.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_embed_chunks_returns_embedded_chunks(
        self, mock_sess_cls: MagicMock
    ) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = [_eden_api_response([[0.1, 0.2], [0.3, 0.4]])]
        mock_sess_cls.return_value.post.return_value = mock_resp

        embedder = self._make_embedder()
        chunks = _make_chunks(2)
        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].chunk is chunks[0]
        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_embed_chunks_empty_list(self, mock_sess_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        result = embedder.embed_chunks([])

        assert result == []

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_default_model_name(self, mock_sess_cls: MagicMock) -> None:
        embedder = self._make_embedder()
        assert embedder._model_name == "text-embedding-ada-002"

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_custom_model_name(self, mock_sess_cls: MagicMock) -> None:
        embedder = self._make_embedder(model_name="custom-model")
        assert embedder._model_name == "custom-model"

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    def test_missing_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            EdenAIEmbedder()

    @patch.dict("os.environ", {"EDENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="EDENAI_API_KEY"):
            EdenAIEmbedder({"provider": "openai"})

    @patch.dict("os.environ", {"EDENAI_API_KEY": "fake-key"}, clear=False)
    @patch("requests.Session")
    def test_batching(self, mock_sess_cls: MagicMock) -> None:
        mock_session = mock_sess_cls.return_value
        mock_resp_1 = MagicMock(status_code=200)
        mock_resp_1.json.return_value = [_eden_api_response([[0.1], [0.2]])]
        mock_resp_2 = MagicMock(status_code=200)
        mock_resp_2.json.return_value = [_eden_api_response([[0.3]])]
        mock_session.post.side_effect = [mock_resp_1, mock_resp_2]

        embedder = self._make_embedder(batch_size=2)
        chunks = _make_chunks(3)
        result = embedder.embed_chunks(chunks)

        assert len(result) == 3
        assert mock_session.post.call_count == 2
