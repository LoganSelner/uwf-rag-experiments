from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from rag_testing import ingest
from rag_testing.components import RecursiveChunker
from rag_testing.config import Settings
from rag_testing.ingest import build_chunker, load_pdfs

# ---------------------------------------------------------------------------
# load_pdfs
# ---------------------------------------------------------------------------


def test_load_pdfs_empty_directory(tmp_path: Path) -> None:
    assert load_pdfs(tmp_path) == []


def test_load_pdfs_ignores_non_pdf_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "data.csv").write_text("a,b")
    assert load_pdfs(tmp_path) == []


# ---------------------------------------------------------------------------
# build_chunker
# ---------------------------------------------------------------------------


def test_build_chunker_recursive_returns_recursive_chunker(
    settings: Settings,
) -> None:
    chunker = build_chunker(settings)
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.chunk_size == settings.index.chunk_size
    assert chunker.chunk_overlap == settings.index.chunk_overlap


def test_build_chunker_unknown_type_raises(settings: Settings) -> None:
    bad = dataclasses.replace(
        settings,
        index=dataclasses.replace(settings.index, chunker_type="sentence"),
    )
    with pytest.raises(ValueError, match="sentence"):
        build_chunker(bad)


# ---------------------------------------------------------------------------
# build_or_update_index
# ---------------------------------------------------------------------------


def _mock_chroma(count: int = 0) -> MagicMock:
    store = MagicMock()
    store.get.return_value = {"ids": ["id"] * count}
    return store


def test_build_or_update_index_skips_when_collection_not_empty(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_store = _mock_chroma(count=10)
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: MagicMock())
    monkeypatch.setattr(ingest, "Chroma", MagicMock(return_value=mock_store))

    result = ingest.build_or_update_index(settings)

    assert result == 0
    mock_store.add_documents.assert_not_called()


def test_build_or_update_index_empty_collection_no_pdfs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_store = _mock_chroma(count=0)
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: MagicMock())
    monkeypatch.setattr(ingest, "Chroma", MagicMock(return_value=mock_store))

    result = ingest.build_or_update_index(settings)

    assert result == 0
    mock_store.add_documents.assert_not_called()


def test_build_or_update_index_force_deletes_collection(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_store = _mock_chroma(count=5)
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: MagicMock())
    monkeypatch.setattr(ingest, "Chroma", MagicMock(return_value=mock_store))

    ingest.build_or_update_index(settings, force=True)

    mock_store.delete_collection.assert_called_once()


def test_build_or_update_index_adds_chunks_when_present(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_store = _mock_chroma(count=0)
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: MagicMock())
    monkeypatch.setattr(ingest, "Chroma", MagicMock(return_value=mock_store))
    # Provide fake docs so chunking produces actual chunks
    fake_docs = [Document(page_content="word " * 300)]
    monkeypatch.setattr(ingest, "load_pdfs", lambda _path: fake_docs)

    result = ingest.build_or_update_index(settings)

    assert result > 0
    mock_store.add_documents.assert_called_once()
