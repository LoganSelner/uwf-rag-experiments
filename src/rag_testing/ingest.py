"""Ingestion pipeline: PDF loading, chunking, and Chroma index management."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .components import Chunker, RecursiveChunker
from .config import Settings, load_settings
from .providers import build_embeddings

_CHUNKERS: dict[str, Callable[[Settings], Chunker]] = {
    "recursive": RecursiveChunker.from_settings,
}


def load_pdfs(source_dir: Path) -> list[Document]:
    """Load all PDFs in ``source_dir`` and return concatenated page documents."""
    pdfs = sorted(source_dir.glob("*.pdf"))
    docs: list[Document] = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs


def build_chunker(settings: Settings) -> Chunker:
    """Create the configured chunker implementation from ``settings``."""
    key = settings.index.chunker_type
    if key not in _CHUNKERS:
        available = ", ".join(_CHUNKERS)
        raise ValueError(f"Unknown chunker_type: {key!r}. Available: {available}")
    return _CHUNKERS[key](settings)


def build_or_update_index(settings: Settings, *, force: bool = False) -> int:
    """Build the Chroma index if empty, or rebuild it when ``force`` is set.

    Returns the number of chunks added to the collection.
    """
    idx = settings.index
    idx.persist_dir.mkdir(parents=True, exist_ok=True)
    idx.source_dir.mkdir(parents=True, exist_ok=True)

    embeddings = build_embeddings(settings.models.embeddings)
    store = Chroma(
        collection_name=idx.collection_name,
        persist_directory=str(idx.persist_dir),
        embedding_function=embeddings,
    )

    if force:
        store.delete_collection()
        store = Chroma(
            collection_name=idx.collection_name,
            persist_directory=str(idx.persist_dir),
            embedding_function=embeddings,
        )
    else:
        existing_ids: list[str] = store.get()["ids"]
        if existing_ids:
            print(
                f"Collection '{idx.collection_name}' already contains "
                f"{len(existing_ids)} chunks. Skipping index build.\n"
                f"Run with --force to wipe and rebuild."
            )
            return 0

    docs = load_pdfs(idx.source_dir)
    chunker = build_chunker(settings)
    chunks = chunker.chunk(docs)
    if chunks:
        store.add_documents(chunks)
    return len(chunks)


def main() -> None:
    """CLI entrypoint for index build/update."""
    parser = argparse.ArgumentParser(
        description="Load PDFs, chunk them, and store embeddings in Chroma."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the existing collection and rebuild from scratch.",
    )
    args = parser.parse_args()

    settings = load_settings()
    n_chunks = build_or_update_index(settings, force=args.force)
    if n_chunks:
        print(f"Indexed {n_chunks} chunks into {settings.index.persist_dir}")


if __name__ == "__main__":
    main()
