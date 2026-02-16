from __future__ import annotations

from langchain_chroma import Chroma

from .config import Config
from .embeddings import PatchedEdenAiEmbeddings
from .ingest import chunk_documents, load_pdfs


def build_or_update_index(cfg: Config, *, force: bool = False) -> int:
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)
    cfg.kb_dir.mkdir(parents=True, exist_ok=True)

    embeddings = PatchedEdenAiEmbeddings(
        provider=cfg.provider,
        model=cfg.embedding_model,
        edenai_api_key=cfg.api_key,  # type: ignore[arg-type]
    )

    store = Chroma(
        collection_name=cfg.collection_name,
        persist_directory=str(cfg.persist_dir),
        embedding_function=embeddings,
    )

    if force:
        store.delete_collection()
        store = Chroma(
            collection_name=cfg.collection_name,
            persist_directory=str(cfg.persist_dir),
            embedding_function=embeddings,
        )

    docs = load_pdfs(cfg.kb_dir)
    chunks = chunk_documents(
        docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    store.add_documents(chunks)
    return len(chunks)
