from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(kb_dir: Path) -> list[Document]:
    pdfs = sorted(kb_dir.glob("*.pdf"))
    docs: list[Document] = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
