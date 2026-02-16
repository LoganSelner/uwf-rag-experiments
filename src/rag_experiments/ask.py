from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.chat_models.edenai import ChatEdenAI

from .config import Config
from .embeddings import PatchedEdenAiEmbeddings

PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say so concisely.


CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


@dataclass(frozen=True)
class Answer:
    text: str
    contexts: list[str]


def ask(cfg: Config, question: str) -> Answer:
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

    docs = store.similarity_search(question, k=cfg.top_k)
    contexts = [d.page_content for d in docs]
    stuffed = "\n\n---\n\n".join(contexts)

    llm = ChatEdenAI(
        provider=cfg.provider,
        model=cfg.llm_model,
        temperature=0,
        edenai_api_key=cfg.api_key,  # type: ignore[arg-type]
    )
    msg = PROMPT.format(context=stuffed, question=question)
    resp = llm.invoke(msg)
    return Answer(text=str(resp.content), contexts=contexts)
