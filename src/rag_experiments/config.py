from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    kb_dir: Path
    persist_dir: Path
    collection_name: str

    chunk_size: int
    chunk_overlap: int
    top_k: int

    embedding_model: str
    llm_model: str

    provider: str
    api_key_env: str

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


def load_config(path: Path) -> Config:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(
        kb_dir=Path(raw["kb_dir"]),
        persist_dir=Path(raw["persist_dir"]),
        collection_name=str(raw["collection_name"]),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        top_k=int(raw["top_k"]),
        embedding_model=str(raw["embedding_model"]),
        llm_model=str(raw["llm_model"]),
        provider=str(raw.get("provider", "openai")),
        api_key_env=str(raw.get("api_key_env", "EDENAI_API_KEY")),
    )
