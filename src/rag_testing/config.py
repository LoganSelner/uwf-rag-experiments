"""Configuration loading and typed settings objects for the experiment harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    provider: str
    model: str
    temperature: float
    api_key_env: str
    max_tokens: int = 1024


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    provider: str
    model: str
    api_key_env: str


@dataclass(frozen=True)
class ModelSettings:
    llm: LLMConfig
    embeddings: EmbeddingConfig


@dataclass(frozen=True)
class IndexSettings:
    source_dir: Path
    persist_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    chunker_type: str = "recursive"


@dataclass(frozen=True)
class EvalSettings:
    qa_path: Path
    runs_dir: Path
    retriever_type: str  # dense | hybrid | mmr
    reranker_type: str  # none  | cross_encoder
    generator_type: str  # stuff | (future: map_reduce, refine)
    prompt_template: str  # "" = use component default
    retrieval_k: int  # initial fetch count from vector store (before reranking)
    reranker_model: str  # "" = use component default
    hybrid_alpha: float  # 0.0 = BM25 only, 1.0 = dense only
    mmr_lambda: float  # 0.0 = max diversity, 1.0 = max relevance
    metrics: list[str]
    top_k: int  # final context count passed to LLM (after reranking)
    ragas_timeout: int  # per-metric timeout in seconds for ragas evaluate()


@dataclass(frozen=True)
class Settings:
    models: ModelSettings
    index: IndexSettings
    eval: EvalSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and ensure the top-level value is a mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def load_settings(
    *,
    models_path: Path = Path("configs/models.yaml"),
    index_path: Path = Path("configs/index.yaml"),
    eval_path: Path = Path("configs/eval.yaml"),  # or any configs/pipelines/*.yaml
) -> Settings:
    """Load model, index, and eval settings from YAML files.

    Environment variables from ``.env`` are loaded before parsing so provider
    configuration can reference API key variable names safely.
    """
    load_dotenv()
    models_raw = _load_yaml(models_path)
    index_raw = _load_yaml(index_path)
    eval_raw = _load_yaml(eval_path)

    llm_raw = models_raw["llm"]
    emb_raw = models_raw["embeddings"]

    models = ModelSettings(
        llm=LLMConfig(
            backend=str(llm_raw["backend"]),
            provider=str(llm_raw["provider"]),
            model=str(llm_raw["model"]),
            temperature=float(llm_raw.get("temperature", 0.0)),
            api_key_env=str(llm_raw["api_key_env"]),
            max_tokens=int(llm_raw.get("max_tokens", 1024)),
        ),
        embeddings=EmbeddingConfig(
            backend=str(emb_raw["backend"]),
            provider=str(emb_raw["provider"]),
            model=str(emb_raw["model"]),
            api_key_env=str(emb_raw["api_key_env"]),
        ),
    )

    index = IndexSettings(
        source_dir=Path(index_raw["source_dir"]),
        persist_dir=Path(index_raw["persist_dir"]),
        collection_name=str(index_raw["collection_name"]),
        chunk_size=int(index_raw["chunk_size"]),
        chunk_overlap=int(index_raw["chunk_overlap"]),
        chunker_type=str(index_raw.get("chunker_type", "recursive")),
    )

    eval_settings = EvalSettings(
        qa_path=Path(eval_raw["qa_path"]),
        runs_dir=Path(eval_raw["runs_dir"]),
        retriever_type=str(eval_raw.get("retriever_type", "dense")),
        reranker_type=str(eval_raw.get("reranker_type", "none")),
        generator_type=str(eval_raw.get("generator_type", "stuff")),
        prompt_template=str(eval_raw.get("prompt_template", "")),
        retrieval_k=int(eval_raw.get("retrieval_k", 20)),
        reranker_model=str(eval_raw.get("reranker_model", "")),
        hybrid_alpha=float(eval_raw.get("hybrid_alpha", 0.5)),
        mmr_lambda=float(eval_raw.get("mmr_lambda", 0.5)),
        metrics=[str(m) for m in eval_raw.get("metrics", [])],
        top_k=int(eval_raw.get("top_k", 3)),
        ragas_timeout=int(eval_raw.get("ragas_timeout", 300)),
    )

    return Settings(models=models, index=index, eval=eval_settings)
