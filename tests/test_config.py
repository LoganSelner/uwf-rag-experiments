from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rag_testing.config import Settings, load_settings

# Shared YAML content for tests that verify defaults and error cases.
_MINIMAL_MODELS_YAML = """\
llm:
  backend: edenai
  provider: openai
  model: gpt-4
  api_key_env: EDENAI_API_KEY
embeddings:
  backend: edenai
  provider: openai
  model: text-embedding-3-small
  api_key_env: EDENAI_API_KEY
"""

_MINIMAL_INDEX_YAML = """\
source_dir: data/raw
persist_dir: indexes/chroma
collection_name: kb
chunk_size: 1000
chunk_overlap: 100
"""


def test_path_fields_are_path_objects(settings: Settings) -> None:
    assert isinstance(settings.index.source_dir, Path)
    assert isinstance(settings.index.persist_dir, Path)
    assert isinstance(settings.eval.qa_path, Path)
    assert isinstance(settings.eval.runs_dir, Path)


def test_values_loaded_correctly(settings: Settings) -> None:
    assert settings.models.llm.model == "gpt-4"
    assert settings.models.llm.backend == "edenai"
    assert settings.models.llm.temperature == 0.0
    assert settings.models.embeddings.model == "text-embedding-3-small"
    assert settings.index.chunk_size == 500
    assert settings.index.chunk_overlap == 50
    assert settings.index.collection_name == "test-kb"
    assert settings.eval.top_k == 3
    assert settings.eval.retriever_type == "dense"
    assert settings.eval.reranker_type == "none"
    assert settings.eval.generator_type == "stuff"
    assert settings.eval.prompt_template == ""
    assert settings.eval.retrieval_k == 20
    assert settings.eval.reranker_model == ""
    assert settings.eval.hybrid_alpha == 0.5
    assert settings.eval.mmr_lambda == 0.5
    assert settings.index.chunker_type == "recursive"
    assert "faithfulness" in settings.eval.metrics
    assert "answer_relevancy" in settings.eval.metrics


def test_temperature_defaults_to_0_when_omitted(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        """\
llm:
  backend: edenai
  provider: openai
  model: gpt-4
  api_key_env: EDENAI_API_KEY
embeddings:
  backend: edenai
  provider: openai
  model: text-embedding-3-small
  api_key_env: EDENAI_API_KEY
""",
        encoding="utf-8",
    )  # no temperature field
    (tmp_path / "index.yaml").write_text(_MINIMAL_INDEX_YAML, encoding="utf-8")
    (tmp_path / "eval.yaml").write_text(
        """\
qa_path: data/queries/qa.csv
runs_dir: runs
retriever_type: dense
reranker_type: none
top_k: 3
metrics:
  - faithfulness
""",
        encoding="utf-8",
    )

    s = load_settings(
        models_path=tmp_path / "models.yaml",
        index_path=tmp_path / "index.yaml",
        eval_path=tmp_path / "eval.yaml",
    )
    assert s.models.llm.temperature == 0.0


def test_missing_required_key_raises(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        """\
llm:
  backend: edenai
  provider: openai
  api_key_env: EDENAI_API_KEY
embeddings:
  backend: edenai
  provider: openai
  model: text-embedding-3-small
  api_key_env: EDENAI_API_KEY
""",
        encoding="utf-8",
    )  # llm.model is missing
    (tmp_path / "index.yaml").write_text(_MINIMAL_INDEX_YAML, encoding="utf-8")
    (tmp_path / "eval.yaml").write_text(
        """\
qa_path: data/queries/qa.csv
runs_dir: runs
retriever_type: dense
reranker_type: none
top_k: 3
metrics: []
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_settings(
            models_path=tmp_path / "models.yaml",
            index_path=tmp_path / "index.yaml",
            eval_path=tmp_path / "eval.yaml",
        )


def test_new_eval_fields_default_correctly(tmp_path: Path) -> None:
    """New EvalSettings fields resolve to documented defaults when omitted."""
    (tmp_path / "models.yaml").write_text(_MINIMAL_MODELS_YAML, encoding="utf-8")
    (tmp_path / "index.yaml").write_text(_MINIMAL_INDEX_YAML, encoding="utf-8")
    (tmp_path / "eval.yaml").write_text(
        """\
qa_path: data/queries/qa.csv
runs_dir: runs
metrics:
  - faithfulness
""",
        encoding="utf-8",
    )  # all optional fields omitted — defaults should apply

    s = load_settings(
        models_path=tmp_path / "models.yaml",
        index_path=tmp_path / "index.yaml",
        eval_path=tmp_path / "eval.yaml",
    )
    assert s.eval.retriever_type == "dense"
    assert s.eval.reranker_type == "none"
    assert s.eval.generator_type == "stuff"
    assert s.eval.prompt_template == ""
    assert s.eval.retrieval_k == 20
    assert s.eval.reranker_model == ""
    assert s.eval.hybrid_alpha == 0.5
    assert s.eval.mmr_lambda == 0.5
    assert s.eval.top_k == 3


def test_settings_dataclasses_are_frozen(settings: Settings) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.eval.top_k = 99  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.index.chunk_size = 0  # type: ignore[misc]
