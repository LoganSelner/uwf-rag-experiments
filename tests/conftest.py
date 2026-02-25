from __future__ import annotations

from pathlib import Path

import pytest

from rag_testing.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A fully-configured Settings object backed by files in tmp_path.

    source_dir and runs_dir are created as real directories.
    qa_path is NOT created — tests that need it must write the CSV themselves.
    """
    source_dir = tmp_path / "raw"
    persist_dir = tmp_path / "chroma"
    runs_dir = tmp_path / "runs"
    qa_path = tmp_path / "qa.csv"

    source_dir.mkdir()
    runs_dir.mkdir()

    (tmp_path / "models.yaml").write_text(
        """\
llm:
  backend: edenai
  provider: openai
  model: gpt-4
  temperature: 0.0
  api_key_env: EDENAI_API_KEY
embeddings:
  backend: edenai
  provider: openai
  model: text-embedding-3-small
  api_key_env: EDENAI_API_KEY
""",
        encoding="utf-8",
    )

    (tmp_path / "index.yaml").write_text(
        f"""\
source_dir: {source_dir}
persist_dir: {persist_dir}
collection_name: test-kb
chunk_size: 500
chunk_overlap: 50
chunker_type: recursive
""",
        encoding="utf-8",
    )

    (tmp_path / "eval.yaml").write_text(
        f"""\
qa_path: {qa_path}
runs_dir: {runs_dir}
retriever_type: dense
reranker_type: none
generator_type: stuff
prompt_template: ""
retrieval_k: 20
reranker_model: ""
hybrid_alpha: 0.5
mmr_lambda: 0.5
top_k: 3
metrics:
  - faithfulness
  - answer_relevancy
""",
        encoding="utf-8",
    )

    return load_settings(
        models_path=tmp_path / "models.yaml",
        index_path=tmp_path / "index.yaml",
        eval_path=tmp_path / "eval.yaml",
    )
