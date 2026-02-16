from pathlib import Path

from rag_experiments.config import load_config


def test_load_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        """
kb_dir: "artifacts/kb"
persist_dir: "artifacts/chroma"
collection_name: "uwf-kb"
chunk_size: 1000
chunk_overlap: 100
top_k: 3
embedding_model: "text-embedding-3-small"
llm_model: "gpt-4o-mini"
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.top_k == 3
    assert cfg.collection_name == "uwf-kb"
