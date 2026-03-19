"""Shared test fixtures for the uwf-rag-experiments test suite."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — bare imports (e.g. ``from core.types import ...``) work in tests
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

# Trigger component registration so registry is populated
import components  # noqa: F401, E402
from core.config import ExperimentConfig  # noqa: E402
from core.types import (  # noqa: E402
    Chunk,
    Document,
    EmbeddedChunk,
    RetrievedChunk,
    ScoredSample,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the ``tests/fixtures/`` directory."""
    return FIXTURES


# ---------------------------------------------------------------------------
# Core data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            content="Grade forgiveness allows students to retake a course.",
            metadata={"source_path": "handbook.pdf", "page_number": 1},
        ),
        Document(
            content="The university police department is open 24 hours.",
            metadata={"source_path": "handbook.pdf", "page_number": 2},
        ),
        Document(
            content="Financial aid applications are due by March 1.",
            metadata={"source_path": "handbook.pdf", "page_number": 3},
        ),
    ]


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            content="Grade forgiveness allows students to retake a course.",
            chunk_id="chunk_001",
            metadata={"source_name": "handbook", "page_number": 1},
        ),
        Chunk(
            content="The policy applies to undergraduate students only.",
            chunk_id="chunk_002",
            metadata={"source_name": "handbook", "page_number": 1},
        ),
        Chunk(
            content="The university police department is open 24 hours.",
            chunk_id="chunk_003",
            metadata={"source_name": "handbook", "page_number": 2},
        ),
        Chunk(
            content="Financial aid applications are due by March 1.",
            chunk_id="chunk_004",
            metadata={"source_name": "handbook", "page_number": 3},
        ),
        Chunk(
            content="Students must maintain a 2.0 GPA for good standing.",
            chunk_id="chunk_005",
            metadata={"source_name": "handbook", "page_number": 4},
        ),
    ]


@pytest.fixture()
def sample_retrieved_chunks(sample_chunks: list[Chunk]) -> list[RetrievedChunk]:
    scores = [0.95, 0.88, 0.82, 0.75, 0.60]
    return [
        RetrievedChunk(chunk=c, score=s, retrieval_method="dense")
        for c, s in zip(sample_chunks, scores, strict=True)
    ]


@pytest.fixture()
def sample_embedded_chunks(sample_chunks: list[Chunk]) -> list[EmbeddedChunk]:
    rng = np.random.default_rng(42)
    return [
        EmbeddedChunk(chunk=c, embedding=rng.standard_normal(8).tolist())
        for c in sample_chunks
    ]


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config_dict(fixtures_dir: Path) -> dict:
    """Config dict matching the test fixture YAML structure."""
    return {
        "name": "test_baseline",
        "description": "Test configuration",
        "pipeline_mode": "linear",
        "indexing": {
            "sources": [
                {
                    "name": "test_source",
                    "path": str(fixtures_dir / "sample.pdf"),
                    "ingest": {"type": "pdf"},
                }
            ],
            "chunking": {
                "type": "recursive_langchain",
                "params": {"chunk_size": 256, "chunk_overlap": 25},
            },
            "embedding": {
                "type": "huggingface",
                "params": {"model_name": "BAAI/bge-m3"},
            },
            "vectorstore": {"type": "faiss", "params": {"metric": "cosine"}},
        },
        "query": {
            "retrieval": {"type": "dense", "top_k_retrieve": 5, "top_k_final": 3},
            "reranking": {"type": "none"},
            "generation": {"type": "ollama"},
            "generation_llm": {"model_name": "test-model", "temperature": 0.0},
            "prompt": {"type": "chat"},
        },
        "evaluation": {
            "dataset": "test_dataset.jsonl",
            "mode": "full",
            "num_runs": 1,
        },
    }


@pytest.fixture()
def minimal_experiment_config(minimal_config_dict: dict) -> ExperimentConfig:
    return ExperimentConfig.from_dict(minimal_config_dict)


# ---------------------------------------------------------------------------
# Evaluation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_scored_samples() -> list[ScoredSample]:
    return [
        ScoredSample(
            id="1",
            query="What is grade forgiveness?",
            response="Grade forgiveness allows retaking a course.",
            retrieved_contexts=["Grade forgiveness allows students..."],
            reference="Grade forgiveness is a policy...",
            scores={"faithfulness": 0.9, "answer_correctness": 0.85},
        ),
        ScoredSample(
            id="2",
            query="When is financial aid due?",
            response="Financial aid is due March 1.",
            retrieved_contexts=["Financial aid applications..."],
            reference="Applications are due by March 1.",
            scores={"faithfulness": 1.0, "answer_correctness": 0.92},
        ),
        ScoredSample(
            id="3",
            query="Is the police department open?",
            response="Yes, it is open 24 hours.",
            retrieved_contexts=["The university police department..."],
            reference="The police department is open 24 hours.",
            scores={"faithfulness": 1.0, "answer_correctness": 0.95},
        ),
    ]
