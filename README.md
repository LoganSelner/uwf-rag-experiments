# UWF RAG Experiments

A config-driven experiment harness for measuring how different Retrieval-Augmented Generation (RAG) component choices affect evaluation metrics. Built for the [ARGObot](https://doi.org/10.1145/3696673.3723065) academic advising chatbot research at the University of West Florida.

This is a research tool, not a production chatbot. Change one variable at a time — chunking strategy, embedding model, vectorstore, retrieval depth, prompt template, LLM — run the pipeline, evaluate with [RAGAS](https://docs.ragas.io/), and compare results side-by-side.

## Research Context

ARGObot is an AI academic advising chatbot developed at the University of West Florida (UWF). The published paper ([ACMSE 2025](https://doi.org/10.1145/3696673.3723065)) compared a retrieval-based implementation (Gemini 1.0 Pro + ChromaDB) against an agent-based implementation (GPT-4 + ReAct loop). Both versions showed low Context Entity Recall (~0.27–0.29), motivating systematic experimentation with alternative RAG components.

This framework enables that experimentation by isolating every variable behind a YAML config parameter, caching indexes to avoid redundant embedding, and running RAGAS evaluation with statistical aggregation across multiple runs.

## Available Components

Every component is swappable via config. No code changes needed.

| Stage | Available Types | Config Key |
|-------|----------------|------------|
| **Ingestion** | PDF (PyMuPDF) | `indexing.sources[].ingest.type` |
| **Chunking** | `recursive_langchain`, `recursive_custom` | `indexing.chunking.type` |
| **Embedding** | `huggingface` (bge-m3 etc.), `google` (Gemini), `openai`, `edenai` (cloud gateway) | `indexing.embedding.type` |
| **Vectorstore** | `faiss` (cosine/L2), `chroma` (cosine/L2/IP) | `indexing.vectorstore.type` |
| **Retrieval** | `dense` (single-vector similarity) | `query.retrieval.type` |
| **Query Transform** | `passthrough`, `contextualizer` (LLM reformulation) | `query.query_transform.type` |
| **Reranking** | `none` (passthrough), `cross_encoder` (HF cross-encoder) | `query.reranking.type` |
| **Generation** | `ollama` (local), `edenai` (cloud gateway), `google` (Gemini), `openai` | `query.generation.type` |
| **Prompts** | `chat` (numbered/plain context, CoT, citation styles) | `query.prompt.type` |

## Prerequisites

- **Python 3.11** (exact — see `pyproject.toml`)
- **[uv](https://github.com/astral-sh/uv)** for dependency management
- At least one LLM backend:
  - **[Ollama](https://ollama.com/)** for local inference (free, requires GPU RAM)
  - **`EDENAI_API_KEY`** for cloud LLMs via [Eden AI](https://www.edenai.co/) (pay-per-use, routes to OpenAI/Google/etc.)
  - **`GOOGLE_API_KEY`** for direct Gemini access (free tier available)

## Setup

```bash
git clone https://github.com/LoganSelner/uwf-rag-experiments.git
cd uwf-rag-experiments
make bootstrap          # installs Python, syncs deps, sets up git hooks
cp .env.example .env    # then edit with your API key(s)
```

Place your source documents and evaluation datasets:

```
data/sources/knowledge_base.pdf          # or student_handbook.pdf for v1 replication
data/datasets/25qKB_dataset.jsonl        # see data/datasets/README.md for format
```

The default `configs/base.yaml` uses Eden AI for embeddings and generation, so `EDENAI_API_KEY` is sufficient to run experiments out of the box. For a local-only smoke test (no paid APIs), use `configs/smoke.yaml`, which routes everything through Ollama + HuggingFace — first pull the model it references:

```bash
ollama pull qwen3:14b
```

## Workflow

### 1. Define an experiment

Experiments are YAML configs that inherit from `configs/base.yaml` and override only the variables being tested. This ensures single-variable isolation — everything not explicitly overridden stays at the baseline value.

```yaml
# configs/experiments/reranker/cross_encoder.yaml
extends: ../../base.yaml

name: "reranker_cross_encoder"
description: "Baseline + cross-encoder reranking (gte-reranker-modernbert-base)"

query:
  reranking:
    type: "cross_encoder"
    params:
      model_name: "Alibaba-NLP/gte-reranker-modernbert-base"
```

### 2. Run an experiment

```bash
python scripts/run_experiment.py configs/base.yaml -v

# or with Make:
make experiment CONFIG=configs/experiments/reranker/cross_encoder.yaml
```

What happens:
1. Config is loaded and validated (typos caught before any model loads)
2. Index is built from source PDFs, or loaded from cache if the indexing config hasn't changed
3. Each evaluation question is run through the pipeline
4. RAGAS computes metrics (answer correctness, context precision, faithfulness, context entity recall, answer relevancy)
5. The run repeats `num_runs` times (default: 3) for statistical significance
6. Results are saved with full config snapshot and git SHA for reproducibility

### 3. Compare results

```bash
python scripts/compare.py results/baseline results/reranker_cross_encoder

# Additional options:
python scripts/compare.py results/* --sort-by faithfulness
python scripts/compare.py results/* --diff              # show config differences
python scripts/compare.py results/* --per-sample        # per-question breakdown
python scripts/compare.py results/* --format csv        # export for spreadsheets
python scripts/compare.py results/* --format json       # export for scripts
```

## Configuration Reference

The base config defines all default values. Experiment configs use `extends:` to inherit and override only what they're testing.

### Indexing (cached by fingerprint)

| Parameter | Config Path | Default |
|-----------|------------|---------|
| Source documents | `indexing.sources[]` | — |
| Chunking strategy | `indexing.chunking.type` | `recursive_langchain` |
| Chunk size | `indexing.chunking.params.chunk_size` | 1000 |
| Chunk overlap | `indexing.chunking.params.chunk_overlap` | 100 |
| Embedding model | `indexing.embedding.type` + `.params.model_name` | `edenai` / `text-embedding-3-small` |
| Vectorstore | `indexing.vectorstore.type` | `chroma` (cosine) |

Experiments that share the same indexing config reuse the cached index automatically.

### Query Pipeline

| Parameter | Config Path | Default |
|-----------|------------|---------|
| Query transform | `query.query_transform.type` | `contextualizer` (Eden AI GPT-4.1) |
| Retrieval depth | `query.retrieval.top_k_retrieve` | 3 |
| Final chunks | `query.retrieval.top_k_final` | 3 |
| Reranker | `query.reranking.type` | `none` |
| Generator | `query.generation.type` | `edenai` (OpenAI sub-provider) |
| LLM model | `query.generation_llm.model_name` | `gpt-4.1` |
| System prompt | `query.prompt.system_template` | QA-from-context prompt |
| Chain-of-thought | `query.prompt.use_chain_of_thought` | `false` |
| Citation style | `query.prompt.citation_style` | `none` |

### Evaluation

| Parameter | Config Path | Default |
|-----------|------------|---------|
| Dataset | `evaluation.dataset` | `data/datasets/25qKB_dataset.jsonl` |
| Mode | `evaluation.mode` | `full` (or `retrieval_only`, or `none`) |
| Number of runs | `evaluation.num_runs` | 3 |
| Evaluator LLM | `evaluation.evaluator_llm` | Eden AI GPT-4.1 |
| Evaluator embedding | `evaluation.evaluator_embedding` | Eden AI text-embedding-3-small |

The evaluator embedding is deliberately separate from the pipeline embedding. This ensures all experiments are measured with the same yardstick, even when comparing different pipeline embedding models.

## Included Experiment Configs

### Replication

Reproduces the ACMSE paper's Retrieval-based ARGObot. Requires `data/sources/student_handbook.pdf` and `data/datasets/19qHB_dataset.jsonl`.

| Config | Stack |
|--------|-------|
| `replication/v1.yaml` | Direct Google: Gemini 1.0 Pro + `embedding-001` + Chroma + contextualizer + verbatim-citation prompt |
| `replication/v1_edenai.yaml` | Same shape via Eden AI: `text-multilingual-embedding-002` + Gemini 2.0 Flash (Gemini 1.0 Pro isn't routable through Eden AI's Google path) |

### Reranking Sweep

Each changes only the reranking-related parameters from `base.yaml`. Compare against the no-reranker baseline run to isolate the reranker's effect.

| Config | top_k_retrieve → top_k_final |
|--------|------------------------------|
| `reranker/cross_encoder.yaml` | 3 → 3 (baseline pool, reranker reorders in place) |
| `reranker/cross_encoder_topk3.yaml` | 10 → 3 (wider pool, same final context) |
| `reranker/cross_encoder_topk5.yaml` | 10 → 5 (wider pool, slightly larger context) |
| `reranker/cross_encoder_topk10.yaml` | 20 → 10 (high-recall upper bound) |

All four use `Alibaba-NLP/gte-reranker-modernbert-base`.

### Smoke Test

`configs/smoke.yaml` (top-level, not under `experiments/`) routes the entire pipeline through local Ollama + HuggingFace with a 1-question dataset and sets `evaluation.mode: "none"` — the pipeline runs end-to-end and writes per-sample outputs to `run_1.jsonl`, but no RAGAS scoring (and no judge-LLM call) occurs. Use it to verify the pipeline without spending API credits or waiting on a local judge. Its results are not comparable to formal experiments.

## Output Format

Each experiment produces:

```
results/<experiment_name>/
  summary.json    aggregated metrics + config snapshot + git SHA
  config.yaml     full resolved config for exact reproducibility
  run_1.jsonl     per-sample scores for run 1
  run_2.jsonl     per-sample scores for run 2
  run_3.jsonl     per-sample scores for run 3
```

## Extending the System

Add a new component in four steps:

```python
# 1. Write the class in src/components/<category>.py
from components.base import BaseChunker
from core.registry import registry

@registry.register("chunking", "my_chunker")
class MyChunker(BaseChunker):
    def chunk(self, documents):
        ...
```

```python
# 2. Ensure the file is imported in src/components/__init__.py
from components import my_module  # noqa: F401
```

```python
# 3. Add an inventory assertion in tests/test_registry.py
def test_chunking_my_chunker(self) -> None:
    assert registry.is_registered("chunking", "my_chunker")
```

```yaml
# 4. Reference it in a config
indexing:
  chunking:
    type: "my_chunker"
    params:
      custom_param: 42
```

No pipeline code changes. No config system changes. No evaluation changes. Run `make qa` to verify.

For full architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md). For the development roadmap, see [ROADMAP.md](ROADMAP.md).

## Code Quality

```bash
make qa          # Full gate: format + typecheck + lint + tests
make test        # Fast tests only (skip @pytest.mark.slow)
make test-all    # All tests (312 tests)
make fmt         # Auto-fix formatting
make lint        # Ruff lint
make typecheck   # Mypy strict type checking
make precommit   # Run all pre-commit hooks
```

CI runs on every push and pull request (ruff + mypy + pytest).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `EDENAI_API_KEY` / `GOOGLE_API_KEY` not found | Ensure `.env` exists with your keys. The app loads it automatically at startup. |
| Ollama connection refused | Start Ollama with `ollama serve`. If using WSL, you may need to override `generation.params.base_url` in your experiment config. |
| Index cache stale after changing source PDFs | Use `--no-cache`: `python scripts/run_experiment.py configs/base.yaml --no-cache` |
| Unknown component type in validation | Check that the `type:` value in your YAML matches a `@registry.register(...)` name exactly. |
| RAGAS timeout on local Ollama | Increase `evaluation.run_config.timeout` (base default: 300s; `smoke.yaml` raises it to 1200s) or reduce `evaluation.run_config.max_workers`. |

## References

Tamascelli, M., Bunch, O., Fowler, B., Taeb, M., & Cohen, A. (2025). Academic Advising Chatbot Powered with AI Agent. In *2025 ACM Southeast Conference (ACMSE 2025)*. ACM. https://doi.org/10.1145/3696673.3723065
