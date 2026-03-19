# UWF RAG Experiments

Reproducible Retrieval-Augmented Generation (RAG) experiments for UWF advising data.

This is an experiment harness, not a production chatbot. It lets you change one RAG factor at a time (chunking, retrieval, reranking, generation, prompts, models), run the pipeline, evaluate with RAGAS, and compare results side-by-side.

## Project Structure

```text
src/
  core/          config loading, component registry, shared types
  components/    chunkers, embedders, generators, ingestors, prompts,
                 retrievers, vectorstores, defaults
  pipeline/      indexing, query, rag orchestration, agent (stub)
  evaluation/    RAGAS evaluator, experiment comparison
configs/
  base.yaml                     default baseline configuration
  experiments/                  per-experiment overrides (inherit from base)
scripts/
  run_experiment.py             CLI: run a full experiment
  compare.py                    CLI: compare experiment results
tests/                          184 unit tests
results/                        experiment outputs (gitignored contents)
data/
  sources/                      source PDFs (gitignored)
  datasets/                     evaluation JSONL datasets (gitignored)
  indexes/                      cached FAISS indexes (gitignored)
```

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/) (for local LLM) **or** an `EDENAI_API_KEY` for cloud LLMs

## Setup

```bash
make bootstrap
cp .env.example .env
# Edit .env with your API key(s)
```

## Workflow

### 1. Define an experiment

Experiments are YAML configs that inherit from `configs/base.yaml` and override only the variables being tested:

```yaml
# configs/experiments/edenai_baseline.yaml
extends: ../base.yaml

name: "edenai_baseline"
description: "Baseline with Eden AI (OpenAI GPT-4)"

query:
  generation:
    type: "edenai"
    params:
      sub_provider: "openai"
  generation_llm:
    provider: "edenai"
    model_name: "gpt-4"
    temperature: 0.0
```

### 2. Run an experiment

```bash
python scripts/run_experiment.py configs/base.yaml -v
# or
make experiment CONFIG=configs/experiments/edenai_baseline.yaml
```

This builds the index (or loads from cache), runs the pipeline against the evaluation dataset, computes RAGAS metrics across multiple runs, and saves results.

### 3. Compare results

```bash
python scripts/compare.py results/baseline results/edenai_baseline
```

Additional flags:

```bash
--diff                  # Show config differences between experiments
--per-sample            # Show per-question score comparison
--sort faithfulness     # Sort by a specific metric
```

## Configuration

The base config (`configs/base.yaml`) defines all default values. Experiment configs use `extends:` to inherit and override:

| Section | Controls |
|---|---|
| `indexing.sources` | PDF paths and ingestor type |
| `indexing.chunking` | Chunk size, overlap, separators |
| `indexing.embedding` | Embedding model (e.g., `BAAI/bge-m3`) |
| `indexing.vectorstore` | Vector store type and metric |
| `query.retrieval` | Retriever type, `top_k_retrieve`, `top_k_final` |
| `query.reranking` | Reranker type (or `none`) |
| `query.generation` | Generator backend (`ollama`, `edenai`) |
| `query.generation_llm` | Model name, temperature, max tokens |
| `query.prompt` | System template, context format, citation style |
| `evaluation` | Dataset path, metrics list, number of runs |

## Output Format

Each experiment produces:

```text
results/<experiment_name>/
  summary.json    aggregated metrics + config snapshot + git metadata
  config.yaml     full resolved configuration for reproducibility
  run_1.jsonl     per-sample scores for run 1
  run_2.jsonl     per-sample scores for run 2
  ...
```

## Extending the System

Components are registered via decorators and resolved from YAML config at runtime:

```python
from components.base import BaseChunker
from core.registry import registry

@registry.register("chunking", "my_chunker")
class MyChunker(BaseChunker):
    def chunk(self, documents):
        ...
```

Then activate it in config:

```yaml
indexing:
  chunking:
    type: "my_chunker"
    params:
      custom_param: 42
```

Available component categories: `ingest`, `chunking`, `embedding`, `vectorstore`, `retrieval`, `reranking`, `generation`, `prompts`, `query_transform`, `memory`.

## Code Quality

```bash
make qa          # Full gate: format check + mypy + ruff + tests
make test        # Fast tests only (skip @pytest.mark.slow)
make test-all    # All tests including slow
make fmt         # Auto-fix formatting
make lint        # Ruff lint check
make typecheck   # Mypy type check
```

## Troubleshooting

- **Missing API key**: Ensure `.env` exists and contains `EDENAI_API_KEY=...`
- **Ollama not running**: Start Ollama with `ollama serve` before running experiments with `generation.type: "ollama"`
- **Index cache stale**: Use `--no-cache` flag to force rebuild: `python scripts/run_experiment.py configs/base.yaml --no-cache`
- **Unknown component type**: The selected type isn't registered. Check spelling matches a `@registry.register(...)` decorator.
