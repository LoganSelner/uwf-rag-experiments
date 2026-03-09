# UWF RAG Experiments

Reproducible Retrieval-Augmented Generation (RAG) experiments for UWF advising data.

This repository is an experiment harness, not a production chatbot service. It is designed to let you change one RAG factor at a time (chunking, retrieval, reranking, generation, prompts, models, datasets), run the pipeline, evaluate with RAGAS, and compare runs side-by-side.

This repository is intended for private internal collaboration by authorized teammates.

## What This README Covers

1. What is implemented right now
2. How to run the full workflow
3. How configuration works
4. What artifacts are produced
5. How to extend the architecture safely

## Current Implementation Status

Implemented in runtime code today:

- `chunker_type`: `recursive`
- `retriever_type`: `dense`, `mmr`
- `reranker_type`: `none`, `cross_encoder`
- `generator_type`: `stuff`

Configured as planned but not yet registered in runtime:

- `retriever_type`: `hybrid`
- `generator_type`: `map_reduce`, `refine`

If a non-implemented type is selected in `configs/eval.yaml`, the pipeline raises a clear `ValueError` listing available options.

## Project Structure

```text
.
├── configs/
│   ├── models.yaml
│   ├── index.yaml
│   ├── eval.yaml
│   └── pipelines/
├── data/
│   ├── raw/           # source PDFs
│   └── queries/       # eval CSVs
├── indexes/           # local Chroma persistence
├── runs/              # per-run artifacts
├── src/rag_testing/
│   ├── components/
│   ├── ingest.py
│   ├── pipeline.py
│   ├── run_pipeline.py
│   ├── evaluate_ragas.py
│   └── compare_runs.py
├── tests/
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Prerequisites

- Python `3.11`
- [`uv`](https://github.com/astral-sh/uv)
- API key(s) matching your configured backend in `configs/models.yaml`

`.env.example`:

```bash
EDENAI_API_KEY=your_edenai_key_here
OPENAI_API_KEY=your_openai_key_here
```

Copy and edit:

```bash
cp .env.example .env
```

## Setup

```bash
make bootstrap
```

This installs Python/dependencies and sets up pre-commit hooks.

## Workflow

### 1. Build or update the vector index

Default:

```bash
make index
```

Force rebuild:

```bash
uv run rag-test-index --force
```

### 2. Run the active pipeline over eval questions

```bash
make run
```

### 3. Evaluate a run with RAGAS

Evaluate latest run for the active pipeline:

```bash
make eval
```

Evaluate a specific run directory:

```bash
uv run rag-test-eval --run-dir runs/<timestamp>_<pipeline_name>
```

### 4. Compare scored runs

```bash
make compare
```

With filters/sorting:

```bash
uv run rag-test-compare --last 10 --sort-by faithfulness
```

## Make vs Direct CLI

Project convention:

- Use `make` targets for default/common paths (`index`, `run`, `eval`, `compare`)
- Use direct CLI when passing flags (`--force`, `--run-dir`, `--last`, `--sort-by`)

## Configuration Model

### `configs/models.yaml`

Defines:

- LLM backend/provider/model/temperature/max tokens
- Embeddings backend/provider/model
- which environment variable contains the API key for each

### `configs/index.yaml`

Defines:

- PDF source directory (`source_dir`)
- Chroma persistence directory (`persist_dir`)
- collection name
- chunking parameters (`chunk_size`, `chunk_overlap`, `chunker_type`)

### `configs/eval.yaml`

Defines:

- evaluation dataset path (`qa_path`)
- runs output directory (`runs_dir`)
- active pipeline component types
- retrieval sizing (`retrieval_k`, `top_k`)
- prompt override (`prompt_template`)
- metric list for RAGAS

### `configs/pipelines/*.yaml`

Reusable presets. Copy one into `configs/eval.yaml` to activate it.

## Output Artifacts

Each run creates a timestamped folder:

```text
runs/<timestamp>_<retriever>_<reranker>_<generator>/
```

Produced files:

- `predictions.jsonl`: one row per question with answer + retrieved contexts
- `config_used.yaml`: full resolved settings snapshot + `git_sha`
- `scores.jsonl`: per-sample RAGAS metric outputs (after evaluation)
- `metrics.json`: aggregate mean metrics (after evaluation)

## Quality and Checks

Run full quality gate:

```bash
make qa
```

Equivalent steps:

- `make fmt-check`
- `make typecheck`
- `make lint`
- `make test`

Also available:

- `make precommit` (runs all pre-commit hooks on all tracked files)

## Housekeeping

- `make clean`: remove caches
- `make deep-clean`: remove caches + env/build/coverage artifacts
- `make purge-artifacts CONFIRM=1`: delete generated experiment artifacts (`runs/`, `indexes/`, etc.)

`purge-artifacts` is intentionally guarded and does nothing unless `CONFIRM=1` is provided.

## Extending the System

Component interfaces are protocol-based and intentionally swappable:

- `Chunker`
- `Retriever`
- `Reranker`
- `Generator`

Extension pattern:

1. Add a component class under `src/rag_testing/components/` with a `from_settings(...)` constructor
2. Register it in the corresponding registry in `ingest.py` or `pipeline.py`
3. Add/update tests in `tests/test_components.py` (and related integration tests)
4. Activate it from config

## Testing Status

The project includes unit tests for config loading, components, ingest flow, run artifacts, evaluation flow, and run comparison.

Run tests:

```bash
make test
```

## Troubleshooting

- Missing API key errors:
  - Ensure `.env` exists and key names match `api_key_env` values in `configs/models.yaml`
- `rag-test-index` skips indexing:
  - Collection already has vectors. Use `uv run rag-test-index --force` to rebuild.
- `rag-test-eval` cannot find a run:
  - Run `make run` first, or pass `--run-dir` explicitly.
- Unknown component type error:
  - Selected type is not implemented/registered yet.
