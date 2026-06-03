# UWF RAG Experiments

A config-driven experiment harness for measuring how different Retrieval-Augmented Generation (RAG) component choices affect evaluation metrics.

This is a research tool, not a production chatbot. Change one variable at a time — chunking strategy, embedding model, vectorstore, retrieval method, reranker, query transform, prompt template, LLM — run the pipeline, evaluate with [RAGAS](https://docs.ragas.io/), and compare results side-by-side.

## What This Is

The harness exists to explore the modern RAG design space under controlled conditions. Every testable variable is a YAML parameter, so an experiment changes exactly one thing and the effect is attributable. Indexes are cached by fingerprint to avoid redundant embedding, and RAGAS evaluation runs multiple times with a fixed evaluator for statistically meaningful, cross-experiment-comparable results.

The goal is broad, faithful coverage of standard RAG practice — the dense + sparse + rerank retrieval baseline, standard query optimization, standard chunking, and agentic retrieval — built out incrementally so each component can be measured against a credible baseline. See [ROADMAP.md](ROADMAP.md) for the planned trajectory and [ARCHITECTURE.md](ARCHITECTURE.md) for the system as built.

## Available Components

Every component is swappable via config. No code changes needed.

| Stage | Available Types | Config Key |
|-------|----------------|------------|
| **Ingestion** | PDF (PyMuPDF) | `indexing.sources[].ingest.type` |
| **Chunking** | `recursive_langchain`, `recursive_custom`, `semantic` (embedding-breakpoint splits) | `indexing.chunking.type` |
| **Chunk Enricher** | `none`, `contextual` (LLM situating context, retrieval-only) | `indexing.chunk_enricher.type` |
| **Embedding** | `huggingface` (bge-m3 etc.; auto query/passage prompts for e5 / bge-v1.5 families), `google` (Gemini), `openai`, `edenai` (cloud gateway) | `indexing.embedding.type` |
| **Vectorstore** | `faiss` (cosine/L2), `chroma` (cosine/L2/IP) | `indexing.vectorstore.type` |
| **Sparse Index** | `bm25` (via [`bm25s`](https://github.com/xhluca/bm25s)) | `indexing.sparse_index.type` |
| **Retrieval** | `dense` (single-vector similarity), `bm25` (lexical), `hybrid` (dense + sparse fusion: RRF or weighted) | `query.retrieval.type` |
| **Query Transform** | `passthrough`, `contextualizer` (LLM reformulation), `hyde` (hypothetical document embeddings), `multi_query` (RAG-Fusion expansion) | `query.query_transform.type` |
| **Reranking** | `none` (passthrough), `cross_encoder` (HF cross-encoder) | `query.reranking.type` |
| **Generation** | `ollama` (local), `edenai` (cloud gateway), `google` (Gemini), `openai` — all support native tool calling | `query.generator.provider` |
| **Prompts** | `chat` (numbered/plain context, CoT, citation styles) | `query.prompt.type` |
| **Pipeline Mode** | `linear`, `agent` (single-agent ReAct, native tool calling) | `pipeline_mode` |
| **Agent Tools** | `rag` (knowledge-base search) | `agent.tools[].type` |
| **Agent Memory** | `none`, `buffer_window` | `agent.memory.type` |

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
data/sources/knowledge_base.pdf          # your source document(s)
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

To sweep a whole set at once — each config is a file, directory, or glob; a
failing config is skipped, not fatal; experiments sharing an index build it only
once — use the matrix runner:

```bash
# every config under configs/experiments/, then print a comparison table
python scripts/run_matrix.py --compare

# a subset (directory / glob), or via Make
python scripts/run_matrix.py configs/experiments/retrieval --compare
make matrix CONFIGS=configs/experiments/chunking
```

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
| Sparse index (optional) | `indexing.sparse_index.type` | unset (built only when configured) |
| Chunk enricher (optional) | `indexing.chunk_enricher.type` | unset (built only when configured) |

Experiments that share the same indexing config reuse the cached index automatically. When `sparse_index` is configured, the BM25 index is built before embeddings (so tokenizer/stemmer misconfigurations fail fast) and cached in a `bm25/` subdirectory next to the vector index. When `chunk_enricher` is configured, it runs between chunking and embedding and may set each chunk's `index_text` (the text that gets embedded + BM25-indexed) while leaving `content` — what's stored, generated from, and scored — untouched, so the enrichment's effect is isolated to retrieval. Both optional components join the index fingerprint only when set, so adding them never invalidates existing dense-only caches.

### Query Pipeline

| Parameter | Config Path | Default |
|-----------|------------|---------|
| Query transform | `query.query_transform.type` | `passthrough` (contextualizer is conversational — a no-op under single-turn eval) |
| Multi-query fusion | `query.query_transform.fusion` | `rrf` (or `max`) |
| Retrieval depth | `query.retrieval.top_k_retrieve` | 10 |
| Final chunks | `query.retrieval.top_k_final` | 5 |
| Reranker | `query.reranking.type` | `none` |
| Generator provider | `query.generator.provider` | `edenai` (sub-provider in `query.generator.params.sub_provider`) |
| LLM model | `query.generator.model_name` | `gpt-4.1` |
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

### Example: Reconstructing a Published Pipeline

These configs demonstrate the harness's flexibility by reconstructing a specific published RAG setup (a retrieval-based advising chatbot) entirely through config. They are kept as a worked example of reproducing an arbitrary pipeline, not as a core objective. Requires `data/sources/student_handbook.pdf` and `data/datasets/19qHB_dataset.jsonl`.

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

### Retrieval Matrix (dense / sparse / hybrid)

The canonical strong-baseline matrix — compare BM25 and hybrid fusion against the dense baseline (`base.yaml`) to see whether sparse retrieval and fusion add lift on this corpus.

| Config | Retrieval | Fusion | Reranker |
|--------|-----------|--------|----------|
| `retrieval/bm25.yaml` | BM25 only | — | none |
| `retrieval/hybrid_rrf.yaml` | dense + BM25 | RRF (k=60) | none |
| `retrieval/hybrid_rrf_rerank.yaml` | dense + BM25 | RRF (k=60) | `cross_encoder` |
| `retrieval/hybrid_weighted.yaml` | dense + BM25 | weighted (0.5/0.5, min-max) | none |

Per-branch `top_k=20` is the production-standard pre-fusion depth. Stage-level `top_k_retrieve` controls the count entering the reranker; `top_k_final` is what the generator sees.

### Query Optimization (HyDE / multi-query)

Standard pre-retrieval query transformations. Each changes only `query.query_transform` from the baseline. Compare against `base.yaml` (passthrough) to isolate the transform's effect.

| Config | Transform | Retrieval | Notes |
|--------|-----------|-----------|-------|
| `query_transform/hyde_dense.yaml` | HyDE (1 hypothetical) | dense | Paper-canonical: hypothetical answer replaces the query for embedding |
| `query_transform/hyde_hybrid.yaml` | HyDE + per-branch routing | hybrid | Hypothetical → dense branch, original question → BM25 branch |
| `query_transform/multi_query_dense.yaml` | multi-query (4 reformulations) | dense | RRF fusion across reformulations |
| `query_transform/multi_query_hybrid.yaml` | multi-query (4) | hybrid | RAG-Fusion canonical (two-level RRF) |
| `query_transform/multi_query_hybrid_rerank.yaml` | multi-query (4) | hybrid + cross-encoder | Full strong-baseline stack |

**HyDE** ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)) replaces the question with an LLM-generated hypothetical answer, on the rationale that an answer-shaped passage embeds closer to supporting documents than a question does. **Multi-query** ([RAG-Fusion](https://arxiv.org/abs/2402.03367)) issues several reformulations and fuses their rank lists. Both can derail when the LLM hallucinates — that trade-off is itself worth measuring on the advising corpus.

Two transforms can emit more than one search query. When that happens, the per-query rank lists are fused via `query.query_transform.fusion` (`rrf` default, or `max`). With hybrid retrieval, fusion is **two-level**: across reformulations *within* each retriever (the `query_transform.fusion` setting), then across retrievers (the hybrid's own `params.fusion`). By default a transformed query broadcasts to every retriever; HyDE's `hyde_hybrid` config opts into `branch` routing (`TransformedQuery.branch`) to send the hypothetical to the dense child and the original to BM25 — branch hints are silently ignored by dense/BM25-only setups, so the same config works in both.

### Chunking (size / semantic / contextual)

Standard chunking alternatives. Each changes only the indexing chunking config; since chunking is part of the index fingerprint, each builds (and caches) its own index.

| Config | Varies |
|--------|--------|
| `chunking/chunk_size_256.yaml` | recursive chunk size 256 (overlap 25) |
| `chunking/chunk_size_512.yaml` | recursive chunk size 512 (overlap 50) |
| `chunking/chunk_size_1024.yaml` | recursive chunk size 1024 (overlap 100) |
| `chunking/chunk_semantic.yaml` | semantic chunker (embedding-breakpoint splits) |
| `chunking/chunk_contextual.yaml` | contextual retrieval enricher (LLM situating context per chunk) |

**Chunk size** often moves retrieval quality more than chunking *strategy*, so the sweep measures it directly against the `base.yaml` default (1000/100). **Semantic chunking** splits on embedding-similarity breakpoints (`percentile`/`standard_deviation`/`interquartile`/`gradient`) rather than fixed character counts, using a small local model for breakpoint detection independent of the indexing embedder. **Contextual retrieval** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)) prepends a short, LLM-generated blurb situating each chunk within its surrounding document — added to the embedded + BM25-indexed text only (via `Chunk.index_text`), so the stored chunk, generator input, and RAGAS contexts stay original and the measured effect is isolated to retrieval. Scope is configurable (`document` / `page` / `window`); the shipped config uses a ±1-page window since the corpus is ingested per page.

### Agentic (single-agent ReAct)

A single reasoning LLM drives a reason→act(tool)→observe loop via **native tool calling**, deciding for itself when to search and when it has enough evidence to answer. Its `knowledge_base` tool searches the *same* index + retrieval/rerank stack as the linear baseline (retrieval-only, query-transform forced to passthrough), so the comparison isolates the agentic control flow.

| Config | Varies |
|--------|--------|
| `agent_single.yaml` (top-level) | `pipeline_mode: agent` — single-agent ReAct vs. the linear `base.yaml` |

```bash
python scripts/run_experiment.py configs/agent_single.yaml
```

Because `agent_single.yaml` inherits `base.yaml`'s indexing, it **shares base's cached index** (the fingerprint excludes agent/query/eval config) — so linear vs. single-agent run on identical retrieval and land in one comparison table. The agent unions the chunks retrieved across all tool calls (dedup by id, capped at `top_k_final`) into its result, so retrieval metrics (context precision/recall) score in agent mode just as in linear. `agent.max_iterations` bounds the loop; a final answer is forced if the budget is exhausted. Per-query loop traces (`iterations`, tool calls, chunk counts) land in `metadata` in the output JSONL.

### Smoke Test

`configs/smoke.yaml` (top-level, not under `experiments/`) routes the entire pipeline through local Ollama + HuggingFace with a 1-question dataset and sets `evaluation.mode: "none"` — the pipeline runs end-to-end and writes per-sample outputs to `run_1.jsonl`, but no RAGAS scoring (and no judge-LLM call) occurs. Use it to verify the pipeline without spending API credits or waiting on a local judge. Its results are not comparable to formal experiments.

`configs/smoke_agent.yaml` is the agent analog: the same all-local, no-judge setup but with `pipeline_mode: agent`, so the ReAct loop and `knowledge_base` tool run end-to-end on local Ollama. The reasoning model must support Ollama tool calling (qwen3 / qwen2.5 / llama3.1); if it doesn't, the loop's forced-final fallback still returns an answer.

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
# 1. Write the class in src/ragbench/components/<category>.py
from ragbench.components.base import BaseChunker, ComponentParams
from ragbench.core.registry import registry

@registry.register("chunking", "my_chunker")
class MyChunker(BaseChunker):
    class Params(ComponentParams):       # typed config: defaults + validation, one place
        custom_param: int = 42

    def __init__(self, config=None):
        super().__init__(config)
        self.p = self.Params.model_validate(self.config)

    def chunk(self, documents):
        ...  # read self.p.custom_param
```

```python
# 2. Ensure the file is imported in src/ragbench/components/__init__.py
from ragbench.components import my_module  # noqa: F401
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
make test-all    # All tests
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

## Background

This harness originated in UWF research on retrieval-augmented advising systems. The example replication configs above reconstruct the pipeline from:

Tamascelli, M., Bunch, O., Fowler, B., Taeb, M., & Cohen, A. (2025). Academic Advising Chatbot Powered with AI Agent. In *2025 ACM Southeast Conference (ACMSE 2025)*. ACM. https://doi.org/10.1145/3696673.3723065
