# Architecture

> Documents the system as built. For planned work, see
> [ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md).

This codebase is an experiment harness that measures how different
RAG component choices affect RAGAS evaluation metrics for an
academic advising chatbot. It is not a production chatbot.

---

## Design Principles

**Config-driven experimentation.** Every testable variable is a YAML
parameter. Adding an experiment means writing a short YAML override,
not editing Python.

**One source of truth per parameter.** `top_k` lives in
`RetrievalConfig`. The LLM model lives in `LLMConfig`. The prompt
text lives in `PromptConfig`. The pipeline reads these values;
nothing overrides them at runtime.

**Dependency Rule.** Dependencies point inward. The entry point
scripts import from pipeline and evaluation. Pipeline imports from
components and core. Evaluation imports from core. Components import
from core. Core imports from nothing internal. No layer reaches
upward.

**Interface-first components.** Each pipeline stage has an abstract
base class defining what goes in and what comes out. Implementations
are registered by name and resolved from config at runtime.
Swapping a component never requires changing pipeline code.

**Separated indexing and querying.** The index (chunking + embedding + vectorstore)
is built once and cached. Query-side experiments
(retrieval strategy, reranking, prompt, LLM) reuse the cached
index. A SHA-256 fingerprint of the indexing config determines
cache identity.

---

## Layers

```
┌─────────────────────────────────────────────┐
│  scripts/                                   │  Application boundary
│    run_experiment.py                        │  Orchestrates: load config →
│    compare.py                               │  validate → build → evaluate → save
└────────┬──────────────────────┬─────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  pipeline/      │   │  evaluation/         │  Orchestration
│    rag.py       │   │    evaluator.py      │  Pipeline wiring +
│    indexing.py  │   │    results.py        │  RAGAS integration
│    query.py     │   │    comparison.py     │
│    agent.py     │   │                      │
└────────┬────────┘   └──────────┬───────────┘
         │                       │
         ▼                       │
┌─────────────────────┐          │
│  components/        │          │  Domain
│    base.py (ABCs)   │          │  Interfaces + implementations
│    chunkers.py      │          │
│    embedders.py     │          │
│    generators.py    │          │
│    ...              │          │
└────────┬────────────┘          │
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────┐
│  core/                                      │  Foundation
│    types.py    config.py    registry.py     │  Shared types, config,
│    git.py                                   │  component registry
└─────────────────────────────────────────────┘
```

Key dependency boundaries:

- `evaluation/evaluator.py` depends on `core.types.Queryable` (a
  Protocol), not on any pipeline class. It receives a pipeline as
  a duck-typed object. This means the evaluator is designed to work
  identically for `QueryPipeline`, `AgentPipeline` (once
  implemented), and test mocks.

- `pipeline/rag.py` does not import from `evaluation/`. The
  experiment orchestration (build pipeline → evaluate → save)
  lives in `scripts/run_experiment.py`, at the application
  boundary.

- `components/__init__.py` triggers registration side effects
  (`@registry.register` decorators). This import happens only at
  application entry points: `scripts/run_experiment.py` and
  `tests/conftest.py`. Pipeline modules never trigger it.

---

## File Map

```
src/
├── core/
│   ├── types.py          Data contracts between pipeline stages
│   ├── config.py         Config dataclasses + YAML loading + validation
│   ├── registry.py       Component registry (@register / get)
│   └── git.py            Git SHA + dirty flag for reproducibility
│
├── components/
│   ├── base.py           Abstract base classes (read-only contract)
│   ├── defaults.py       Passthrough/no-op defaults (always available)
│   ├── ingestors.py      PDFIngestor (PyMuPDF)
│   ├── chunkers.py       LangChainRecursiveChunker, CustomRecursiveChunker
│   ├── embedders.py      HuggingFaceEmbedder, GoogleEmbedder
│   ├── vectorstores.py   FAISSVectorStore, ChromaVectorStore
│   ├── retrievers.py     DenseRetriever
│   ├── generators.py     OllamaGenerator, EdenAIGenerator, GoogleGenerator
│   ├── prompts.py        ChatPromptTemplate
│   ├── query_transforms.py  ContextualizerQueryTransformer
│   ├── rerankers.py      (stub — Phase 3C)
│   ├── tools.py          (stub — Phase 4)
│   └── __init__.py       Imports all implementation files → triggers registration
│
├── pipeline/
│   ├── indexing.py        IndexArtifact + IndexingPipeline
│   ├── query.py           QueryPipeline (linear RAG)
│   ├── agent.py           AgentPipeline (stub — Phase 4)
│   └── rag.py             RAGPipeline (top-level dispatcher)
│
└── evaluation/
    ├── evaluator.py       RAGAS integration + multi-run execution
    ├── results.py         Experiment result serialization
    └── comparison.py      Cross-experiment tables + config diffing

configs/
├── base.yaml              Default baseline (all experiments inherit)
└── experiments/            Per-experiment overrides (inherit from base)

scripts/
├── run_experiment.py      CLI entry point for running experiments
└── compare.py             CLI entry point for comparing results

tests/
├── conftest.py            Fixtures + sys.path + registration trigger
├── fixtures/              Test YAML configs + sample PDF
└── test_*.py              Per-module unit tests
```

---

## Data Flow

### Linear Pipeline

```
Source PDFs
    │
    ▼
[Ingestor] ─── one per source in config, tags source_name
    │
    ▼
list[Document]
    │
    ▼
[Chunker] ─── shared across sources
    │
    ▼
list[Chunk] ─── each carries source_name in metadata
    │
    ▼
[Embedder] ─── produces float vectors
    │
    ▼
list[EmbeddedChunk]
    │
    ▼
[VectorStore] ─── FAISS index + chunk metadata, persisted to disk
    │
    ║  ═══ INDEX BOUNDARY ═══  cached by fingerprint
    ║
    ▼
User Query
    │
    ▼
[QueryTransformer] ─── returns list[str] (1 for passthrough, N for multi-query)
    │
    ▼
[Retriever] ─── embeds query, searches vectorstore, deduplicates across queries
    │
    ▼
list[RetrievedChunk]  (top_k_retrieve candidates)
    │
    ▼
[Reranker] ─── re-scores, truncates to top_k_final
    │
    ▼
[PromptTemplate] ─── formats query + chunks + history → prompt
    │
    ▼
[Generator] ─── pure LLM call, receives formatted prompt only
    │
    ▼
GenerationResult ─── pipeline attaches query + chunks to LLM answer
    │
    ▼
[Evaluator] ─── RAGAS metrics against ground truth
```

The critical handoff is PromptTemplate → Generator. The generator
never sees raw chunks. The pipeline calls `prompt_template.format()`
to produce the prompt, passes it to `generator.generate()`, then
assembles the final `GenerationResult` with query and chunks. This
is what makes YAML prompt changes actually affect LLM output.

### Retrieval-Only Mode

When `evaluation.mode: "retrieval_only"`, the pipeline skips
generator and prompt template construction entirely. `run()`
returns a `GenerationResult` with empty `answer` and populated
`retrieved_chunks`. The evaluator computes only retriever metrics
(context_precision, context_entity_recall). This skips the
generation LLM entirely, but the evaluator LLM is still invoked
— RAGAS metrics like context_precision require an LLM judge.
Faster and cheaper than full mode, not free.

---

## Type System

All types live in `src/core/types.py`.

| Type | Produced By | Consumed By |
|------|------------|-------------|
| `Document` | Ingestor | Chunker |
| `Chunk` | Chunker | Embedder |
| `EmbeddedChunk` | Embedder | VectorStore |
| `RetrievedChunk` | VectorStore / Retriever | Reranker, PromptTemplate |
| `GenerationResult` | Pipeline (assembled) | Evaluator |
| `ToolResult` | BaseTool | AgentPipeline (future) |
| `AgentStep` | AgentPipeline (future) | Results / logging |
| `EvalSample` | Evaluator | RAGAS |
| `ScoredSample` | Evaluator | Result files (JSONL) |
| `ExperimentResult` | Evaluator | `results.py` serialization |

`Queryable` is a `@runtime_checkable Protocol` in `core/types.py`:

```python
class Queryable(Protocol):
    def query(self, question: str) -> GenerationResult: ...
```

`RAGPipeline` satisfies it. `AgentPipeline` will satisfy it. Test
mocks satisfy it. The evaluator depends on this protocol, never on
concrete pipeline classes.

`IndexArtifact` lives in `pipeline/indexing.py` (not in
`core/types.py`) because it references `BaseVectorStore` and
`BaseEmbedder` from the components layer. Placing it in the
pipeline layer avoids a circular import and keeps the type
annotations real (no `Any`).

---

## Config System

### Structure

```
ExperimentConfig
├── name, description, pipeline_mode
├── IndexingConfig
│   ├── sources: list[SourceConfig]    per-source ingest type + path
│   ├── chunking: ComponentConfig      type + params
│   ├── embedding: ComponentConfig
│   └── vectorstore: ComponentConfig
├── QueryConfig                        (linear pipeline)
│   ├── query_transform: ComponentConfig
│   ├── retrieval: RetrievalConfig     top_k_retrieve, top_k_final, filters
│   ├── reranking: ComponentConfig
│   ├── generation: ComponentConfig
│   ├── generation_llm: LLMConfig      provider, model, temperature
│   └── prompt: PromptConfig           system_template, CoT, citation style
├── AgentConfig                        (agent pipeline — future)
│   ├── mode, llm, supervisor, memory
│   ├── agents: list[AgentDefinitionConfig]
│   └── tools: list[AgentDefinitionConfig]
└── EvaluationConfig
    ├── dataset, mode, num_runs
    ├── metrics, retrieval_only_metrics
    ├── evaluator_llm: LLMConfig
    └── run_config: EvalRunConfig    timeout, retries, workers
```

### Dedicated vs Generic Config Types

| Type | Why Dedicated |
|------|---------------|
| `RetrievalConfig` | `top_k_retrieve`, `top_k_final`, and `filters` are too important to bury in a generic params dict. They are the most commonly changed retrieval variables. |
| `PromptConfig` | `system_template`, `use_chain_of_thought`, `citation_style` are each independently testable. |
| `LLMConfig` | Reused by generator, query transformer, reranker, evaluator, supervisor. Stable shared structure. |
| `ComponentConfig` | For stages where params are implementation-specific and vary widely (chunkers, embedders, vectorstores). Passed to the implementation as `config: dict`. |
| `EvalRunConfig` | `timeout`, `max_retries`, `max_wait`, `max_workers` for the RAGAS evaluator. Differs between local and cloud setups. |

### Inheritance

Experiment YAML files declare `extends: ../base.yaml`. Resolution
uses `load_yaml_with_inheritance()`:

- Nested dicts: merge recursively (override only specified keys)
- Lists: full replacement (child provides the entire list)
- Scalars: child replaces parent

The `extends` path is relative to the child file's directory.
Multi-level inheritance is supported (child → parent → grandparent).

### Index Fingerprinting

`ExperimentConfig.index_fingerprint()` returns a 12-char hex SHA-256
of: sources + chunking + embedding + vectorstore config.

**Included:** Source names, paths, ingest types/params. Chunking,
embedding, vectorstore types and all params.

**Excluded:** Everything in QueryConfig, AgentConfig,
EvaluationConfig. Pipeline mode.

Two experiments with the same fingerprint share a cached index.
This means query-side experiments (retrieval strategy, reranking,
prompt template, LLM model) never re-index.

### Validation

`validate_config(config, registry)` runs before pipeline
construction (called in `scripts/run_experiment.py`). It checks:

- Every component type referenced in config is registered
- `top_k_final <= top_k_retrieve` and both are positive
- `pipeline_mode: "agent"` has agents or tools defined
- `evaluation.mode: "full"` has a generation type and model name
- `generation_llm.model_name` is non-empty when generation needed
- `indexing.sources` is non-empty
- `EvalRunConfig` values are positive (timeout, workers, wait)
- Evaluation dataset file exists on disk

All errors are collected into a list and raised as a single
`ConfigValidationError` with a readable bullet-list message. This
catches typos and constraint violations before any heavyweight
work (model loading, embedding) begins.

---

## Component Interfaces

All abstract base classes live in `src/components/base.py`.

| Interface | Registry Category | Primary Method | Signature |
|-----------|------------------|----------------|-----------|
| `BaseIngestor` | `ingest` | `ingest(path)` | `str → list[Document]` |
| `BaseChunker` | `chunking` | `chunk(docs)` | `list[Document] → list[Chunk]` |
| `BaseEmbedder` | `embedding` | `embed_chunks(chunks)` | `list[Chunk] → list[EmbeddedChunk]` |
| | | `embed_query(query)` | `str → list[float]` |
| `BaseVectorStore` | `vectorstore` | `add(chunks)` | `list[EmbeddedChunk] → None` |
| | | `search(vec, top_k, filters)` | `→ list[RetrievedChunk]` |
| | | `save(dir)` / `load(dir)` | Persistence for caching |
| `BaseRetriever` | `retrieval` | `retrieve(query, top_k, filters)` | `str → list[RetrievedChunk]` |
| `BaseQueryTransformer` | `query_transform` | `transform(query, history)` | `str → list[str]` |
| `BaseReranker` | `reranking` | `rerank(query, chunks, top_k)` | `→ list[RetrievedChunk]` |
| `BaseGenerator` | `generation` | `generate(prompt)` | `str\|list[dict] → GenerationResult` |
| `BasePromptTemplate` | `prompts` | `format(query, chunks, history)` | `→ str\|list[dict]` |
| `BaseTool` | `tool` | `execute(query)` | `str → ToolResult` |
| `BaseMemory` | `memory` | `add_turn()` / `get_history()` / `clear()` | Turn management |

### Key Interface Decisions

**BaseGenerator takes a formatted prompt, not raw chunks.** The
pipeline calls `PromptTemplate.format()` first. The generator is a
pure LLM wrapper that knows nothing about retrieval context. This
is what makes prompt config changes actually affect output.

**BaseRetriever holds injected dependencies.** The vectorstore and
embedder are set via `set_vectorstore()` / `set_embedder()` after
construction. Default filters from config are set via
`set_default_filters()`. This allows the pipeline to construct
components independently from config and wire them afterward.

**BaseQueryTransformer always returns a list.** Even single-query
transforms return `[query]`. The pipeline deduplicates results
across all returned queries. This naturally supports passthrough
(1 query), HyDE (1 transformed query), multi-query (N queries),
and sub-question decomposition (N sub-queries).

**BaseVectorStore.search() takes explicit filters.** Not buried
in `**kwargs`. FAISS doesn't support native filtering, so its
implementation over-retrieves (4× top_k, minimum 50), filters by
metadata, and truncates.

### Registered Implementations

| Category | Name | Class | File |
|----------|------|-------|------|
| `ingest` | `pdf` | `PDFIngestor` | `ingestors.py` |
| `chunking` | `recursive_langchain` | `LangChainRecursiveChunker` | `chunkers.py` |
| `chunking` | `recursive_custom` | `CustomRecursiveChunker` | `chunkers.py` |
| `embedding` | `huggingface` | `HuggingFaceEmbedder` | `embedders.py` |
| `embedding` | `google` | `GoogleEmbedder` | `embedders.py` |
| `vectorstore` | `faiss` | `FAISSVectorStore` | `vectorstores.py` |
| `vectorstore` | `chroma` | `ChromaVectorStore` | `vectorstores.py` |
| `retrieval` | `dense` | `DenseRetriever` | `retrievers.py` |
| `generation` | `ollama` | `OllamaGenerator` | `generators.py` |
| `generation` | `edenai` | `EdenAIGenerator` | `generators.py` |
| `generation` | `google` | `GoogleGenerator` | `generators.py` |
| `prompts` | `chat` | `ChatPromptTemplate` | `prompts.py` |
| `query_transform` | `passthrough` | `PassthroughQueryTransformer` | `defaults.py` |
| `query_transform` | `contextualizer` | `ContextualizerQueryTransformer` | `query_transforms.py` |
| `reranking` | `none` | `NoOpReranker` | `defaults.py` |
| `memory` | `none` | `NoMemory` | `defaults.py` |
| `memory` | `buffer_window` | `BufferWindowMemory` | `defaults.py` |

---

## Pipeline Orchestration

### IndexingPipeline (`pipeline/indexing.py`)

Builds the vector index from source documents.

`from_config(config)` constructs per-source ingestors plus shared
chunker, embedder, and vectorstore from the registry.

`build()` iterates sources (ingest → tag source_name), chunks all
documents, embeds, stores. Returns an `IndexArtifact` carrying the
populated vectorstore and embedder.

`run_or_load_cache(config)` checks if
`data/indexes/<fingerprint>/` exists. If yes, constructs a fresh
pipeline (to get embedder + vectorstore instances) and loads the
stored index. If no, builds from scratch and saves.

**Cache load note:** Loading from cache still instantiates the
embedding model (needed at query time for `embed_query()`). For
large models this takes several seconds. This is a known cost, not
a bug.

### QueryPipeline (`pipeline/query.py`)

Linear RAG: transform → retrieve → rerank → format → generate.

`from_config(config, index_artifact, retrieval_only)` builds each
component from the registry, wires the retriever to the
vectorstore/embedder from the `IndexArtifact`, and sets default
filters.

`run(query, history)` executes the pipeline. All top_k values and
filters come from config — `run()` takes no override parameters.
Deduplication across multi-query results keeps the highest score
per chunk_id and sorts descending.

### AgentPipeline (`pipeline/agent.py`)

Stub. Defines the class interface (`__init__`, `run`, `query`)
and raises `NotImplementedError`. Will support single-agent (v2)
and multi-agent (v3) modes in Phase 4.

### RAGPipeline (`pipeline/rag.py`)

Top-level dispatcher. `from_config(config, no_cache)` builds the
index via `IndexingPipeline.run_or_load_cache()`, then constructs
either a `QueryPipeline` (linear mode) or raises
`NotImplementedError` (agent mode).

`query(question)` delegates to the active pipeline's `run()`.
Satisfies the `Queryable` protocol.

---

## Evaluation

### Evaluator (`evaluation/evaluator.py`)

`Evaluator(config)` wraps the RAGAS evaluation library.

`evaluate(pipeline, embedder, experiment_name)` runs the
pipeline against every question in the dataset, `num_runs` times.
Failed queries are logged and skipped (one bad response doesn't
crash the run). Returns an `ExperimentResult` with aggregated and
per-run metrics.

**RAGAS integration:** Uses the RAGAS 0.4.x API
(`EvaluationDataset`, `SingleTurnSample`, private metric
submodules to avoid deprecation warnings). Metrics are built once
and cached via `@functools.cache`. The pipeline's embedder is
reused for RAGAS through an `_EmbedderAdapter` that implements the
LangChain Embeddings interface — avoids loading a second model.

**Evaluator LLM:** RAGAS needs an LLM judge for most metrics
(faithfulness, answer_correctness, context_precision, etc.).
Configured via `evaluator_llm` in the config. Currently supports
Ollama (local) and EdenAI (cloud) providers. Built through
`_build_evaluator_llm()` which returns a `LangchainLLMWrapper`.

**Run config:** `EvalRunConfig` controls RAGAS execution settings:
`timeout` (seconds per evaluation call), `max_retries`,
`max_wait` (backoff ceiling), `max_workers` (parallel evaluation
threads). These differ between local Ollama (longer timeout, fewer
workers) and cloud API (shorter timeout, more workers) setups.

**Aggregation:** Per-metric mean and standard deviation across
runs. NaN and None values (RAGAS returns these when the LLM judge
fails to parse) are filtered before averaging. Metric keys are
collected from all runs (not just the first), handling the case
where different runs produce different metric sets.

### Results (`evaluation/results.py`)

`save_experiment(config, result, base_dir, git_info)` writes:

```
results/<name>/
  summary.json    Aggregated metrics + config snapshot + git SHA/dirty
  config.yaml     Full resolved config (dataclasses.asdict → YAML)
  run_1.jsonl     Per-sample ScoredSample data
  run_2.jsonl
  ...
```

The experiment directory is cleared and recreated on each save
(prevents stale run files from previous executions with different
`num_runs`).

`summary.json` includes a curated `config_snapshot` with the key
experimental variables at a glance (chunk_size, embedding_model,
retrieval_type, generation_model, etc.) — not the full config
(which lives in `config.yaml`).

### Comparison (`evaluation/comparison.py`)

`compare_experiments(result_dirs)` loads summaries and produces
comparison rows. `format_comparison_table(rows)` renders Markdown
matching the ACMSE paper Table 2 format.

`diff_configs(dir_a, dir_b)` flattens both configs to dotted key
paths and returns only the keys that differ — essential for
verifying single-variable isolation.

`load_per_sample_scores(result_dirs, run)` loads per-question
scores across experiments for drill-down analysis.

The `scripts/compare.py` CLI wraps these with Rich tables,
color-coded thresholds, and output format options (table, markdown,
CSV, JSON).

---

## Component Registry

`core/registry.py` provides a global `ComponentRegistry` singleton.

**Registration:** `@registry.register("category", "name")` on a
class. Duplicate registrations raise `ValueError`. All
registrations are triggered by importing `components/__init__.py`,
which imports every implementation file.

**Resolution:** `registry.get("category", "name")` returns the
class (not an instance). Construction happens at the call site:
`cls = registry.get(...); instance = cls(config=params)`.

**Introspection:** `registry.list_category("chunking")` returns
registered names. `registry.is_registered(...)` checks existence.

**Testing:** `registry.clear()` enables isolated tests with a
fresh `ComponentRegistry()` instance.

---

## Experiment Lifecycle

```
scripts/run_experiment.py
    │
    ├── import components          (triggers registration)
    ├── ExperimentConfig.from_yaml (loads + resolves inheritance)
    ├── validate_config            (checks types, constraints, files)
    ├── get_git_sha / get_git_dirty (captures repo state)
    │
    ├── RAGPipeline.from_config
    │   ├── IndexingPipeline.run_or_load_cache
    │   │   ├── check fingerprint cache → load or build
    │   │   └── return IndexArtifact
    │   └── QueryPipeline.from_config
    │       └── wire retriever ← vectorstore + embedder
    │
    ├── Evaluator.evaluate(rag, embedder)
    │   ├── for each run:
    │   │   ├── for each question: rag.query() → EvalSample
    │   │   └── RAGAS evaluate → per-sample scores
    │   └── aggregate metrics (mean ± std)
    │
    └── save_experiment
        └── write summary.json + config.yaml + run_N.jsonl
```

---

## Adding a New Component

1. Write the class in the appropriate `src/components/*.py` file
2. Inherit from the base class, implement required methods
3. Decorate with `@registry.register("category", "name")`
4. Ensure the file is imported in `src/components/__init__.py`
5. Add an inventory assertion in `tests/test_registry.py`
6. Reference `type: "name"` in a YAML experiment config
7. Run `make qa` to verify

No pipeline code changes. No config system changes. No evaluation
changes.
