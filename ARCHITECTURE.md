# Architecture

> Documents the system as built. For planned work, see
> [ROADMAP.md](ROADMAP.md).

This codebase is a general-purpose RAG experimentation harness. It
measures how different RAG component choices — chunking, embedding,
retrieval, reranking, query transformation, generation — affect
end-to-end evaluation metrics. It is a research testbed, not a
production chatbot.

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
│   ├── fusion.py         Rank-list fusion primitives (RRF + weighted + max-dedup)
│   └── git.py            Git SHA + dirty flag for reproducibility
│
├── components/
│   ├── base.py           Abstract base classes (read-only contract)
│   ├── defaults.py       Passthrough/no-op defaults (always available)
│   ├── ingestors.py      PDFIngestor (PyMuPDF)
│   ├── chunkers.py       LangChainRecursiveChunker, CustomRecursiveChunker, SemanticChunker
│   ├── enrichers.py      ContextualChunkEnricher (chunk_enricher) — sets index_text pre-embedding
│   ├── embedders.py      HuggingFaceEmbedder, GoogleEmbedder, OpenAIEmbedder, EdenAIEmbedder
│   ├── vectorstores.py   FAISSVectorStore, ChromaVectorStore
│   ├── lexical_indexes.py   BM25LexicalIndex (bm25s + PyStemmer)
│   ├── retrievers.py     DenseRetriever, BM25Retriever, HybridRetriever
│   ├── generators.py     OllamaGenerator, EdenAIGenerator, GoogleGenerator, OpenAIGenerator
│   ├── prompts.py        ChatPromptTemplate
│   ├── query_transforms.py  ContextualizerQueryTransformer, HyDEQueryTransformer, MultiQueryQueryTransformer
│   ├── rerankers.py      CrossEncoderReranker
│   ├── tools.py          (stub — Phase D)
│   └── __init__.py       Imports all implementation files → triggers registration
│
├── pipeline/
│   ├── indexing.py        IndexArtifact + IndexingPipeline (vector + auxiliary stores)
│   ├── query.py           QueryPipeline (linear RAG)
│   ├── agent.py           AgentPipeline (stub — Phase D)
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
[ChunkEnricher (optional)] ─── contextual retrieval; sets chunk.index_text
    │                          (the embed/index text) without touching
    │                          content (stored / generated / scored). Runs
    │                          before the lexical index + embedder.
    │
    ├──► [LexicalIndex (optional)] ─── BM25 etc.; indexes text_for_index;
    │                                    built before embedding so tokenizer
    │                                    errors fail fast. Stored under
    │                                    IndexArtifact.auxiliary_stores
    ▼
[Embedder] ─── embeds text_for_index → float vectors
    │
    ▼
list[EmbeddedChunk]
    │
    ▼
[VectorStore] ─── FAISS or Chroma index + chunk metadata, persisted to disk
    │
    ║  ═══ INDEX BOUNDARY ═══  cached by fingerprint (vector + sparse together)
    ║
    ▼
User Query
    │
    ▼
[QueryTransformer] ─── returns list[TransformedQuery] (1 for passthrough,
    │                   1 hypothetical for HyDE, N for multi-query). Each
    │                   carries an optional branch hint for hybrid routing.
    ▼
[Retriever.retrieve_multi] ─── dense / BM25 / hybrid. Fuses per-query
    │             rank lists via query_transform.fusion (rrf/max).
    │             Hybrid routes each query to children by branch and
    │             applies two-level fusion: across reformulations
    │             (intra-branch), then across retrieval methods
    │             (cross-branch, the hybrid's own fusion).
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

### Evaluation Modes

`evaluation.mode` controls how much of the eval loop runs. Three
values are supported:

- **`full`** (default): retrieve + generate + score with all
  configured RAGAS metrics. Calls both the generation LLM and the
  judge LLM.
- **`retrieval_only`**: pipeline skips generator and prompt
  template construction entirely. `run()` returns a
  `GenerationResult` with empty `answer` and populated
  `retrieved_chunks`. The evaluator computes only retriever
  metrics (context_precision, context_entity_recall). The judge
  LLM is still invoked — RAGAS retriever metrics require it.
  Faster and cheaper than `full`, not free.
- **`none`**: pipeline runs end-to-end against the dataset, but
  the evaluator skips RAGAS scoring entirely. No judge LLM or
  judge embedder is built. Per-sample query / response /
  retrieved_contexts are still written to `run_<N>.jsonl` (with
  empty `scores`), so the output remains inspectable. Intended
  for smoke tests that verify pipeline plumbing without paying
  for a judge.

---

## Type System

All types live in `src/core/types.py`.

| Type | Produced By | Consumed By |
|------|------------|-------------|
| `Document` | Ingestor | Chunker |
| `Chunk` | Chunker | Embedder |
| `EmbeddedChunk` | Embedder | VectorStore |
| `TransformedQuery` | QueryTransformer | Retriever (`retrieve_multi`) |
| `RetrievedChunk` | VectorStore / Retriever | Reranker, PromptTemplate |
| `GenerationResult` | Pipeline (assembled) | Evaluator |
| `ToolResult` | BaseTool | AgentPipeline (future) |
| `AgentStep` | AgentPipeline (future) | Results / logging |
| `EvalSample` | Evaluator | RAGAS |
| `ScoredSample` | Evaluator | Result files (JSONL) |
| `ExperimentResult` | Evaluator | `results.py` serialization |

`TransformedQuery` is a frozen dataclass carrying a search-query string
plus an optional `branch` hint. `branch=None` broadcasts to every
retriever (the common case — passthrough, contextualizer, multi-query);
a non-None branch (e.g. `"dense"`, `"bm25"`) routes the query to the
matching child inside a `HybridRetriever`. All transformers broadcast
by default; routing is opt-in. The canonical HyDE+hybrid config uses it
to send the hypothetical document to the dense branch and the original
question to BM25. Non-hybrid retrievers ignore the hint, so a
branch-tagged config runs unchanged on a dense-only setup. A non-None
branch that matches no child raises `ValueError` at retrieval time
(the validator also catches explicitly-set mismatches up front).

`Queryable` is a `@runtime_checkable Protocol` in `core/types.py`:

```python
class Queryable(Protocol):
    def query(self, question: str) -> GenerationResult: ...
```

`RAGPipeline` satisfies it. `AgentPipeline` will satisfy it. Test
mocks satisfy it. The evaluator depends on this protocol, never on
concrete pipeline classes.

`IndexArtifact` lives in `pipeline/indexing.py` (not in
`core/types.py`) because it references `BaseVectorStore`,
`BaseEmbedder`, and `BaseLexicalIndex` from the components layer.
Placing it in the pipeline layer avoids a circular import and keeps
the type annotations real (no `Any`).

The artifact carries three things:

- `vectorstore: BaseVectorStore` — populated dense index.
- `embedder: BaseEmbedder` — needed at query time to embed incoming
  queries.
- `auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore]`
  — optional indexes built alongside the vector store. The BM25
  index, when configured, lives under the key `"bm25"`. The dict is
  empty for dense-only experiments; it leaves room for a future
  SPLADE store (a `BaseVectorStore`) without changing the artifact's
  shape.

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
│   ├── vectorstore: ComponentConfig
│   ├── sparse_index: ComponentConfig  optional; e.g. {type: bm25}
│   └── chunk_enricher: ComponentConfig  optional; e.g. {type: contextual}
├── QueryConfig                        (linear pipeline)
│   ├── query_transform: QueryTransformConfig  type + fusion + params
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
    ├── evaluator_embedding: ComponentConfig
    └── run_config: EvalRunConfig    timeout, retries, workers
```

### Dedicated vs Generic Config Types

| Type | Why Dedicated |
|------|---------------|
| `RetrievalConfig` | `top_k_retrieve`, `top_k_final`, and `filters` are too important to bury in a generic params dict. They are the most commonly changed retrieval variables. |
| `QueryTransformConfig` | `fusion` (how N>1 transformer outputs are combined) is a first-class experimental variable, parallel to `RetrievalConfig`. Burying it in `params` would hide it from config diffs and the comparison snapshot. |
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
of: sources + chunking + embedding + vectorstore + sparse_index +
chunk_enricher (the last two only when set) config.

**Included:** Source names, paths, ingest types/params. Chunking,
embedding, vectorstore types and all params. Sparse-index type and
params, *only when its `type` is non-empty* — empty `ComponentConfig`s
canonicalize to absence, so adding an unused optional component
doesn't disturb the fingerprint of existing experiments.

**Excluded:** Everything in QueryConfig, AgentConfig,
EvaluationConfig. Pipeline mode.

Two experiments with the same fingerprint share a cached index.
This means query-side experiments (retrieval strategy, reranking,
prompt template, LLM model) never re-index. Switching from dense to
hybrid (which adds `sparse_index`) does change the fingerprint and
trigger a one-time rebuild — the vector embeddings are recomputed
alongside the new sparse index, by design.

### Validation

`validate_config(config, registry)` runs before pipeline
construction (called in `scripts/run_experiment.py`). It checks:

- Every component type referenced in config is registered
- `query.query_transform.fusion` is one of `{rrf, max}`
- Transformer `branch` / `original_branch` hints reference a real
  hybrid child name (static check; only when retrieval is `hybrid`).
  The runtime guard in `HybridRetriever.retrieve_multi` is authoritative
- `top_k_final <= top_k_retrieve` and both are positive
- `pipeline_mode: "agent"` has agents or tools defined
- `evaluation.mode` is one of `{full, retrieval_only, none}`
- Modes other than `retrieval_only` require a generation type, prompt
  type, and `generation_llm.model_name`
- `indexing.sources` is non-empty
- `EvalRunConfig` values are positive (timeout, workers, wait)
- `evaluator_llm.provider` is one of `{ollama, edenai, google, openai}` (or empty)
- `evaluator_embedding.type` is registered in the `embedding` category
  (required for modes other than `none`)
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
| `BaseChunkEnricher` | `chunk_enricher` | `enrich(docs, chunks)` | `→ list[Chunk]` (may set `index_text`) |
| `BaseEmbedder` | `embedding` | `embed_chunks(chunks)` | `list[Chunk] → list[EmbeddedChunk]` |
| | | `embed_query(query)` | `str → list[float]` |
| `BaseVectorStore` | `vectorstore` | `add(chunks)` | `list[EmbeddedChunk] → None` |
| | | `search(vec, top_k, filters)` | `→ list[RetrievedChunk]` |
| | | `save(dir)` / `load(dir)` | Persistence for caching |
| `BaseLexicalIndex` | `sparse_index` | `add(chunks)` | `list[Chunk] → None` (takes raw text, no embeddings) |
| | | `search(query, top_k, filters)` | `→ list[RetrievedChunk]` |
| | | `save(dir)` / `load(dir)` | Persistence for caching |
| `BaseRetriever` | `retrieval` | `retrieve(query, top_k, filters)` | `str → list[RetrievedChunk]` |
| | | `retrieve_multi(queries, top_k, fusion, filters)` | `list[TransformedQuery] → list[RetrievedChunk]` |
| `BaseQueryTransformer` | `query_transform` | `transform(query, history)` | `str → list[TransformedQuery]` |
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

**BaseRetriever holds injected dependencies.** Dense retrievers
receive vectorstore + embedder via `set_vectorstore()` /
`set_embedder()`. Lexical retrievers receive a sparse index via
`set_sparse_index()` (the BM25 index lives in
`IndexArtifact.auxiliary_stores["bm25"]`). Hybrid retrievers compose
child retrievers and inject each child's sources independently.
Default filters from config are set via `set_default_filters()` and
the hybrid retriever pushes them down to every child so per-branch
over-retrieve + filter semantics stay consistent with the
single-retriever paths.

**HybridRetriever composes children; fusion lives in `core.fusion`.**
The hybrid retriever holds an ordered list of named child retrievers
plus a fusion strategy (RRF or weighted). Sub-retrievers are declared
in YAML as a list of `{name, type, top_k, params}` entries; named
weights (dict keyed by sub-retriever `name`) are used for weighted
fusion so config diffs stay stable under list reordering. Nested
hybrid retrievers are disallowed by the validator. Multi-query fusion
(across query reformulations) is handled by `retrieve_multi`, which the
hybrid layers *on top of* its cross-retriever fusion — see the
two-level fusion note under Key Interface Decisions. Both levels share
the same `core.fusion` helpers. The fused output's `retrieval_method`
is set to `"hybrid_rrf"` or `"hybrid_weighted"` rather than inheriting
a child's label.

**BaseQueryTransformer always returns a list of `TransformedQuery`.**
Even single-query transforms return `[TransformedQuery(text=query)]`.
This naturally supports passthrough (1 query), HyDE (1 hypothetical,
optionally tagged for a branch), multi-query (N queries), and
sub-question decomposition (N sub-queries). The `branch` field lets a
transformer route specific queries to specific hybrid children without
the pipeline needing to know the transformer's internals.

**Fusion across transformer outputs lives in `BaseRetriever.retrieve_multi`,
not the pipeline.** The pipeline calls `retrieve_multi(transformed,
top_k, fusion)` and the retriever owns the fan-out + fusion. The
default implementation (dense, BM25) ignores branch hints, runs each
query, and fuses the rank lists via `core.fusion`
(`reciprocal_rank_fusion` for `rrf`, `max_score_dedup` for `max`; a
single query short-circuits both since fusion is then a no-op).
`HybridRetriever` overrides it
to route queries by branch and apply **two-level fusion**: intra-branch
(across reformulations within a child, using the pipeline-level
`fusion`) then cross-branch (across children, using the hybrid's own
`params.fusion`). Two-level is deliberate — a single flat RRF over all
(query × retriever) lists would bias toward whichever branch received
more queries; two-level preserves the hybrid's intended branch
weighting regardless of reformulation count. A branch hint that matches
no child raises `ValueError`.

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
| `chunking` | `semantic` | `SemanticChunker` | `chunkers.py` |
| `chunk_enricher` | `none` | `NoOpChunkEnricher` | `defaults.py` |
| `chunk_enricher` | `contextual` | `ContextualChunkEnricher` | `enrichers.py` |
| `embedding` | `huggingface` | `HuggingFaceEmbedder` | `embedders.py` |
| `embedding` | `google` | `GoogleEmbedder` | `embedders.py` |
| `embedding` | `openai` | `OpenAIEmbedder` | `embedders.py` |
| `embedding` | `edenai` | `EdenAIEmbedder` | `embedders.py` |
| `vectorstore` | `faiss` | `FAISSVectorStore` | `vectorstores.py` |
| `vectorstore` | `chroma` | `ChromaVectorStore` | `vectorstores.py` |
| `sparse_index` | `bm25` | `BM25LexicalIndex` | `lexical_indexes.py` |
| `retrieval` | `dense` | `DenseRetriever` | `retrievers.py` |
| `retrieval` | `bm25` | `BM25Retriever` | `retrievers.py` |
| `retrieval` | `hybrid` | `HybridRetriever` | `retrievers.py` |
| `query_transform` | `passthrough` | `PassthroughQueryTransformer` | `defaults.py` |
| `query_transform` | `contextualizer` | `ContextualizerQueryTransformer` | `query_transforms.py` |
| `query_transform` | `hyde` | `HyDEQueryTransformer` | `query_transforms.py` |
| `query_transform` | `multi_query` | `MultiQueryQueryTransformer` | `query_transforms.py` |
| `reranking` | `none` | `NoOpReranker` | `defaults.py` |
| `reranking` | `cross_encoder` | `CrossEncoderReranker` | `rerankers.py` |
| `generation` | `ollama` | `OllamaGenerator` | `generators.py` |
| `generation` | `edenai` | `EdenAIGenerator` | `generators.py` |
| `generation` | `google` | `GoogleGenerator` | `generators.py` |
| `generation` | `openai` | `OpenAIGenerator` | `generators.py` |
| `prompts` | `chat` | `ChatPromptTemplate` | `prompts.py` |
| `memory` | `none` | `NoMemory` | `defaults.py` |
| `memory` | `buffer_window` | `BufferWindowMemory` | `defaults.py` |

---

## Pipeline Orchestration

### IndexingPipeline (`pipeline/indexing.py`)

Builds the vector index — and any configured auxiliary indexes —
from source documents.

`from_config(config)` constructs per-source ingestors plus shared
chunker, embedder, vectorstore, and (when `indexing.sparse_index.type`
is set) a sparse index, all from the registry.

`build()` iterates sources (ingest → tag source_name), chunks all
documents, then runs **sparse indexing before embedding** so a
misconfigured BM25 tokenizer/stemmer fails fast — before spending
embedding API budget. After embedding, the vector store is populated.
Returns an `IndexArtifact` carrying the vectorstore, embedder, and an
`auxiliary_stores` dict (containing the BM25 index under `"bm25"`
when configured).

`run_or_load_cache(config)` checks if
`data/indexes/<fingerprint>/` exists. If yes, constructs a fresh
pipeline (to get embedder + vectorstore + sparse-index instances)
and loads each from disk: the vector store from the fingerprint
directory, and the BM25 index (if any) from the `bm25/` subdirectory
underneath. If no, builds from scratch and saves.

On save: the vector store writes its native files at the fingerprint
root; the sparse index writes to `<fingerprint>/bm25/` (bm25s files +
a position-aligned `chunks.pkl` + a versioned `meta.pkl`).

**Cache load note:** Loading from cache still instantiates the
embedding model (needed at query time for `embed_query()`). For
large models this takes several seconds. The BM25 index loads with
`mmap=True` to keep memory overhead minimal. These are known costs,
not bugs.

### QueryPipeline (`pipeline/query.py`)

Linear RAG: transform → retrieve → rerank → format → generate.

`from_config(config, index_artifact, retrieval_only)` builds each
component from the registry. Wiring is per-retriever-type:

- **`dense`** retrievers receive `index_artifact.vectorstore` and
  `index_artifact.embedder`.
- **`bm25`** retrievers receive `index_artifact.auxiliary_stores["bm25"]`
  via `set_sparse_index()`. A missing sparse index here raises at
  pipeline-construction time — caught earlier by `validate_config`
  for typo'd configs, but the runtime guard is the source of truth.
- **`hybrid`** retrievers construct each declared sub-retriever from
  `params["retrievers"]`, recursively apply the dispatch above to
  each child, then assemble a `HybridRetriever` with the fusion
  configuration. Hybrid-level filters propagate to every child.

`run(query, history)` executes the pipeline. All top_k values and
filters come from config — `run()` takes no override parameters. The
transformer returns `list[TransformedQuery]`, which the pipeline hands
to `retriever.retrieve_multi(transformed, top_k_retrieve,
fusion=qt_fusion)`. The retriever owns all fan-out and fusion: the
pipeline no longer deduplicates or merges results itself. The
`qt_fusion` value is read from `query.query_transform.fusion` at
construction. Cross-retriever score merging (dense vs. BM25, not
comparable) remains the hybrid retriever's job via `core.fusion`.

### AgentPipeline (`pipeline/agent.py`)

Stub. Defines the class interface (`__init__`, `run`, `query`)
and raises `NotImplementedError`. Will support single-agent (ReAct)
and multi-agent (supervisor-routed) modes in Phase D.

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

`evaluate(pipeline, experiment_name)` runs the pipeline against
every question in the dataset, `num_runs` times. Failed queries
are logged and skipped (one bad response doesn't crash the run).
Returns an `ExperimentResult` with aggregated and per-run metrics.

**RAGAS integration:** Uses the RAGAS 0.4.x API
(`EvaluationDataset`, `SingleTurnSample`, private metric
submodules to avoid deprecation warnings). Metrics are built once
and cached via `@functools.cache`.

**Evaluator embedder:** The evaluator builds its own embedder from
`evaluator_embedding` config via the component registry,
independent of the pipeline. This ensures all experiments are
measured with the same embedding model regardless of which
pipeline embedder is used — eliminating a confounding variable
in cross-experiment comparisons. The embedder is adapted to
RAGAS's `BaseRagasEmbedding` interface via
`_wrap_embedder_for_ragas()`.

**Evaluator LLM:** RAGAS needs an LLM judge for most metrics
(faithfulness, answer_correctness, context_precision, etc.).
Configured via `evaluator_llm` in the config. Supports Ollama
(local), EdenAI (cloud), and Google (cloud) providers. Built
through `_build_evaluator_llm()` which returns a
`LangchainLLMWrapper`.

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
experimental variables at a glance — not the full config (which
lives in `config.yaml`). Snapshot fields: `name`, `description`,
`pipeline_mode`, `evaluation_mode`, `index_fingerprint`,
`chunking_type`, `chunk_size`, `chunk_overlap`, `embedding_type`,
`embedding_model`, `vectorstore_type`, `query_transform_type`,
`query_transform_fusion`, `query_transform_num_queries`,
`retrieval_type`, `top_k_retrieve`, `top_k_final`,
`reranking_type`, `generation_type`, `generation_model`,
`evaluator_embedding_type`.

### Comparison (`evaluation/comparison.py`)

`compare_experiments(result_dirs)` loads summaries and produces
comparison rows. `format_comparison_table(rows)` renders a Markdown
table of metrics across experiments for side-by-side comparison.

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
    ├── Evaluator.evaluate(rag)
    │   ├── build evaluator embedder from config (via registry)
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
