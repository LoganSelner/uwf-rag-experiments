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

**Typed config end to end.** Config is a tree of **Pydantic v2 models**
(`ExperimentConfig` and friends in `core/config`). Each component
declares a nested `Params(ComponentParams)` model and validates its raw
config dict into `self.p` exactly once at construction, so a component's
defaults and constraints live in one typed place — not scattered across
`self.config.get(key, default)` reads. Required dependencies (a
retriever's vectorstore, a transformer's generator) arrive through the
constructor, so an object is never half-built.

**Dependency Rule.** Dependencies point inward. The entry point
scripts import from pipeline and evaluation. Pipeline imports from
components and core. Evaluation imports from core. Components import
from core. Core imports from nothing internal. No layer reaches
upward.

**Interface-first components.** Each pipeline stage has an abstract
base class defining what goes in and what comes out. Implementations
are registered by name and resolved from config at runtime.
Swapping a component never requires changing pipeline code.

**One construction contract.** The registry resolves a *class* by name. A
dependency-free stage is then constructed directly (`cls(config=params)`) — it
needs nothing but its params. A stage that needs runtime dependencies goes
through one seam: `BuildContext` (`components/build.py`) + a polymorphic
`build(...)` classmethod. The component pulls what it needs from the context
inside its own `build` (a retriever pulls the index; a tool pulls the query
stack), so the pipeline calls `registry.get(category, type).build(...)` and
**never branches on a concrete implementation name** — the previous
`if retrieval_type == "bm25"` dispatch is gone. Components that compose others
(the hybrid retriever; the planned multi-agent supervisor) recurse through the
same seam. Validation is symmetric: a `ValidateContext` carries the cross-config
values a component needs to check its own params.

**Separated indexing and querying.** The index (chunking + embedding + vectorstore)
is built once and cached. Query-side experiments
(retrieval strategy, reranking, prompt, LLM) reuse the cached
index. A SHA-256 fingerprint of the indexing config determines
cache identity.

---

## Layers

```
┌─────────────────────────────────────────────┐
│  scripts/ (thin CLIs)  +  ragbench.experiment│  Application boundary
│    run_experiment.py   run_matrix.py        │  experiment.py orchestrates:
│    compare.py                               │  load → validate → build →
│                                             │  evaluate → save
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
│    generators/      │          │
│    ...              │          │
└────────┬────────────┘          │
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────┐
│  core/                                      │  Foundation
│    types.py    config/     registry.py     │  Shared types, config,
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
  experiment orchestration (load config → validate → build pipeline →
  evaluate → save) lives in `ragbench.experiment`
  (`run_single_experiment` / `run_matrix`); the `scripts/` are thin CLI
  wrappers over it. `experiment.py` sits above both `pipeline/` and
  `evaluation/`, so the rule "`pipeline` never imports `evaluation`"
  still holds.

- `ragbench/components/__init__.py` triggers registration side effects
  (`@registry.register` decorators). This import happens only at
  application entry points: `scripts/run_experiment.py`,
  `scripts/run_matrix.py`, and `tests/conftest.py`
  (`import ragbench.components`). Pipeline modules
  never trigger it.

All source lives under a single installed package, `src/ragbench/`
(`uv pip install -e .`), so modules import as `ragbench.core`,
`ragbench.components`, etc. — no `sys.path` manipulation at the entry
points.

---

## File Map

```
src/ragbench/
├── core/
│   ├── types.py          Data contracts between pipeline stages (Message TypedDict, dataclasses)
│   ├── config/           Config package (re-exported from its __init__):
│   │   ├── models.py       Pydantic models + ValidateContext + index fingerprint
│   │   ├── loading.py      YAML loading + extends-inheritance deep-merge
│   │   ├── validation.py   validate_config + per-section checks
│   │   └── errors.py       ConfigValidationError
│   ├── registry.py       Component registry (@register / get)
│   ├── fusion.py         Rank-list fusion primitives (RRF + weighted + max-dedup)
│   └── git.py            Git SHA + dirty flag for reproducibility
│
├── components/
│   ├── base.py           Abstract base classes + ComponentParams (read-only contract)
│   ├── build.py          BuildContext + IndexHandle (construction seam; BM25_AUX_KEY)
│   ├── defaults.py       Passthrough/no-op defaults (always available)
│   ├── ingestors.py      PDFIngestor (PyMuPDF)
│   ├── chunkers.py       LangChainRecursiveChunker, CustomRecursiveChunker, SemanticChunker
│   ├── enrichers.py      ContextualChunkEnricher (chunk_enricher) — sets index_text pre-embedding
│   ├── embedders.py      HuggingFaceEmbedder, GoogleEmbedder, OpenAIEmbedder, EdenAIEmbedder
│   ├── vectorstores.py   FAISSVectorStore, ChromaVectorStore
│   ├── lexical_indexes.py   BM25LexicalIndex (bm25s + PyStemmer)
│   ├── retrievers.py     DenseRetriever, BM25Retriever, HybridRetriever
│   ├── generators/      One module per provider (ollama/edenai/google/openai) + base.py (_SDKGenerator + build_generator)
│   ├── prompts.py        ChatPromptTemplate
│   ├── query_transforms.py  ContextualizerQueryTransformer, HyDEQueryTransformer, MultiQueryQueryTransformer
│   ├── rerankers.py      CrossEncoderReranker
│   ├── tools.py          RAGSearchTool (tool/rag) + tool_to_spec
│   ├── _tool_protocol.py Pure helpers for the native tool-calling schema
│   └── __init__.py       Imports all implementation files → triggers registration
│
├── pipeline/
│   ├── indexing.py        IndexArtifact + IndexingPipeline (vector + auxiliary stores)
│   ├── query.py           QueryPipeline (linear RAG)
│   ├── agent.py           AgentPipeline (single-agent ReAct, native tool calling)
│   └── rag.py             RAGPipeline (top-level dispatcher)
│
├── evaluation/
│   ├── evaluator.py       RAGAS integration + multi-run execution
│   ├── _ragas_adapter.py  Anti-corruption layer: the only module importing ragas
│   ├── results.py         Experiment result serialization
│   └── comparison.py      Cross-experiment tables + config diffing
│
└── experiment.py         Orchestration: run_single_experiment / run_matrix

configs/
├── base.yaml              Default baseline (all experiments inherit)
└── experiments/            Per-experiment overrides (inherit from base)

scripts/
├── run_experiment.py      CLI: run one experiment (wraps run_single_experiment)
├── run_matrix.py          CLI: run + compare a matrix (wraps run_matrix)
└── compare.py             CLI entry point for comparing results

tests/
├── conftest.py            Fixtures + registration trigger
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

All types live in `src/ragbench/core/types.py`.

| Type | Produced By | Consumed By |
|------|------------|-------------|
| `Message` (TypedDict) | Pipeline / agent loop / transformers | BaseGenerator (`generate`) |
| `Document` | Ingestor | Chunker, ChunkEnricher |
| `Chunk` | Chunker | ChunkEnricher, Embedder, LexicalIndex |
| `EmbeddedChunk` | Embedder | VectorStore |
| `TransformedQuery` | QueryTransformer | Retriever (`retrieve_multi`) |
| `RetrievedChunk` | VectorStore / Retriever | Reranker, PromptTemplate |
| `GenerationResult` | Pipeline (assembled) | Evaluator |
| `ToolSpec` | AgentPipeline (from tools) | BaseGenerator (`generate(tools=…)`) |
| `ToolCall` | BaseGenerator | AgentPipeline (loop) |
| `ToolResult` | BaseTool | AgentPipeline |
| `AgentStep` | AgentPipeline | Results / logging (metadata) |
| `EvalSample` | Evaluator | RAGAS |
| `ScoredSample` | Evaluator | Result files (JSONL) |
| `ExperimentResult` | Evaluator | `results.py` serialization |

`Chunk` carries the canonical `content` plus an optional, build-time-only
`index_text` (with a `text_for_index` accessor that falls back to `content`).
A chunk enricher may set `index_text` so the embedder and lexical index use
the enriched text, while the vector store still persists `content` — keeping
the enrichment's effect isolated to retrieval. `index_text` is never persisted,
so retrieved chunks always carry `index_text=None`.

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

`Message` is the neutral, OpenAI-style chat message exchanged between the
pipeline/agent loop and every generator. It is a `TypedDict` (`role`
required; `content` / `tool_calls` / `tool_call_id` optional) — a plain
`dict` at runtime, so the OpenAI generator forwards messages verbatim while
the Ollama/Google/EdenAI translators get static shape-checking. Conversation
history and prompt-template output are `list[Message]` too.

`Queryable` is a `@runtime_checkable Protocol` in `core/types.py`:

```python
class Queryable(Protocol):
    def query(self, question: str) -> GenerationResult: ...
```

`RAGPipeline`, `QueryPipeline`, and `AgentPipeline` all satisfy it, as do
test mocks. The evaluator depends on this protocol, never on concrete
pipeline classes.

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
│   ├── query_transform: QueryTransformConfig  type + fusion + generator + params
│   ├── retrieval: RetrievalConfig     top_k_retrieve, top_k_final, filters
│   ├── reranking: ComponentConfig
│   ├── generator: LLMConfig           provider (= registry name), model, temperature, params
│   └── prompt: PromptConfig           system_template, CoT, citation style
├── AgentConfig                        (agent pipeline — single-agent ReAct)
│   ├── mode, max_iterations, system_prompt, llm, memory
│   └── tools: list[AgentDefinitionConfig]    roster the single agent reasons over
│       (mode: "multi" + the supervisor/agent-roster config arrive with Phase D2)
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
| `LLMConfig` | Reused by the answer generator (`query.generator`), query-transformer/enricher generators, the agent reasoning LLM, and the evaluator judge. `provider` doubles as the `generation` registry name; `params` carries provider extras (EdenAI `sub_provider`, Ollama `base_url`). One factory — `build_generator(LLMConfig)` — is the single construction path. |
| `ComponentConfig` | For stages where params are implementation-specific and vary widely (chunkers, embedders, vectorstores). `type` is the registry name; `params` is validated into the component's nested `Params` model at construction. |
| `EvalRunConfig` | `timeout`, `max_retries`, `max_wait`, `max_workers` for the RAGAS evaluator. Differs between local and cloud setups. |

The former redundant `query.generation` (a `ComponentConfig` whose `type` named
the provider) + `query.generation_llm` (an `LLMConfig`) pair is collapsed into a
single `query.generator: LLMConfig`. `query.query_transform.generator` (and the
contextual enricher's `params.generator`) describe their reasoning LLM the same
way.

### Inheritance

Experiment YAML files declare `extends: ../base.yaml`. Resolution
uses `load_yaml_with_inheritance()`:

- Nested dicts: merge recursively (override only specified keys)
- Lists: full replacement (child provides the entire list)
- Scalars: child replaces parent
- **Implementation switch replaces params.** When a child changes a
  component's implementation selector — `type` (for `ComponentConfig`) or
  `provider` (for `LLMConfig`) — the whole subtree is *replaced*, not merged,
  because a component's `params` are private to its implementation. This is
  why switching `indexing.embedding.type` from `edenai` to `huggingface` (or a
  generator's `provider` from `edenai` to `ollama`) does not bleed the parent's
  `provider` / `sub_provider` onto the new implementation. Mirrors Hydra's
  config-group selection.

The `extends` path is relative to the child file's directory.
Multi-level inheritance is supported (child → parent → grandparent).

### Index Fingerprinting

`ExperimentConfig.index_fingerprint()` returns a 12-char hex SHA-256
of: sources (config **and file content**) + chunking + embedding +
vectorstore + sparse_index + chunk_enricher (the last two only when
set) config.

**Included:** Source names, paths, ingest types/params, **and a
content digest of each source file** (sha256 of its bytes, with a
`(size, mtime)` fallback) — so editing a document in place changes the
fingerprint and rebuilds, rather than silently serving a stale cache.
Chunking, embedding, vectorstore types and all params. Sparse-index
type and params, *only when its `type` is non-empty* — empty
`ComponentConfig`s canonicalize to absence, so adding an unused
optional component doesn't disturb the fingerprint of existing
experiments.

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
construction (called in `ragbench.experiment.run_single_experiment`).
It checks:

- Every component type referenced in config is registered
- `query.query_transform.fusion` is one of `{rrf, max}`
- Transformer `branch` / `original_branch` hints reference a real
  hybrid child name (static check; only when retrieval is `hybrid`).
  The runtime guard in `HybridRetriever.retrieve_multi` is authoritative
- `top_k_final <= top_k_retrieve` and both are positive
- `pipeline_mode: "agent"` has at least one tool defined
- `evaluation.mode` is one of `{full, retrieval_only, none}`
- Modes other than `retrieval_only` require a prompt type and a
  `query.generator` with both `provider` and `model_name`
- `indexing.sources` is non-empty
- `EvalRunConfig` values are positive (timeout, workers, wait)
- `evaluator_llm.provider` is one of `{ollama, edenai, google, openai}` (or empty)
- `evaluator_embedding.type` is registered in the `embedding` category
  (required for modes other than `none`)
- Evaluation dataset file exists on disk

**Component-owned param checks.** Retrieval-type-specific rules (a `bm25`
retriever needs `indexing.sparse_index`; a `hybrid` retriever's child roster,
nesting ban, fusion mode, rrf_k, and weights) are *not* hard-coded in
`core/config`. `validate_config` resolves the retriever class from the
registry and calls its `validate_params(params, ctx)` classmethod — where `ctx`
is a `ValidateContext` carrying the registry plus the cross-config values
(`sparse_index_type`, `top_k_retrieve`) — folding the returned strings into the
same error list. Component knowledge stays in the component; core only supplies
the context.

All errors are collected into a list and raised as a single
`ConfigValidationError` with a readable bullet-list message. This
catches typos and constraint violations before any heavyweight
work (model loading, embedding) begins. A separate, lighter guard
(`tests/test_config.py::TestShippedConfigsValidate`) validates every shipped
config's component `params` against each component's `Params` model.

---

## Component Interfaces

All abstract base classes (and `ComponentParams`) live in
`src/ragbench/components/base.py`.

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
| `BaseGenerator` | `generation` | `generate(prompt, tools=None)` | `str\|list[Message] → GenerationResult` (optional `tools` → tool-calling) |
| `BasePromptTemplate` | `prompts` | `format(query, chunks, history)` | `→ str\|list[Message]` |
| `BaseTool` | `tool` | `execute(query)` | `str → ToolResult` |
| `BaseMemory` | `memory` | `add_turn()` / `get_history()` / `clear()` | Turn management |

### Key Interface Decisions

**BaseGenerator takes a formatted prompt, not raw chunks.** The
pipeline calls `PromptTemplate.format()` first. The generator is a
pure LLM wrapper that knows nothing about retrieval context. This
is what makes prompt config changes actually affect output. Every
generator is built through one factory, `build_generator(LLMConfig)`
in `components/generators/` (its `base.py`), used by the linear pipeline, the agent's
reasoning LLM, and the query transformers / contextual enricher — there
is a single construction path, not a per-call-site idiom.

**Components validate their own params.** Each concrete component declares a
nested `class Params(ComponentParams)` (a Pydantic model) and, in `__init__`,
runs `self.p = self.Params.model_validate(self.config)` once. Defaults and
field constraints (`Field(ge=1)`, etc.) live only in `Params`. The registry
still constructs components as `cls(config=params_dict)` — the dict→typed step
happens inside the component, so resolve-by-name stays uniform while reads
become typed (`self.p.chunk_size`, not `self.config.get("chunk_size", 512)`).

**Tool calling is an additive, optional capability.** `generate` grew an
optional `tools: list[ToolSpec] | None` parameter; when omitted (the linear
pipeline's only call) behavior is byte-for-byte unchanged and the result's
`tool_calls` / `finish_reason` stay empty. When provided, the model may return
`tool_calls` instead of an answer. The agent loop and generators exchange a
**neutral OpenAI-style message schema** — assistant turns may carry
`tool_calls`; a `tool`-role turn carries `tool_call_id` + `content`
(`ToolCall.arguments` is a parsed dict; the JSON-string↔dict and per-provider
shape translation lives inside each generator and the pure helpers in
`components/_tool_protocol.py`). All four providers implement it (EdenAI via
`ChatEdenAI.bind_tools`; Gemini keys tool results by function name; Ollama/
Gemini omit ids, which the loop synthesizes).

**BaseRetriever builds itself from the context; dependencies arrive through
the constructor.** Each retriever implements `build(*, params, default_filters,
ctx)`: `DenseRetriever` pulls `ctx.index.vectorstore` + `embedder`,
`BM25Retriever` pulls `ctx.index.auxiliary_stores["bm25"]`, and
`HybridRetriever` recurses through `ctx.registry` to build its children. The
pipeline calls `registry.get("retrieval", type).build(...)` — it does **not**
branch on the retriever type. Each is constructed fully-formed (no
post-construction `set_*` wiring), so a retriever is never half-built. Default
filters (`query.retrieval.filters`) are injected too, and the hybrid passes
them to every child at retrieve time, keeping per-branch over-retrieve + filter
semantics consistent with the single-retriever paths. Retrieval-type-specific
config validation lives on each retriever as a `validate_params(params, ctx)`
classmethod (see Validation), not in core.

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
| `chunking` | `recursive` | `LangChainRecursiveChunker` | `chunkers.py` |
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
| `generation` | `ollama` | `OllamaGenerator` | `generators/ollama.py` |
| `generation` | `edenai` | `EdenAIGenerator` | `generators/edenai.py` |
| `generation` | `google` | `GoogleGenerator` | `generators/google.py` |
| `generation` | `openai` | `OpenAIGenerator` | `generators/openai.py` |
| `prompts` | `chat` | `ChatPromptTemplate` | `prompts.py` |
| `memory` | `none` | `NoMemory` | `defaults.py` |
| `memory` | `buffer_window` | `BufferWindowMemory` | `defaults.py` |
| `tool` | `rag` | `RAGSearchTool` | `tools.py` |
| `tool` | `web_search` | `WebSearchTool` | `tools.py` |

### Reserved / not-yet-active

A couple of config surfaces are wired and honest stubs, but inert until a later
phase — kept (rather than removed) because the seam is cheap and the harness
values stable config keys:

- **Agent memory** (`agent.memory`, `BufferWindowMemory`): registered and
  constructed, but the single-turn ReAct loop does not thread history yet.
  Activated by Phase E multi-turn evaluation.
- **`query.prompt.max_context_tokens`**: carried through to the prompt template
  but not enforced (no context-budget trimming today). A research harness
  prefers a precise, opt-in token budget over an approximate one, so enforcement
  is deferred rather than shipped as a char-count guess.

---

## Pipeline Orchestration

### IndexingPipeline (`pipeline/indexing.py`)

Builds the vector index — and any configured auxiliary indexes —
from source documents.

`from_config(config)` constructs per-source ingestors plus shared
chunker, embedder, vectorstore, and (when set) a sparse index
(`indexing.sparse_index.type`) and a chunk enricher
(`indexing.chunk_enricher.type`), all from the registry.

`build()` iterates sources (ingest → tag source_name), chunks all
documents, optionally **enriches** them (the chunk enricher sets each
chunk's `index_text` — e.g. contextual retrieval), then runs **sparse
indexing before embedding** so a misconfigured BM25 tokenizer/stemmer
fails fast — before spending embedding API budget. Enrichment runs
before both the sparse index and the embedder, since both consume
`Chunk.text_for_index`. After embedding, the vector store is populated.
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

`from_config(config, index_artifact, retrieval_only)` assembles a
`BuildContext` (the index, this query stack, the generator factory, the
registry) and builds each component through it. The retriever is built by
`registry.get("retrieval", config.retrieval.type).build(params=…,
default_filters=…, ctx=ctx)` — the pipeline never branches on the retriever
type:

- **`dense`** retrievers pull `ctx.index.vectorstore` + `ctx.index.embedder`.
- **`bm25`** retrievers pull `ctx.index.auxiliary_stores["bm25"]`. A missing
  sparse index raises at build time — caught earlier by `validate_config` for
  typo'd configs, but the runtime guard is the source of truth.
- **`hybrid`** retrievers build each declared sub-retriever by recursing
  through `ctx.registry` (the same `build` path), then assemble a
  `HybridRetriever` with the fusion configuration. Hybrid-level filters are
  injected into every child.

The reasoning LLM for an LLM-backed query transformer is built here via
`ctx.make_generator(query.query_transform.generator)` and injected into the
transformer; the answer generator is `build_generator(query.generator)`.

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

Single-agent ReAct (Phase D1). One reasoning LLM drives a
reason→act(tool)→observe loop via **native tool calling**:
`from_config(config, index_artifact)` assembles a `BuildContext` (the index,
`config.query` as the query stack, the generator factory, the registry), then
builds the reasoning generator (`ctx.make_generator(agent.llm)`), the tool
roster (each `agent.tools` entry → `registry.get("tool", …).build(entry,
ctx)`), and the loop budget (`agent.max_iterations`, `top_k_final`). That same
context is the recursion point multi-agent (D2) will build its sub-agents on.

`run(query)` seeds `[system?, user]` and loops up to `max_iterations`:
`generate(messages, tools=specs)`; if the result has `tool_calls`, each is
executed, its result appended as a `tool`-role message, and any
`retrieved_chunks` accumulated; otherwise the result's text is the final
answer. If the budget is exhausted, a final answer is forced with
`tools=None`. The accumulated chunks are unioned (dedup by `chunk_id`, keep
max score, cap at `top_k_final`) into the returned `GenerationResult`, so the
evaluator scores retrieval in agent mode on the same metrics as linear; an
`AgentStep` trace + counters land in `metadata`. A missing tool or a tool
failure becomes a recoverable observation, not a crash. `query()` delegates to
`run()`, satisfying `Queryable`. The loop is monolithic by design (one
mechanism today); multi-agent supervisor mode is Phase D2.

### RAGPipeline (`pipeline/rag.py`)

Top-level dispatcher. `from_config(config, no_cache)` builds the
index via `IndexingPipeline.run_or_load_cache()`, then constructs
either a `QueryPipeline` (linear mode) or an `AgentPipeline` (agent mode).

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

**RAGAS integration:** All `ragas` use is quarantined in
`evaluation/_ragas_adapter.py` (see *RAGAS is quarantined* below) —
the legacy `evaluate()` API (`EvaluationDataset`, `SingleTurnSample`,
private metric submodules to avoid deprecation warnings; metrics
built once via `@functools.cache`). The evaluator calls the adapter,
never `ragas` directly.

**Evaluator embedder:** The evaluator builds its own embedder from
`evaluator_embedding` config via the component registry,
independent of the pipeline. This ensures all experiments are
measured with the same embedding model regardless of which
pipeline embedder is used — eliminating a confounding variable
in cross-experiment comparisons. The embedder is adapted to
RAGAS's `BaseRagasEmbedding` interface inside the ragas adapter
(`evaluation/_ragas_adapter.wrap_embedder`).

**Evaluator LLM:** RAGAS needs an LLM judge for most metrics
(faithfulness, answer_correctness, context_precision, etc.).
Configured via `evaluator_llm`; supports Ollama (local), EdenAI,
Google, and OpenAI. `_build_evaluator_llm()` builds a per-provider
**LangChain** chat model and hands it to the ragas adapter's
`wrap_langchain_llm`.

**RAGAS is quarantined.** Every `ragas` import — including the
deprecated private `ragas.metrics._*` classes — lives in one module,
`evaluation/_ragas_adapter.py`; the evaluator imports only that. The
harness stays on the legacy `evaluate()` path (pinned `ragas<0.5`)
because ragas's non-deprecated `collections` metrics require an
`InstructorBaseRagasLLM` that can't wrap EdenAI, whereas
`evaluate(llm=LangchainLLMWrapper(...))` can. A ragas bump touches that
one file.

**Two LLM construction paths (deliberate).** The pipeline's answer /
reasoning generators are built by `build_generator(LLMConfig)` →
provider **SDK clients** (`BaseGenerator`), while the RAGAS judge is a
**LangChain** chat model wrapped for ragas. They are separate by
necessity: ragas only accepts a LangChain-wrapped (or its own
structured) LLM, and EdenAI — the harness's gateway provider — exists
only as `langchain-community`'s `ChatEdenAI`. So the model named in
`query.generator` vs `evaluation.evaluator_llm` runs through different
client libraries; the upside is EdenAI works on both sides.

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

### Methodology & Caveats

Two properties of the harness shape how results should be read:

- **The cross-run std is mostly judge variance, not pipeline sampling.**
  Generation defaults to `temperature=0`, so for a deterministic provider the
  `num_runs` repeats produce near-identical pipeline outputs; the reported
  `*_std` then largely reflects the RAGAS judge LLM's own nondeterminism (plus
  any provider-side variation), not answer-sampling spread. Read it as a
  measurement-stability band, not a model-variability estimate.

- **Agent retrieval metrics score a re-capped chunk set.** In agent mode the
  loop unions the chunks retrieved across all tool calls, dedups by `chunk_id`,
  and caps at `top_k_final` before handing them to the evaluator
  (`AgentPipeline._aggregate_chunks`). That keeps the context-precision/recall
  denominator identical to the linear pipeline — but it can differ from the
  superset of chunks the agent's own LLM actually saw across iterations. So a
  linear-vs-agent retrieval-metric delta isolates control flow on a common
  budget; it is not a claim about how much context the agent reasoned over.

### Results (`evaluation/results.py`)

`save_experiment(config, result, base_dir, git_info)` writes:

```
results/<name>/
  summary.json    Aggregated metrics + config snapshot + git SHA/dirty
  config.yaml     Full resolved config (config.model_dump() → YAML)
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
class (not an instance). A dependency-free stage is constructed directly
(`cls(config=params)`); a stage needing runtime dependencies is built through
its `build(cfg, ctx)` classmethod with a `BuildContext` (see *One construction
contract* under Design Principles). Either way the pipeline resolves by name and
never branches on a concrete implementation.

**Introspection:** `registry.list_category("chunking")` returns
registered names. `registry.is_registered(...)` checks existence.

**Testing:** `registry.clear()` enables isolated tests with a
fresh `ComponentRegistry()` instance.

---

## Experiment Lifecycle

```
scripts/run_experiment.py  (thin CLI)
    │
    ├── import ragbench.components  (triggers registration)
    ├── configure_runtime()         (load .env, GPU perf)
    └── ragbench.experiment.run_single_experiment()
        ├── ExperimentConfig.from_yaml (loads + resolves inheritance)
        ├── validate_config            (checks types, constraints, files)
        ├── capture_git_info           (git sha + dirty, for reproducibility)
        │
        ├── RAGPipeline.from_config
        │   ├── IndexingPipeline.run_or_load_cache
        │   │   ├── check fingerprint cache → load or build
        │   │   └── return IndexArtifact
        │   └── QueryPipeline.from_config
        │       └── build retriever(vectorstore + embedder) + build_generator(...)
        │
        ├── Evaluator.evaluate(rag)
        │   ├── build evaluator embedder from config (via registry)
        │   ├── for each run:
        │   │   ├── for each question: rag.query() → EvalSample
        │   │   └── RAGAS evaluate (via _ragas_adapter) → per-sample scores
        │   └── aggregate metrics (mean ± std)
        │
        └── save_experiment
            └── write summary.json + config.yaml + run_N.jsonl

scripts/run_matrix.py wraps ragbench.experiment.run_matrix() — the same
run_single_experiment over many configs (shared git snapshot, failures skipped),
then an optional comparison table.
```

---

## Adding a New Component

1. Write the class in the appropriate `src/ragbench/components/*.py` file
2. Inherit from the base class, implement required methods
3. Declare a nested `class Params(ComponentParams)` for its config and read
   `self.p` (validate it in `__init__` with `self.Params.model_validate(self.config)`)
4. If the component needs runtime dependencies (the index, a sub-LLM, the query
   stack), override `build(...)` to pull them from the `BuildContext` instead of
   relying on the default `cls(config=params)`. Retrievers and tools must;
   most stages don't.
5. Decorate with `@registry.register("category", "name")`
6. Ensure the file is imported in `src/ragbench/components/__init__.py`
7. Add an inventory assertion in `tests/test_registry.py`
8. Reference `type: "name"` in a YAML experiment config
9. Run `make qa` to verify

No pipeline code changes. No config system changes. No evaluation
changes.
