# ARGObot Bench — Architecture Plan

> **Status:** Architecture design complete. Ready for implementation.
> **Last updated:** 2026-03-14
> **Authors:** Research team, Dr. Cohen's lab, UWF Department of Computer Science

This document is the definitive reference for the argobot-bench
experimentation framework. Every architectural decision, every
interface contract, every known limitation is documented here.
Read this before writing any implementation code.

---

## 1. Purpose

This framework exists to answer one question: **which combination of
RAG components produces the best academic advising chatbot?**

Specifically, it lets us:

- Change one variable at a time (chunking strategy, embedding model,
  retrieval method, etc.) and measure the effect on RAGAS metrics
- Compare linear RAG (v1), single-agent (v2), and multi-agent (v3)
  architectures with identical underlying components
- Cache indexes so query-side experiments don't re-embed documents
- Run multiple evaluation passes for statistical significance
- Save full configs alongside results for reproducibility

What this framework is NOT:

- Not a production chatbot (no web UI, no live deployment)
- Not a LangChain/LlamaIndex wrapper (we build from interfaces,
  implementations may optionally use those libraries internally)
- Not an agent framework (the agent pipeline is one mode of
  operation, not the core identity of the framework)

---

## 2. Core Design Principles

**Config-driven everything.** Every testable variable is a YAML
parameter. If you can't change it from a config file, it's not
testable. No experiment should require editing Python code.

**One source of truth per parameter.** top_k is defined in
RetrievalConfig and nowhere else. The LLM model is in LLMConfig.
The prompt text is in PromptConfig. The pipeline reads these;
nothing overrides them at runtime.

**Interface-first components.** Every pipeline stage has an abstract
base class (ABC) that defines exactly what goes in and what comes
out. Implementations are swappable because they honor the same
contract. The registry maps config names to classes.

**Separated indexing and query.** The index (chunking + embedding +
vectorstore) is built once and cached. Query-side experiments
(retrieval strategy, reranking, prompt, agent architecture) reuse
the cached index. The `index_fingerprint` determines cache validity.

**Pipeline-agnostic evaluation.** The evaluator doesn't know whether
it's testing a linear pipeline or an agent pipeline. Both produce
`GenerationResult` with the same fields. RAGAS metrics work the same.

---

## 3. File Structure

```
argobot-bench/
│
├── ARCHITECTURE_PLAN.md        ← This document (you are here)
├── pyproject.toml              ← Dependencies and project config
├── .gitignore
│
├── configs/
│   ├── base.yaml               ← Default baseline — all experiments inherit from this
│   └── experiments/
│       ├── chunking_*.yaml     ← Chunking strategy/size experiments
│       ├── embedding_*.yaml    ← Embedding model experiments
│       ├── retrieval_*.yaml    ← Retrieval method/top_k experiments
│       ├── reranking_*.yaml    ← Reranking method experiments
│       ├── prompt_*.yaml       ← Prompt template experiments
│       ├── agent_*.yaml        ← Agent architecture experiments
│       └── retrieval_only_*.yaml  ← Fast retriever-only evaluation
│
├── data/
│   ├── sources/                ← Source documents (handbook PDF, confluence HTML)
│   │   └── .gitkeep
│   ├── datasets/               ← Evaluation datasets (JSON)
│   │   └── 18q_handbook.json   ← Example: 18-question handbook dataset
│   └── indexes/                ← GITIGNORED — auto-generated cached indexes
│       └── .gitignore
│
├── results/                    ← GITIGNORED — auto-generated experiment outputs
│   └── .gitignore
│
├── src/
│   ├── core/                   ← Framework infrastructure (rarely changes)
│   │   ├── types.py            ← All shared data types
│   │   ├── config.py           ← Config system (dataclasses + YAML loading)
│   │   └── registry.py         ← Component registry (@register decorator)
│   │
│   ├── components/             ← Interfaces + implementations
│   │   ├── base.py             ← All abstract base classes (READ-ONLY CONTRACT)
│   │   ├── defaults.py         ← Passthrough/no-op + memory defaults
│   │   ├── ingestors.py        ← PDF, HTML, text ingestor implementations
│   │   ├── chunkers.py         ← Recursive, semantic, etc. implementations
│   │   ├── embedders.py        ← HuggingFace, OpenAI, etc. implementations
│   │   ├── vectorstores.py     ← FAISS, ChromaDB, etc. implementations
│   │   ├── retrievers.py       ← Dense, BM25, hybrid implementations
│   │   ├── rerankers.py        ← Cross-encoder, ColBERT, etc. implementations
│   │   ├── generators.py       ← Ollama, OpenAI, etc. implementations
│   │   ├── prompts.py          ← Chat, completion prompt template implementations
│   │   ├── tools.py            ← Web search, email tool implementations
│   │   └── __init__.py         ← Imports all implementation files to trigger registration
│   │
│   ├── pipeline/               ← Orchestration (how components are wired)
│   │   ├── indexing.py         ← Documents → chunks → embeddings → vectorstore
│   │   ├── query.py            ← Query → transform → retrieve → rerank → generate
│   │   ├── agent.py            ← Query → supervisor → agent selection → tools → response
│   │   └── rag.py              ← Top-level dispatcher + run_experiment()
│   │
│   └── evaluation/
│       ├── evaluator.py        ← RAGAS integration, multi-run execution, result saving
│       └── comparison.py       ← Cross-experiment comparison tables and analysis
│
├── scripts/
│   ├── run_experiment.py       ← CLI: python scripts/run_experiment.py configs/experiments/X.yaml
│   └── compare.py              ← CLI: python scripts/compare.py results/exp1 results/exp2 ...
│
└── tests/
    ├── test_config.py          ← Config parsing, inheritance, fingerprinting tests
    └── test_pipeline.py        ← Integration tests with mock components
```

### Why flat files, not subdirectories per component category

Each category (chunkers, embedders, etc.) will have 3-10
implementations, each typically 30-100 lines. Putting them all in
one file per category means:

- You see all implementations side by side when writing a new one
- No directory navigation overhead
- Copy-paste patterns are visible
- The __init__.py just imports the files, no nested packages

If any single file exceeds ~500 lines, split it. But that's
unlikely for research implementations.

### Adding a new implementation

1. Open the appropriate file (e.g., `src/components/chunkers.py`)
2. Import the base class and registry
3. Write your class, decorate with `@registry.register("chunking", "my_name")`
4. The `__init__.py` already imports the file — your class is auto-registered
5. Reference `type: "my_name"` in any YAML config

---

## 4. Data Flow

### 4.1 Linear Pipeline (pipeline_mode: "linear")

```
Source Files
    │
    ▼
[Ingestor] ─── one per source, tagged with source_name
    │
    ▼
list[Document]
    │
    ▼
[Chunker] ─── shared across all sources
    │
    ▼
list[Chunk] ─── each carries source_name metadata
    │
    ▼
[Embedder] ─── shared, produces vectors
    │
    ▼
list[EmbeddedChunk]
    │
    ▼
[VectorStore] ─── persisted, cached by index_fingerprint
    │
    ║  ← INDEX BOUNDARY — everything above is cached
    ║
    ▼
User Query
    │
    ▼
[QueryTransformer] ─── may produce multiple queries
    │
    ▼
list[str] (transformed queries)
    │
    ▼
[Retriever] ─── embeds query, searches vectorstore, applies filters
    │                results deduplicated across multiple queries
    ▼
list[RetrievedChunk] (top_k_retrieve candidates)
    │
    ▼
[Reranker] ─── re-scores, truncates to top_k_final
    │
    ▼
list[RetrievedChunk] (top_k_final context)
    │
    ▼
[PromptTemplate] ─── formats query + chunks + history into prompt
    │
    ▼
str | list[dict] (formatted prompt)
    │
    ▼
[Generator] ─── pure LLM call, returns answer + metadata
    │
    ▼
GenerationResult ─── pipeline attaches query + chunks to result
    │
    ▼
[Evaluator] ─── computes RAGAS metrics against ground truth
```

**Critical handoff: PromptTemplate → Generator.** The prompt
template formats; the generator generates. The generator never
sees raw chunks. This is what makes YAML prompt changes actually
affect the LLM output. The pipeline assembles the final
GenerationResult by attaching the query and chunks to the
generator's answer.

### 4.2 Agent Pipeline (pipeline_mode: "agent")

```
User Query
    │
    ▼
[Supervisor/Agent LLM] ─── decides which tool to call
    │
    ├──► [RAGTool: handbook_agent]
    │        └── mini QueryPipeline (filtered to handbook)
    │              transform → retrieve → rerank → format prompt → generate
    │
    ├──► [RAGTool: knowledge_base_agent]
    │        └── mini QueryPipeline (filtered to knowledge_base)
    │
    ├──► [ExternalTool: web_search]
    │        └── API call → ToolResult
    │
    └──► [ExternalTool: email_sender]
             └── API call → ToolResult
    │
    ▼
ToolResult(s) ─── observed by supervisor
    │
    ▼
[Supervisor LLM] ─── decides: call another tool, or respond
    │                  (loop up to max_iterations)
    ▼
GenerationResult ─── same type as linear pipeline
    │
    ▼
[Evaluator] ─── identical evaluation, same RAGAS metrics
```

**Key insight:** RAG-type agents internally use the same
QueryPipeline as the linear mode, including prompt template
formatting and generator calls. Every retrieval improvement
(reranking, hybrid search, etc.) that works in linear mode
automatically works inside agents.

---

## 5. Type System

All types live in `src/core/types.py`. These are the contracts
between pipeline stages.

| Type | Produced By | Consumed By | Key Fields |
|------|------------|-------------|------------|
| `Document` | Ingestor | Chunker | content, metadata (source_name) |
| `Chunk` | Chunker | Embedder | content, metadata, chunk_id |
| `EmbeddedChunk` | Embedder | VectorStore | chunk, embedding |
| `RetrievedChunk` | VectorStore/Retriever | Reranker, PromptTemplate | chunk, score, retrieval_method |
| `GenerationResult` | Pipeline (assembled) | Evaluator | query, answer, retrieved_chunks, metadata |
| `ToolResult` | BaseTool | AgentPipeline | tool_name, content, success, retrieved_chunks |
| `AgentStep` | AgentPipeline | Results/logging | step_number, action, agent_name, tool_name |
| `EvalSample` | Evaluator | RAGAS | query, response, retrieved_contexts, reference |
| `ExperimentResult` | Evaluator | Results files | metrics, per_sample_metrics, config_snapshot |

---

## 6. Config System

### 6.1 Structure

All config dataclasses live in `src/core/config.py`.

```
ExperimentConfig                    ← top-level
├── name, description
├── pipeline_mode                   ← "linear" or "agent"
├── IndexingConfig
│   ├── sources: list[SourceConfig] ← per-source ingest config
│   ├── chunking: ComponentConfig
│   ├── embedding: ComponentConfig
│   └── vectorstore: ComponentConfig
├── QueryConfig                     ← used when pipeline_mode="linear"
│   ├── query_transform: ComponentConfig
│   ├── retrieval: RetrievalConfig  ← dedicated type (top_k, filters)
│   ├── reranking: ComponentConfig
│   ├── generation: ComponentConfig
│   ├── generation_llm: LLMConfig   ← dedicated type (provider, model, temp)
│   └── prompt: PromptConfig        ← dedicated type (template, CoT, citation)
├── AgentConfig                     ← used when pipeline_mode="agent"
│   ├── mode                        ← "single" or "multi"
│   ├── llm: LLMConfig             ← the agent LLM (single mode)
│   ├── supervisor: SupervisorConfig
│   │   ├── llm: LLMConfig
│   │   ├── routing                 ← "llm" or "rule_based"
│   │   ├── routing_prompt
│   │   └── max_iterations
│   ├── memory: MemoryConfig
│   ├── agents: list[AgentDefinitionConfig]
│   │   ├── name, type ("rag" or "tool")
│   │   ├── retrieval: RetrievalConfig  ← per-agent filters, top_k
│   │   ├── prompt: PromptConfig        ← per-agent prompt
│   │   └── tool                        ← tool registry name (for type="tool")
│   └── tools: list[AgentDefinitionConfig]
└── EvaluationConfig
    ├── dataset
    ├── mode                        ← "full" or "retrieval_only"
    ├── metrics, retrieval_only_metrics
    ├── num_runs
    └── evaluator_llm: LLMConfig
```

### 6.2 Dedicated vs Generic Config Types

Some stages have dedicated config dataclasses; others use the
generic `ComponentConfig(type, params)`. Here's why:

| Config Type | Why Dedicated |
|------------|---------------|
| `RetrievalConfig` | top_k_retrieve, top_k_final, and filters are too important to bury in params. They're the most commonly changed retrieval variables. |
| `PromptConfig` | system_template, use_chain_of_thought, citation_style, few_shot_examples are all independently testable. A generic params dict would be opaque. |
| `LLMConfig` | Reused by 4+ config locations (generator, query transformer, reranker, evaluator, supervisor, agent). Needs a stable shared structure. |
| `MemoryConfig` | type + window_size are the key variables. Clean dedicated type. |
| `ComponentConfig` | Used for stages where the params are implementation-specific and vary widely (chunkers, embedders, vectorstores, rerankers, ingestors). The registry passes params as kwargs. |

### 6.3 LLM Config Convention

Components that need an LLM configure it in one of two ways:

**Dedicated field** (for the primary generator):
```yaml
query:
  generation:
    type: "ollama"               # Selects the backend class
  generation_llm:                # Separate LLMConfig
    provider: "ollama"
    model_name: "qwen2.5:32b"
    temperature: 0.0
```

**Nested in params** (for components where LLM is optional):
```yaml
query:
  query_transform:
    type: "hyde"
    params:
      llm:                       # LLMConfig nested inside params
        provider: "ollama"
        model_name: "qwen2.5:32b"
```

Implementations access the nested LLM config via
`self.config["llm"]` and construct an LLMConfig from it.

### 6.4 Config Inheritance

Experiment configs extend base.yaml via `extends: base.yaml`.
Deep merge rules:
- Nested dicts: merge recursively (override only specified keys)
- Lists: full replacement (override provides the entire list)
- Scalars: override replaces base

This means: to change one chunking param, you override just that
param. To change the sources list, you provide the full list.

### 6.5 Index Fingerprinting

`ExperimentConfig.index_fingerprint()` produces a SHA-256 hash
(first 12 chars) of: sources + chunking + embedding + vectorstore.

**Included:** Source names, paths, ingest configs. Chunking type
and all params. Embedding model and params. Vectorstore type and
params.

**Excluded:** Everything in QueryConfig, AgentConfig, and
EvaluationConfig. Pipeline mode.

**Consequence:** Linear, single-agent, and multi-agent experiments
with the same indexing config share a cached index. Verified:
all three produce fingerprint `e5371550c87a`.

**Cache invalidation caveat:** Changing source file content
(e.g., updating the handbook PDF) without changing any config
field does NOT invalidate the cache. Use `--no-cache` CLI flag
or manually delete `data/indexes/<fingerprint>/`.

---

## 7. Component Interfaces

All interfaces live in `src/components/base.py`. This file is the
architectural contract — read it before implementing anything.

### 7.1 Interface Summary

| Interface | Category | Primary Method | Input → Output |
|-----------|----------|---------------|----------------|
| `BaseIngestor` | `ingest` | `ingest(path)` | str → list[Document] |
| `BaseChunker` | `chunking` | `chunk(docs)` | list[Document] → list[Chunk] |
| `BaseEmbedder` | `embedding` | `embed_chunks(chunks)` / `embed_query(str)` | list[Chunk] → list[EmbeddedChunk] / str → list[float] |
| `BaseVectorStore` | `vectorstore` | `add(chunks)` / `search(vec, top_k, filters)` | Storage + retrieval |
| `BaseRetriever` | `retrieval` | `retrieve(query, top_k, filters)` | str → list[RetrievedChunk] |
| `BaseQueryTransformer` | `query_transform` | `transform(query, history)` | str → list[str] |
| `BaseReranker` | `reranking` | `rerank(query, chunks, top_k)` | list[RetrievedChunk] → list[RetrievedChunk] |
| `BaseGenerator` | `generation` | `generate(prompt)` | str\|list[dict] → GenerationResult |
| `BasePromptTemplate` | `prompts` | `format(query, chunks, history)` | Components → str\|list[dict] |
| `BaseTool` | `tool` | `execute(query)` | str → ToolResult |
| `BaseMemory` | `memory` | `add_turn()` / `get_history()` | Turn management |

### 7.2 Key Interface Decisions

**BaseGenerator takes a formatted prompt, not raw chunks.** The
pipeline calls `prompt_template.format()` first, then passes the
result to `generator.generate()`. The generator is a pure LLM
wrapper. This ensures YAML prompt changes actually affect output.

**BaseRetriever holds references to vectorstore and embedder.** These
are injected via `set_vectorstore()` / `set_embedder()` after
construction. This allows the pipeline to construct components
independently from config and wire them together afterward. The
retriever also carries default `filters` from config.

**BaseQueryTransformer returns a list.** Even single-query
transforms return `[query]`. This naturally supports HyDE (1 query),
multi-query (N queries), sub-question decomposition (N queries),
and passthrough (1 unchanged query) without interface branching.

**BaseVectorStore.search() takes explicit filters.** Not buried
in **kwargs. Implementations that don't support native filtering
(e.g., FAISS) must implement post-retrieval filtering internally:
over-retrieve, filter, return top_k.

**BaseMemory.get_history() returns windowed results.** The
implementation decides how much history to return based on its
config (window_size for BufferWindowMemory). Callers don't need to
know the windowing strategy.

---

## 8. Pipeline Orchestration

### 8.1 IndexingPipeline (src/pipeline/indexing.py)

Builds the vector index from source documents. Separated from the
query pipeline because the index can be cached and reused.

**Multi-source flow:**
1. For each source in config: create ingestor, ingest, tag chunks
   with source_name
2. All chunks flow through shared chunker → embedder → vectorstore
3. Return IndexArtifact (vectorstore + embedder + stats)

**IndexArtifact** is the handoff between indexing and query
pipelines. It carries the populated vectorstore and the embedder
(needed at query time to embed queries).

**Cache logic:** `run_or_load_cache()` checks if
`data/indexes/<fingerprint>/` exists. If yes, loads the vectorstore
from disk. If no, builds fresh and saves.

### 8.2 QueryPipeline (src/pipeline/query.py)

Processes queries through the linear RAG pipeline. All behavior
is determined by config — no runtime parameter overrides.

**Construction:** `from_config()` builds each component from the
registry, wires the retriever to the vectorstore/embedder from
the IndexArtifact, and reads top_k/filters from RetrievalConfig.

**Retrieval-only mode:** When `retrieval_only=True`, the generator
and prompt template are not constructed. `run()` returns a
GenerationResult with empty answer and populated retrieved_chunks.
This enables fast, cheap retriever evaluation.

### 8.3 AgentPipeline (src/pipeline/agent.py)

Processes queries through an agent-based system. Produces the same
GenerationResult type as QueryPipeline.

**RAGTool:** Each RAG-type agent wraps a mini QueryPipeline with
its own retrieval config (filters, top_k) and prompt config, but
sharing the same vectorstore and embedder. This means per-agent
source filtering works through the existing retrieval filter
mechanism.

**Single-agent mode (v2):** One LLM in a ReAct loop choosing
between tools. Needs its own generator for the reasoning loop.

**Multi-agent mode (v3):** Supervisor LLM routes to specialized
agents, inspects results, decides next action. Needs its own
generator for routing decisions and final response synthesis.

### 8.4 RAGPipeline (src/pipeline/rag.py)

Top-level dispatcher. Reads `pipeline_mode` from config and
constructs either QueryPipeline or AgentPipeline. Manages index
building/caching. Provides `active_pipeline` property and `query()`
convenience method. Contains `run_experiment()` function as the
CLI entry point.

---

## 9. Evaluation System

### 9.1 Dataset Format

Evaluation datasets are JSON files in `data/datasets/`. Format:

```json
[
  {
    "query": "What is the University policy on grade forgiveness?",
    "reference": "The University allows students to retake a course and use grade forgiveness up to 3 times..."
  },
  {
    "query": "What services does the University Police Department offer?",
    "reference": "The University Police Department offers safety escorts, crime prevention programs..."
  }
]
```

Fields:
- `query` (required): The student's question
- `reference` (required): The ground truth answer, used by RAGAS
  for answer_correctness, context_recall, context_entity_recall

### 9.2 Evaluation Modes

**Full mode** (`evaluation.mode: "full"`): Runs the complete
pipeline (retrieve + generate). Computes all metrics: answer_correctness,
context_precision, faithfulness, context_entity_recall, answer_relevancy.

**Retrieval-only mode** (`evaluation.mode: "retrieval_only"`): Skips
generation. Only retrieves context and computes retriever-specific
metrics: context_precision, context_entity_recall. 10x faster, no
LLM cost. Use this when iterating on chunking, embedding, or
retrieval strategy.

### 9.3 Multi-Run Execution

Each experiment runs `num_runs` times (default: 3). Results are
aggregated as mean ± std per metric. This handles LLM
non-determinism (especially at temperature > 0) and gives
confidence intervals.

### 9.4 Results Output

```
results/<experiment_name>/
├── summary.json        ← Aggregated metrics + full config snapshot
├── run_1.json          ← Per-sample metrics for run 1
├── run_2.json
└── run_3.json
```

### 9.5 RAGAS Integration

The `compute_metrics()` method in `evaluator.py` is the integration
point. Implementation will:
1. Build metric objects from `config.active_metrics`
2. Construct a HuggingFace Dataset from EvalSamples
3. Call `ragas.evaluate(dataset, metrics)`
4. Return the scores dict

### 9.6 Cross-Experiment Comparison

`src/evaluation/comparison.py` (to be implemented) loads multiple
experiment summaries and produces comparison tables matching the
format from the ACMSE paper (Table 2):

```
| Experiment                  | Ans.Corr | Ctx.Prec | Faith. | CER   | Ans.Rel |
|-----------------------------|----------|----------|--------|-------|---------|
| base (recursive 512)        | 0.815    | 0.883    | 0.906  | 0.291 | 0.867   |
| chunking_recursive_1000     | ...      | ...      | ...    | ...   | ...     |
| reranking_cross_encoder     | ...      | ...      | ...    | ...   | ...     |
```

---

## 10. Known Architectural Gaps

These are limitations we've identified and accepted for now. Each
has a severity rating and a plan for when to address it.

### GAP 1: Agent supervisor lacks its own generator (MEDIUM)

**Problem:** In multi-agent mode, the supervisor needs to: (a) make
routing decisions (which agent to call), (b) synthesize responses
from multiple agents, and (c) generate the final answer. All three
require LLM generation calls. Currently, the AgentPipeline has
placeholder loop logic that returns the first successful tool's
raw output without supervisor synthesis.

**Impact:** Agent experiments will run but won't match the real v3
behavior where the supervisor synthesizes and adds context.

**Plan:** When implementing the real ReAct/supervisor loops, the
AgentPipeline will use the supervisor's LLMConfig to make generation
calls for routing and synthesis. The config already carries
`supervisor.llm` — the infrastructure is ready, only the loop
logic is placeholder.

### GAP 2: Hybrid retrieval has no clean BM25 index cache (LOW)

**Problem:** A hybrid retriever (dense + BM25) needs a keyword index
alongside the vector index. The current IndexArtifact carries only
one vectorstore. A hybrid retriever can build its own BM25 index
internally, but it won't be cached by the fingerprint system.

**Impact:** Hybrid retrieval experiments will be slower on first
run (BM25 index rebuilt each time). Subsequent runs with cached
vector index will still rebuild BM25.

**Plan:** When hybrid retrieval is implemented, extend IndexArtifact
with an optional `auxiliary_stores` dict. The hybrid retriever
stores/loads its BM25 index alongside the vector index in the
same cache directory.

### GAP 3: No multi-turn session evaluation (LOW)

**Problem:** The evaluator runs each query independently. There's
no concept of a multi-turn session where memory accumulates across
questions. Conversation memory exists in the agent pipeline but the
evaluator doesn't exercise it.

**Impact:** Can't measure whether BufferWindowMemory with
window_size=5 actually helps advising conversations.

**Plan:** Add a SessionEvaluator mode when multi-turn datasets are
available. Dataset format would change to sessions containing
ordered turn sequences.

### GAP 4: No config validation (LOW)

**Problem:** Setting `pipeline_mode: "agent"` without an agent
block, or referencing an unregistered component type, fails at
runtime with unhelpful errors.

**Plan:** Add a `validate()` method to ExperimentConfig that checks
internal consistency before the pipeline runs. Call it at the start
of `run_experiment()`.

### GAP 5: Contextual chunking has no independent stage (LOW)

**Problem:** Contextual chunking (prepending section titles to
chunks before embedding) is a post-chunking, pre-embedding
transformation. It's currently baked into chunker implementations,
meaning you can't test "recursive chunking + contextual enrichment"
vs "recursive chunking without enrichment" as an isolated variable.

**Plan:** If this becomes a priority, add a `BaseChunkEnricher`
stage between chunking and embedding. For now, chunker
implementations can include it as a boolean param (e.g.,
`params.enrich_with_context: true`).

---

## 11. Variable Coverage Matrix

Every testable variable from the research factor analysis, mapped
to where it lives in the config:

### 11.1 Indexing Variables

| Variable | Config Location | Isolated? |
|----------|----------------|-----------|
| Document parser | `indexing.sources[].ingest.type` / `.params.parser` | Yes |
| Number of sources | `indexing.sources` list length | Yes |
| Source selection | `indexing.sources` list contents | Yes |
| Chunking strategy | `indexing.chunking.type` | Yes |
| Chunk size | `indexing.chunking.params.chunk_size` | Yes |
| Chunk overlap | `indexing.chunking.params.chunk_overlap` | Yes |
| Separators | `indexing.chunking.params.separators` | Yes |
| Embedding model | `indexing.embedding.params.model_name` | Yes |
| Embedding normalization | `indexing.embedding.params.normalize` | Yes |
| Instruction prefix | `indexing.embedding.params.instruction_prefix` | Yes |
| Vectorstore backend | `indexing.vectorstore.type` | Yes |
| Index type (HNSW/flat) | `indexing.vectorstore.params.index_type` | Yes |
| Distance metric | `indexing.vectorstore.params.metric` | Yes |

### 11.2 Query Variables (Linear Pipeline)

| Variable | Config Location | Isolated? |
|----------|----------------|-----------|
| Query transform technique | `query.query_transform.type` | Yes |
| Query transform LLM | `query.query_transform.params.llm` | Yes |
| Retrieval method | `query.retrieval.type` | Yes |
| top_k_retrieve | `query.retrieval.top_k_retrieve` | Yes |
| top_k_final | `query.retrieval.top_k_final` | Yes |
| Search type (similarity/MMR) | `query.retrieval.params.search_type` | Yes |
| Source filters | `query.retrieval.filters` | Yes |
| Reranking method | `query.reranking.type` | Yes |
| Reranking model | `query.reranking.params.model_name` | Yes |
| Generator backend | `query.generation.type` | Yes |
| LLM model | `query.generation_llm.model_name` | Yes |
| LLM temperature | `query.generation_llm.temperature` | Yes |
| LLM max tokens | `query.generation_llm.max_tokens` | Yes |
| System prompt text | `query.prompt.system_template` | Yes |
| Chain-of-thought | `query.prompt.use_chain_of_thought` | Yes |
| Citation style | `query.prompt.citation_style` | Yes |
| Few-shot examples | `query.prompt.few_shot_examples` | Yes |
| Context format | `query.prompt.context_format` | Yes |
| Max context tokens | `query.prompt.max_context_tokens` | Yes |

### 11.3 Agent Variables

| Variable | Config Location | Isolated? |
|----------|----------------|-----------|
| Pipeline mode (none/single/multi) | `pipeline_mode` | Yes |
| Agent mode (single/multi) | `agent.mode` | Yes |
| Supervisor LLM | `agent.supervisor.llm` | Yes |
| Routing strategy | `agent.supervisor.routing` | Yes |
| Routing prompt | `agent.supervisor.routing_prompt` | Yes |
| Max iterations | `agent.supervisor.max_iterations` | Yes |
| Memory type | `agent.memory.type` | Yes |
| Memory window size | `agent.memory.window_size` | Yes |
| Agent roster | `agent.agents` list | Yes |
| Per-agent source filter | `agent.agents[].retrieval.filters` | Yes |
| Per-agent top_k | `agent.agents[].retrieval.top_k_*` | Yes |
| Per-agent prompt | `agent.agents[].prompt` | Yes |
| Available tools | `agent.tools` list | Yes |

### 11.4 Evaluation Variables

| Variable | Config Location | Isolated? |
|----------|----------------|-----------|
| Dataset | `evaluation.dataset` | Yes |
| Evaluation mode | `evaluation.mode` | Yes |
| Metric selection | `evaluation.metrics` | Yes |
| Number of runs | `evaluation.num_runs` | Yes |
| Evaluator LLM | `evaluation.evaluator_llm` | Yes |

---

## 12. Recreating Historical Versions

### ARGObot v1 (Retrieval-based, ACMSE paper)

```yaml
pipeline_mode: "linear"
indexing:
  sources:
    - name: "student_handbook"
      path: "data/sources/student_handbook.pdf"
      ingest: { type: "pdf" }
  chunking: { type: "recursive", params: { chunk_size: 1200, chunk_overlap: 200 } }
  embedding: { type: "google", params: { model_name: "models/embedding-001" } }
  vectorstore: { type: "chroma" }
query:
  retrieval: { type: "dense", top_k_retrieve: 3, top_k_final: 3 }
  generation_llm: { provider: "google", model_name: "gemini-1-pro" }
  prompt:
    system_template: "Answer using ONLY verbatim quotes..."
    citation_style: "verbatim"
```

### ARGObot v2 (Single-Agent, ACMSE paper)

```yaml
pipeline_mode: "agent"
agent:
  mode: "single"
  llm: { provider: "openai", model_name: "gpt-4" }
  memory: { type: "buffer_window", window_size: 5 }
  agents:
    - name: "rag_handbook"
      type: "rag"
      retrieval: { type: "dense", top_k_retrieve: 3, top_k_final: 3 }
  tools:
    - { name: "web_search", type: "tool", tool: "google_search" }
    - { name: "email", type: "tool", tool: "gmail" }
```

### ARGObot v3 (Multi-Agent, SURP 2025)

```yaml
pipeline_mode: "agent"
agent:
  mode: "multi"
  supervisor:
    llm: { provider: "ollama", model_name: "qwen2.5:32b" }
    routing: "llm"
  memory: { type: "buffer_window", window_size: 5 }
  agents:
    - name: "handbook_agent"
      type: "rag"
      retrieval: { filters: { source_name: "student_handbook" } }
    - name: "knowledge_base_agent"
      type: "rag"
      retrieval: { filters: { source_name: "knowledge_base" } }
  tools:
    - { name: "web_search", type: "tool", tool: "google_search" }
    - { name: "email", type: "tool", tool: "gmail" }
```

### Controlled Comparison (same index, different orchestration)

All three configs above can use identical indexing blocks. The
index_fingerprint will be the same. Change ONLY pipeline_mode
and agent config to isolate the effect of orchestration.

---

## 13. Implementation Priority

Each step unlocks a testable capability. Don't skip ahead.

### Phase 1: Minimal viable pipeline

1. **PDF ingestor** (PyMuPDF) — unlocks ingestion
2. **Recursive chunker** (LangChain-style) — unlocks chunking
3. **HuggingFace embedder** (bge-m3) — unlocks embedding
4. **FAISS vectorstore** — unlocks index building + caching
5. **Dense retriever** — unlocks retrieval-only experiments

**Milestone:** Can run `retrieval_only_baseline.yaml` and inspect
what chunks are retrieved for each test question.

### Phase 2: Full linear pipeline

6. **Chat prompt template** — formats context into prompts
7. **Ollama generator** — wraps local LLM calls
8. **RAGAS evaluator integration** — computes real metrics
9. **Comparison utility** — cross-experiment tables

**Milestone:** Can run `base.yaml` end-to-end and get real RAGAS
scores. Can compare multiple experiments in a table.

### Phase 3: Experiment breadth

10. Additional chunkers (semantic, document-structure)
11. Additional embedding models (nomic, OpenAI, GTE)
12. Cross-encoder reranker
13. HyDE / multi-query transformers
14. Additional vectorstores (ChromaDB for comparison)

**Milestone:** Can run the full Phase 1-5 experiment matrix from
the factor analysis document.

### Phase 4: Agent pipeline

15. Single-agent ReAct loop implementation
16. Multi-agent supervisor loop implementation
17. Web search tool (Serper/Tavily API)
18. Gmail tool (or mock)

**Milestone:** Can recreate v1/v2/v3 and run the controlled
comparison.

### Phase 5: Advanced

19. Hybrid retrieval (BM25 + dense)
20. Multi-turn session evaluation
21. Additional rerankers (ColBERT, Cohere)
22. Config validation utility

---

## 14. Dependencies

Minimal required (Phase 1-2):
```
pyyaml          # Config loading
faiss-cpu       # Vector store
sentence-transformers  # Embedding models (HuggingFace)
pymupdf         # PDF parsing
ragas           # Evaluation metrics
datasets        # HuggingFace datasets (RAGAS dependency)
```

Phase 3 additions:
```
langchain-text-splitters  # Semantic chunking
rank-bm25       # BM25 retrieval
chromadb        # Alternative vectorstore
```

Phase 4 additions:
```
langchain-core  # Agent ReAct loop utilities (optional)
langgraph       # Agent orchestration (optional)
```
