# Roadmap

> Forward-looking plan for the RAG experimentation harness.
> For the system as currently built, see [ARCHITECTURE.md](ARCHITECTURE.md).
>
> **Last updated:** 2026-06-02

---

## 1. Goals

This repository is a **RAG experimentation harness**: a config-driven
framework for isolating and measuring how individual RAG components
and paradigms affect end-to-end quality. The long-term aim is broad,
faithful coverage of the modern RAG design space so that any component
choice can be swapped via config and evaluated under controlled
conditions.

Guiding objectives:

1. **Cover the standard pipeline first.** Prioritize the components
   and paradigms that constitute the field's default, well-established
   practice — the "de facto strong baseline" of dense + sparse + rerank
   retrieval, standard query optimization, and standard chunking — before
   specialized or research-frontier techniques.

2. **Isolate every variable.** Each component is swappable through a
   single config change, so an experiment changes exactly one thing and
   the effect is attributable.

3. **Grow toward full coverage.** Build out the three layers of the
   design space in priority order: (a) the core retrieval/generation
   pipeline, (b) advanced paradigms layered on top (agentic, self-
   reflective), and (c) evaluation and robustness. All three are in
   scope eventually; the sequence is driven by how standard each piece is.

This roadmap is no longer organized around replicating specific prior
ARGObot versions. Version replication has been retired as an objective;
the harness stands on its own as a general-purpose RAG testbed.

---

## 2. Design Principles

These are documented in ARCHITECTURE.md and apply to all future
work. One principle governs component implementation specifically:

### Framework Wrapper Strategy

**Use framework libraries when the component is a commodity.
Write custom when the component IS the experiment.**

Chunking, PDF parsing, embedding, reranking, and lexical search are
commodities — use proven libraries (`langchain-text-splitters`,
`sentence-transformers`, `bm25s`). Score fusion, domain-tuned query
transformers, and novel chunking strategies are the experiment itself —
implement those directly.

**Never let framework dependencies leak past the interface
boundary.** A LangChain-backed chunker imports
`RecursiveCharacterTextSplitter` inside its class, but produces
`list[Chunk]` — our type. The pipeline never touches LangChain
types. If the library is swapped, nothing changes outside that
one file.

This is already the pattern for embeddings (`sentence-transformers`
behind `BaseEmbedder`), vectorstore (`faiss-cpu` behind
`BaseVectorStore`), reranking (`sentence-transformers` `CrossEncoder`
behind `BaseReranker`), and PDF parsing (`pymupdf` behind
`BaseIngestor`). Extend it consistently.

---

## 3. What Exists Today

### Registered Components

| Category | Component | Registry Name | Notes |
|----------|-----------|---------------|-------|
| Ingest | `PDFIngestor` | `pdf` | PyMuPDF |
| Chunking | `LangChainRecursiveChunker` | `recursive` | `langchain-text-splitters` wrapper (config-facing name is impl-agnostic) |
| Chunking | `SemanticChunker` | `semantic` | Custom; embedding-breakpoint splits (no new dep) |
| Chunk Enricher | `NoOpChunkEnricher` | `none` | No-op passthrough |
| Chunk Enricher | `ContextualChunkEnricher` | `contextual` | LLM situating context; sets `index_text` only (retrieval-isolated) |
| Embedding | `HuggingFaceEmbedder` | `huggingface` | sentence-transformers, bge-m3 default |
| Embedding | `GoogleEmbedder` | `google` | `google-genai` SDK |
| Embedding | `OpenAIEmbedder` | `openai` | `openai` SDK |
| Embedding | `EdenAIEmbedder` | `edenai` | `langchain-community` gateway |
| Vectorstore | `FAISSVectorStore` | `faiss` | Cosine + L2, metadata post-filtering |
| Vectorstore | `ChromaVectorStore` | `chroma` | ChromaDB, native metadata filtering |
| Sparse Index | `BM25LexicalIndex` | `bm25` | `bm25s` + PyStemmer; mmap load; metadata over-retrieve + post-filter |
| Retrieval | `DenseRetriever` | `dense` | Single-vector similarity search |
| Retrieval | `BM25Retriever` | `bm25` | Delegates to `sparse_index` |
| Retrieval | `HybridRetriever` | `hybrid` | Composes named child retrievers; RRF or weighted fusion |
| Query Transform | `PassthroughQueryTransformer` | `passthrough` | Returns query unchanged |
| Query Transform | `ContextualizerQueryTransformer` | `contextualizer` | LLM reformulation with history |
| Query Transform | `HyDEQueryTransformer` | `hyde` | LLM hypothetical-doc; per-branch routing for hybrid |
| Query Transform | `MultiQueryQueryTransformer` | `multi_query` | LLM N reformulations; RRF fusion across them |
| Reranking | `NoOpReranker` | `none` | Passes through, truncates to top_k |
| Reranking | `CrossEncoderReranker` | `cross_encoder` | gte-reranker-modernbert-base default |
| Generation | `OllamaGenerator` | `ollama` | Local LLM via Ollama |
| Generation | `EdenAIGenerator` | `edenai` | Cloud LLM via Eden AI gateway |
| Generation | `GoogleGenerator` | `google` | Gemini via `google-genai` SDK |
| Generation | `OpenAIGenerator` | `openai` | OpenAI chat completions |
| Prompts | `ChatPromptTemplate` | `chat` | Numbered/plain context, CoT, citations |
| Memory | `NoMemory` | `none` | No-op |
| Memory | `BufferWindowMemory` | `buffer_window` | Last N turns |

### Infrastructure Completed

- Config system with YAML inheritance, cycle detection, index
  fingerprinting, and upfront validation
- Linear query pipeline (query transform → retrieve → rerank →
  prompt → generate)
- Index caching by SHA-256 fingerprint, with **empty-config
  canonicalization** so optional indexing components (sparse index
  today; chunk enricher in Phase C) don't disturb the fingerprint
  when unused
- **Auxiliary indexes** carried inside `IndexArtifact.auxiliary_stores`
  (typed `dict[str, BaseLexicalIndex | BaseVectorStore]`); BM25 lives
  under the `"bm25"` key
- **Rank-list fusion helpers** in `core/fusion.py` (RRF + weighted
  score fusion + max-score dedup); reused by the hybrid retriever and
  by multi-query / HyDE result fusion
- **Multi-query plumbing** (Phase B): `TransformedQuery(text, branch)`
  type, `BaseQueryTransformer.transform() -> list[TransformedQuery]`,
  and `BaseRetriever.retrieve_multi()` with **two-level fusion** (across
  query reformulations, then across retrieval methods) and **per-branch
  routing** so HyDE can send a hypothetical doc to the dense child and
  the original query to BM25. Pipeline-level fusion strategy is a
  first-class config field (`query.query_transform.fusion`)
- RAGAS evaluation with multi-run aggregation (mean ± std) and a
  dedicated, fixed evaluator embedding for cross-experiment comparability
- Standardized evaluator LLM (GPT-4.1 via EdenAI) for formal runs;
  Ollama for local smoke testing
- No-score smoke evaluation mode (verifies the pipeline without
  spending judge-LLM calls)
- Configurable RAGAS execution settings (`EvalRunConfig`)
- Experiment result saving (summary.json + config.yaml + JSONL)
- Cross-experiment comparison with config diffing
- CI pipeline (ruff + mypy + pytest), pre-commit, Makefile
- Single-agent ReAct pipeline (Phase D1) via native tool calling
- Uniform construction contract: `BuildContext` + polymorphic `build`
  (`components/build.py`) so the pipeline resolves components by name and
  never branches on a concrete implementation; symmetric `ValidateContext`
  for component-owned param validation. The recursion substrate for D2.
- Git SHA tracking for reproducibility
- Config structure: `base.yaml` + `smoke.yaml` + `experiments/`
  organized by what each isolates

### Coverage Against the Standard Pipeline

The component categories map directly onto the standard RAG pipeline
anatomy. Current coverage by stage:

| Pipeline Stage | Standard Options | Have | Missing (standard) |
|----------------|------------------|------|--------------------|
| Ingestion | PDF, HTML, text | PDF | — (sufficient for now) |
| Chunking | recursive, semantic, contextual | recursive, semantic, contextual | — |
| Embedding | open + commercial | 4 providers | — (broad coverage) |
| Vectorstore | dense ANN | FAISS, Chroma | — |
| Sparse index | BM25 / SPLADE | BM25 | (SPLADE — advanced) |
| Query optimization | rewrite/expand, decompose, multi-query | contextualizer, HyDE, multi-query | (decompose — advanced) |
| Retrieval | dense, sparse, hybrid | dense, BM25, hybrid (RRF + weighted) | — |
| Reranking | cross-encoder, late-interaction, LLM | cross-encoder | (ColBERT/LLM — advanced) |
| Generation | grounded gen, citation, abstention | chat + citation | abstention |

With **Phases A, B, and C complete**, the harness now covers the field's
de facto strong-baseline retrieval matrix (dense + sparse + hybrid +
reranking), standard query optimization (HyDE, multi-query), and standard
chunking (recursive at multiple sizes + semantic + contextual enrichment).
The standard pipeline is fully covered; remaining work is advanced
paradigms (Phase D agentic) and evaluation depth (Phase E).

---

## 4. Sequencing Strategy

Work is organized into tracks rather than a strict linear sequence.
The priority ordering reflects how standard each piece is:

1. **Phase A — Complete the standard retrieval baseline** (sparse +
   hybrid). ✅ **Done.** The harness now runs dense / BM25 / hybrid
   (RRF or weighted) under the same evaluation harness.
2. **Phase B — Standard query optimization** (HyDE, multi-query).
   ✅ **Done.** Both run as `BaseQueryTransformer` implementations with
   pipeline-level fusion and per-branch routing for hybrid setups.
3. **Phase C — Standard chunking alternatives** (chunk-size sweeps,
   semantic, contextual retrieval). ✅ **Done.** Custom `semantic`
   chunker + a `BaseChunkEnricher` stage with the `contextual` enricher
   (situating context applied to retrieval text only).
4. **Phase D — Agentic RAG** (single-agent ReAct, then multi-agent).
   Kept because agentic retrieval is now a standard paradigm, not
   because it replicates any prior version.
5. **Phase E — Evaluation depth & robustness** (retrieval/answer
   decoupling analysis, abstention, knowledge-conflict probes).
6. **Phase F — Advanced/optional** (late-interaction reranking,
   graph RAG, multimodal). Explicitly lower priority; pursued only
   if a concrete need arises.

Phases A and B both landed within the existing linear pipeline; Phase C
added one well-scoped extension (Section 7.2 — the chunk-enricher stage)
plus the `index_text` decoupling. Phase D activates the dormant agent
pipeline. Phases E–F are open-ended.

---

## 5. Implementation Phases

### Phase A — Standard Retrieval Baseline (sparse + hybrid) ✅ Done

Delivered the field's de facto strong retrieval baseline — dense +
sparse + fusion — so every later experiment compares against a
credible reference rather than dense-only.

What landed:

- `BaseLexicalIndex` ABC and `BM25LexicalIndex` (registry:
  `sparse_index/bm25`) wrapping `bm25s` with PyStemmer, mmap-friendly
  load, and FAISS-style metadata over-retrieve + post-filter.
- `BM25Retriever` (`retrieval/bm25`) and `HybridRetriever`
  (`retrieval/hybrid`) — the latter composes named child retrievers
  declared as an ordered list, with **named weights** (dict keyed by
  child `name`) so config diffs stay stable under list reordering.
  Nested hybrid is rejected by the validator.
- `core/fusion.py` — pure rank-list math (`reciprocal_rank_fusion`,
  `weighted_score_fusion`). Reused by Phase B's multi-query.
- `IndexArtifact.auxiliary_stores` and `IndexingConfig.sparse_index`
  (Section 7.1, also done). BM25 is built **before** embedding so
  tokenizer/stemmer misconfiguration fails fast; the cache lives
  under `data/indexes/<fingerprint>/bm25/`.
- Four experiment configs under `configs/experiments/retrieval/`:
  `bm25`, `hybrid_rrf`, `hybrid_rrf_rerank`, `hybrid_weighted`.
  (The dense baseline is `base.yaml` itself, not a new file.)

Achieved milestone: dense vs. BM25 vs. hybrid (with and without
cross-encoder reranking) all run through the same harness and land
in a single comparison table. This is the canonical strong-baseline
matrix the rest of the roadmap measures against.

### Phase B — Standard Query Optimization ✅ Done

Delivered the standard pre-retrieval query transformations. Both reuse
the existing generator infrastructure and `BaseQueryTransformer`
interface; both can derail when the LLM hallucinates — a trade-off the
harness now measures rather than assumes.

| # | Component | Registry | Interface | Notes |
|---|-----------|----------|-----------|-------|
| 3 | HyDE | `hyde` | `BaseQueryTransformer` | LLM generates a hypothetical answer; retrieve on its embedding |
| 4 | Multi-query | `multi_query` | `BaseQueryTransformer` | LLM generates N reformulations; RRF-fuse results |

What landed:

- `HyDEQueryTransformer` (`query_transform/hyde`) — emits one
  hypothetical document by default (paper-canonical pure replacement),
  with `num_hypotheticals`, `include_original`, `branch`, and
  `original_branch` config knobs. Branch routing is opt-in (default
  broadcast, like every transformer); the canonical HyDE+hybrid recipe
  sets `branch="dense"` for the hypothetical and `original_branch="bm25"`
  for the original.
- `MultiQueryQueryTransformer` (`query_transform/multi_query`) — one
  LLM call yields N reformulations (default 4, the RAG-Fusion count),
  parsed by a tolerant numbered-list parser with a bare-line fallback.
  All reformulations broadcast to every retriever (`branch=None`).
- **Architectural prerequisites** (shipped as a foundation commit):
  `TransformedQuery(text, branch)`, the `transform() ->
  list[TransformedQuery]` interface change, `BaseRetriever.retrieve_multi`
  (default loop+fuse) with a `HybridRetriever` override that routes by
  branch and applies **two-level fusion** (intra-branch across
  reformulations, then cross-branch across retrieval methods), a
  dedicated `QueryTransformConfig` with a first-class `fusion` field,
  and `core.fusion.max_score_dedup`.
- Five experiment configs under `configs/experiments/query_transform/`:
  `hyde_dense`, `hyde_hybrid`, `multi_query_dense`, `multi_query_hybrid`
  (RAG-Fusion canonical), `multi_query_hybrid_rerank` (full stack).

Note: multi-query and HyDE result fusion reuse
`core.fusion.reciprocal_rank_fusion` (built in Phase A) — no new
fusion code needed.

**Milestone reached:** passthrough vs. contextualizer vs. HyDE vs.
multi-query are all runnable through the same harness on the hybrid
baseline, landing in a single comparison table.

### Phase C — Standard Chunking Alternatives

Goal: add the two standard alternatives to fixed-size recursive
chunking. The review notes that beyond a competent embedder, chunk
*size* often matters more than chunking *strategy* — so chunk-size
sweeps on the existing recursive chunker are part of this phase, not
just new chunkers.

Delivered in two parts: **Part 1** (chunk-size sweeps + semantic chunker)
needed no new pipeline stage; **Part 2** (contextual enrichment) added the
`BaseChunkEnricher` stage (§7.2) plus the `index_text` decoupling so the
situating context affects retrieval only.

| # | Component | Registry | Interface | Library | Status |
|---|-----------|----------|-----------|---------|--------|
| 5 | Semantic chunker | `semantic` | `BaseChunker` | Custom embedding-breakpoint split | ✅ Done (Part 1) |
| 6 | Contextual chunk enricher | `contextual` | `BaseChunkEnricher` | Custom (LLM prepends situating context per chunk) | ✅ Done (Part 2) |

All Phase C experiment configs live under `configs/experiments/chunking/`:
`chunk_size_{256,512,1024}`, `chunk_semantic`, and `chunk_contextual`.

**Semantic chunker** splits on embedding-similarity breakpoints rather
than fixed token counts.

**Contextual retrieval** (the Anthropic recipe) prepends a short,
model-generated description situating each chunk in its source document
*before* embedding and indexing. Combined with hybrid retrieval and
reranking, it is reported to cut top-20 retrieval failures by roughly
two-thirds. This requires the new enricher stage (Section 7.2) that
sits between chunking and embedding — a clean, reusable place for any
pre-embedding transformation.

**Milestone:** recursive (multiple sizes) vs. semantic vs. contextual-
enriched, on the hybrid baseline.

### Phase D — Agentic RAG

Goal: activate the dormant agent pipeline. Agentic retrieval — the
model deciding when and what to retrieve as part of multi-step
reasoning — is now a standard paradigm, so it belongs in the harness.
This is **not** framed as replicating any prior agent version.

#### Phase D1 — Single-Agent (ReAct)

| # | Component | Location | Status |
|---|-----------|----------|--------|
| 7 | ReAct agent loop | `pipeline/agent.py` | ✅ Done — reason → act (tool) → observe → repeat |
| 8 | RAG tool wrapper | `components/tools.py` | ✅ Done — wraps a retrieval-only QueryPipeline |
| 9 | Web search tool | `components/tools.py` | Pending (end of D1) — `BaseTool`; external API |

The agent treats retrieval as one tool among several, deciding for
itself when it has gathered enough evidence. **Mechanism: native
tool/function calling** (not text-protocol parsing) — `BaseGenerator.generate`
gained an optional `tools` parameter, implemented for all four providers
(OpenAI, Google, Ollama, and EdenAI via `ChatEdenAI.bind_tools`); the
experimental variable is *agentic control flow vs. linear*, so a robust
mechanism avoids confounding the measurement. The agent and the generators
exchange messages in a neutral OpenAI-style schema (assistant turns carry
`tool_calls`; tool results carry `tool_call_id`); each generator translates it
to its provider.

The RAG tool wraps a `QueryPipeline` (the experiment's retrieval + rerank
stack, in retrieval-only mode, query-transform forced to passthrough) and
returns the raw retrieved chunks as the observation while populating
`ToolResult.retrieved_chunks`; the agent unions those across calls (dedup by
`chunk_id`, cap at `top_k_final`) into the final `GenerationResult`, so the
evaluator scores retrieval in agent mode on the same metrics as linear. The
loop runs up to `agent.max_iterations` steps, forcing a final answer
(`tools=None`) if the budget is exhausted; tool failures become recoverable
observations. Chief risk to measure: error propagation from a bad early step.

Architectural prerequisites shipped as a foundation first (mirroring Phase B):
`ToolSpec`/`ToolCall` types + the `GenerationResult.tool_calls` extension, the
`generate(prompt, tools=None)` interface change, `AgentConfig.max_iterations` /
`system_prompt`, the `tool` registry category, and agent-mode config
validation. Experiment config: `configs/agent_single.yaml` (shares
`base.yaml`'s cached index, so linear-vs-agent isolates control flow);
`configs/smoke_agent.yaml` for a free local Ollama smoke.

#### Phase D2 — Multi-Agent Supervisor

| # | Component | Location | Notes |
|---|-----------|----------|-------|
| 10 | Supervisor routing | `pipeline/agent.py` | LLM routes queries to specialized agents |
| 11 | Agent roster | `pipeline/agent.py` | Per-agent retrieval filters + prompts |

A supervisor LLM routes each query to a specialized agent; each agent
has its own retrieval filters and prompt (e.g., a handbook-only agent
vs. a deadlines agent). Resolves GAP 1.

**Construction substrate (ready).** The `BuildContext` + polymorphic `build`
seam (`components/build.py`) is the recursion D2 builds on: a `multi`
`AgentPipeline` builds a supervisor that constructs each specialist sub-agent
via `ctx.build`-style recursion through the registry — the same pattern the
hybrid retriever already uses to build its children. The dead D1-era
`SupervisorConfig` / `AgentConfig.agents` / per-agent `AgentDefinitionConfig`
fields were removed; D2 reintroduces a coherent supervisor + agent-roster
config (each agent carrying its own `llm` / `tools` / retrieval / prompt) rather
than carrying speculative fields ahead of the implementation. `AgentConfig.mode`
is the seam; `multi` is rejected by the validator until this phase lands.

**Milestone:** linear vs. single-agent vs. multi-agent on identical
indexing, measuring whether agentic control flow actually improves
answer quality on advising queries or just adds latency.

### Phase E — Evaluation Depth & Robustness

Goal: address the review's recurring caution that retrieval metrics and
answer metrics are only loosely coupled, and that automated judges
carry biases.

| # | Item | Notes |
|---|------|-------|
| 12 | Retrieval/answer decoupling report | Surface where better retrieval did NOT improve answers (and vice-versa) |
| 13 | Abstention handling | Prompt + metric for "I don't know" when evidence is absent |
| 14 | Multi-turn session evaluation | Exercises memory across ordered turns (resolves GAP 3) |
| 15 | Knowledge-conflict probe (optional) | Inject contradicting context; measure whether the model defers appropriately |

LLM-as-judge biases (position, verbosity, self-enhancement) should be
documented in evaluation outputs and periodically spot-checked against
human judgment on a held-out slice. This is a methodology note as much
as a feature.

### Phase F — Advanced / Optional

Goal: specialized techniques pursued only on concrete need. Explicitly
lower priority; the review covers these but they are not standard
default practice for a text corpus of this kind.

| # | Component | Notes |
|---|-----------|-------|
| 16 | Late-interaction reranker (ColBERT) | Multi-vector; storage-heavy |
| 17 | LLM reranker (RankGPT-style) | Listwise prompting; latency/cost-heavy |
| 18 | Graph RAG | Entity-relation graph + traversal; pays off only on multi-hop/corpus-level synthesis |
| 19 | Additional ingestors | HTML, markdown, OCR for scanned docs |

---

## 6. Variable Coverage Matrix

The variables the harness can sweep, by pipeline stage. Bold entries
are not yet implemented.

### 6.1 Indexing Variables

| Variable | Config Location | Status |
|----------|----------------|--------|
| Source documents | `indexing.sources` | done |
| Chunking strategy | `indexing.chunking.type` | done (recursive, semantic) |
| Chunk size / overlap | `indexing.chunking.params` | done |
| Embedding model | `indexing.embedding` | done (4 providers) |
| Vectorstore | `indexing.vectorstore.type` | done (FAISS, Chroma) |
| Sparse index | `indexing.sparse_index.type` | done (BM25) |
| Chunk enricher | `indexing.chunk_enricher.type` | done (none, contextual) |

### 6.2 Query Variables

| Variable | Config Location | Status |
|----------|----------------|--------|
| Query transform | `query.query_transform.type` | done (passthrough, contextualizer, HyDE, multi-query) |
| Multi-query fusion | `query.query_transform.fusion` | done (rrf, max) |
| Retrieval method | `query.retrieval.type` | done (dense, BM25, hybrid) |
| Fusion strategy | `query.retrieval.params.fusion` | done (RRF, weighted) |
| Retrieval depth | `query.retrieval.top_k_retrieve` | done |
| Final chunk count | `query.retrieval.top_k_final` | done |
| Reranker | `query.reranking.type` | none, cross_encoder done |
| Generator / LLM | `query.generator` (LLMConfig) | done (4 providers) |
| Prompt strategy | `query.prompt` | done (CoT, citation, context format) |

### 6.3 Agent Variables (Phase D)

| Variable | Config Location | Status |
|----------|----------------|--------|
| Pipeline mode | `pipeline_mode` | linear + **agent (single)** done |
| Agent mode | `agent.mode` | single done; **multi** pending (D2) |
| Max iterations | `agent.max_iterations` | done |
| Reasoning LLM | `agent.llm` | done (native tool calling, 4 providers) |
| Tool roster | `agent.tools` | done (`rag`; **web_search** pending — end of D1) |
| Memory | `agent.memory.type` | none, buffer_window done (plumbed; single-turn loop) |

### 6.4 Evaluation Variables

| Variable | Config Location | Status |
|----------|----------------|--------|
| Dataset | `evaluation.dataset` | done |
| Mode | `evaluation.mode` | full, retrieval_only, no-score smoke done |
| Metrics | `evaluation.metrics` | done |
| Number of runs | `evaluation.num_runs` | done |
| Evaluator LLM | `evaluation.evaluator_llm` | done |
| Evaluator embedding | `evaluation.evaluator_embedding` | done |
| Run config | `evaluation.run_config` | done |

---

## 7. Architectural Extensions Required

Two well-scoped changes unblock Phases A and C. Both are additive and
backward-compatible.

### 7.1 Auxiliary indexes in `IndexArtifact` ✅ Done (delivered with Phase A)

Shipped as part of Phase A. Final shape:

- `IndexArtifact.auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore]`
  carries auxiliary indexes; BM25 lives under `"bm25"`. Typed (not
  `Any`) but extensible — admits a future SPLADE store (a
  `BaseVectorStore`) under its own key.
- `IndexingConfig.sparse_index: ComponentConfig` declares the index;
  the indexing pipeline builds it when `type` is non-empty.
- The fingerprint includes `sparse_index` **only when configured**
  (empty-config canonicalization), so adding the field to the
  config schema didn't disturb existing dense-only fingerprints.
- BM25 is indexed before embedding (fail-fast on tokenizer config).

### 7.2 Chunk-enricher stage (unblocks contextual retrieval) ✅ Done (Phase C Part 2)

Delivered. A `BaseChunkEnricher` stage (category `chunk_enricher`) runs
between chunking and embedding/indexing; the no-op default (`none`) leaves
existing configs unchanged, and `indexing.chunk_enricher` joins the
fingerprint only when set (empty-config canonicalization), so dense/hybrid
fingerprints were undisturbed. The `ContextualChunkEnricher` (`contextual`)
implements the Anthropic recipe.

Coupled change: `Chunk.index_text` (with the `text_for_index` accessor)
decouples the embed/index text from the stored `content`. Embedders and the
BM25 index read `text_for_index`; the vector store still persists `content`,
so the enrichment's effect is isolated to retrieval (the generator and RAGAS
keep seeing the original chunk). This is the clean home for any future
pre-embedding transform (late chunking, proposition extraction,
title-prepending).

### Lower-priority gaps (unchanged)

- **GAP — Embedder re-instantiation on cache load (LOW):** loading a
  cached index still instantiates the embedding model for query-time
  embedding. Performance only, no correctness issue. Could lazy-load
  `HuggingFaceEmbedder` if startup becomes a bottleneck.

### Deferred quality follow-ups (documented, not scheduled)

Surfaced by the 2026-06 hardening review; intentionally not done then to keep that
PR focused. None affect correctness.

- **Generators cross-provider duplication (LOW):** `components/generators.py` repeats
  a Params/api-key/retry/extract skeleton per provider. A thin SDK-client base
  (`_get_api_key`, unified `_call_api` naming, a message-translation base) could lift
  ~20%. Quality only — provider APIs genuinely differ.
- **Index persistence uses pickle (LOW):** FAISS/BM25 `chunks.pkl` is version-fragile
  and unsafe to share. Move to a versioned/safe format only if indexes are ever shared.
- **HyDE `num_hypotheticals>1` (LOW):** fuses rank lists rather than averaging
  embeddings as in Gao et al.; the default (1) is paper-faithful. Documented in the
  transformer's docstring.
- **`recursive_custom` vs `recursive_langchain` (LOW):** two near-equivalent chunkers;
  keep as a deliberate alt implementation or drop one.
- **`tests/` excluded from mypy (LOW):** the strict type gate doesn't cover tests.

---

## 8. Dependencies by Phase

### Current

```
pydantic                  Config models + per-component Params validation
pyyaml                    Config loading
faiss-cpu                 Vector store
numpy                     Array operations
sentence-transformers     Embeddings + cross-encoder reranking
pymupdf                   PDF parsing
bm25s                     Sparse BM25 retrieval (NumPy/SciPy)
PyStemmer                 Snowball stemmer for BM25 tokenization
ragas                     Evaluation metrics
datasets                  RAGAS transitive dependency
python-dotenv             API key management
rich                      CLI table rendering
langchain-community       EdenAI gateway
langchain-ollama          Ollama wrapper
langchain-openai          OpenAI evaluator LLM
langchain-google-genai    Google evaluator LLM
tenacity                  Retry logic
chromadb                  ChromaDB vectorstore
google-genai              Gemini generation + embeddings
openai                    OpenAI generation + embeddings
langchain-text-splitters  Recursive chunker
```

### Phase C additions

```
(none)
```

Phase C adds **no new dependencies**. The semantic chunker is a custom
embedding-breakpoint implementation reusing `numpy` + the existing
embedder (`langchain-experimental` was evaluated and rejected: its
`SemanticChunker` forces a `langchain-community` bump that breaks the
pinned RAGAS import, and the package is being sunset). HyDE, multi-query,
hybrid fusion, the chunk enricher, and the agent loop are likewise custom.

---

## 9. Experiment Matrix

Experiments are organized by what each isolates. All inherit from
`base.yaml`; all formal runs use the standardized evaluator
(GPT-4.1 via EdenAI, fixed text-embedding-3-small evaluator embedding).

### Retrieval (Phase A, done) — the core matrix

All configs under `configs/experiments/retrieval/`. The dense
baseline is `configs/base.yaml` itself; no separate `dense.yaml`.

| Experiment | Isolates | Compare against |
|-----------|----------|-----------------|
| `base.yaml` (dense baseline) | dense retrieval | — |
| `retrieval/bm25.yaml` | sparse retrieval | base (dense) |
| `retrieval/hybrid_rrf.yaml` | dense + sparse, RRF fusion | base, bm25 |
| `retrieval/hybrid_rrf_rerank.yaml` | hybrid + cross-encoder | hybrid_rrf |
| `retrieval/hybrid_weighted.yaml` | weighted vs. RRF fusion | hybrid_rrf |

### Reranking (Phase 3C, done)

| Experiment | Isolates |
|-----------|----------|
| `cross_encoder` | reranking on/off (on the 10/5 baseline) |
| `cross_encoder_topk{3,5,10}` | top_k under reranking (pool depth + final-chunk count) |

### Query Optimization (Phase B, done)

All configs under `configs/experiments/query_transform/`. Compare
against `base.yaml` (passthrough) and the Phase A retrieval matrix.

| Experiment | Isolates |
|-----------|----------|
| `hyde_dense` | HyDE vs. baseline on dense (paper-canonical) |
| `hyde_hybrid` | HyDE + per-branch routing (hypothetical→dense, original→bm25) |
| `multi_query_dense` | multi-query vs. baseline on dense |
| `multi_query_hybrid` | RAG-Fusion canonical (multi-query × hybrid, two-level RRF) |
| `multi_query_hybrid_rerank` | full strong-baseline stack (+ cross-encoder) |

### Chunking (Phase C)

| Experiment | Isolates |
|-----------|----------|
| `chunk_size_{256,512,1024}` | chunk size (recursive) |
| `chunk_semantic` | semantic vs. recursive |
| `chunk_contextual` | contextual enrichment on/off |

### Agentic (Phase D)

| Experiment | Isolates |
|-----------|----------|
| `agent_single` | ReAct single-agent vs. linear |
| `agent_multi` | multi-agent vs. single-agent |

### Standing Methodology Note

Because retrieval metrics and answer metrics decouple, every retrieval
experiment should be read on **both** levels: a change that improves
context precision but not answer correctness (or vice-versa) is a
finding, not a null result. The comparison tooling already surfaces
per-metric deltas; Phase E formalizes the decoupling report.
