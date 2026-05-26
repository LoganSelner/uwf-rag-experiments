# Roadmap

> Forward-looking plan for the RAG experimentation harness.
> For the system as currently built, see [ARCHITECTURE.md](ARCHITECTURE.md).
>
> **Last updated:** 2026-05-26

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
| Chunking | `RecursiveChunker` | `recursive_custom` | Custom implementation |
| Chunking | `LangChainRecursiveChunker` | `recursive_langchain` | `langchain-text-splitters` wrapper |
| Embedding | `HuggingFaceEmbedder` | `huggingface` | sentence-transformers, bge-m3 default |
| Embedding | `GoogleEmbedder` | `google` | `google-genai` SDK |
| Embedding | `OpenAIEmbedder` | `openai` | `openai` SDK |
| Embedding | `EdenAIEmbedder` | `edenai` | `langchain-community` gateway |
| Vectorstore | `FAISSVectorStore` | `faiss` | Cosine + L2, metadata post-filtering |
| Vectorstore | `ChromaVectorStore` | `chroma` | ChromaDB, native metadata filtering |
| Retrieval | `DenseRetriever` | `dense` | Single-vector similarity search |
| Query Transform | `PassthroughQueryTransformer` | `passthrough` | Returns query unchanged |
| Query Transform | `ContextualizerQueryTransformer` | `contextualizer` | LLM reformulation with history |
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
- Index caching by SHA-256 fingerprint
- RAGAS evaluation with multi-run aggregation (mean ± std) and a
  dedicated, fixed evaluator embedding for cross-experiment comparability
- Standardized evaluator LLM (GPT-4o-mini via EdenAI) for formal runs;
  Ollama for local smoke testing
- No-score smoke evaluation mode (verifies the pipeline without
  spending judge-LLM calls)
- Configurable RAGAS execution settings (`EvalRunConfig`)
- Experiment result saving (summary.json + config.yaml + JSONL)
- Cross-experiment comparison with config diffing
- CI pipeline (ruff + mypy + pytest), pre-commit, Makefile
- Agent pipeline stub (raises `NotImplementedError`)
- Git SHA tracking for reproducibility
- Config structure: `base.yaml` + `smoke.yaml` + `experiments/`
  organized by what each isolates

### Coverage Against the Standard Pipeline

The component categories map directly onto the standard RAG pipeline
anatomy. Current coverage by stage:

| Pipeline Stage | Standard Options | Have | Missing (standard) |
|----------------|------------------|------|--------------------|
| Ingestion | PDF, HTML, text | PDF | — (sufficient for now) |
| Chunking | recursive, semantic, contextual | recursive | semantic, contextual enrichment |
| Embedding | open + commercial | 4 providers | — (broad coverage) |
| Vectorstore | dense ANN | FAISS, Chroma | — |
| Query optimization | rewrite/expand, decompose, multi-query | contextualizer | HyDE, multi-query |
| Retrieval | dense, sparse, hybrid | dense | **sparse (BM25), hybrid (RRF)** |
| Reranking | cross-encoder, late-interaction, LLM | cross-encoder | (ColBERT/LLM — advanced) |
| Generation | grounded gen, citation, abstention | chat + citation | abstention |

The single largest gap relative to standard practice is **retrieval**:
the harness only does dense retrieval, while the field's de facto
strong baseline is dense + sparse + reranking. Closing this is the
top priority (Phase A).

---

## 4. Sequencing Strategy

Work is organized into tracks rather than a strict linear sequence.
The priority ordering reflects how standard each piece is:

1. **Phase A — Complete the standard retrieval baseline** (sparse +
   hybrid). Highest priority: this is what makes the harness a credible
   strong baseline rather than a dense-only toy.
2. **Phase B — Standard query optimization** (HyDE, multi-query).
3. **Phase C — Standard chunking alternatives** (semantic, contextual
   retrieval).
4. **Phase D — Agentic RAG** (single-agent ReAct, then multi-agent).
   Kept because agentic retrieval is now a standard paradigm, not
   because it replicates any prior version.
5. **Phase E — Evaluation depth & robustness** (retrieval/answer
   decoupling analysis, abstention, knowledge-conflict probes).
6. **Phase F — Advanced/optional** (late-interaction reranking,
   graph RAG, multimodal). Explicitly lower priority; pursued only
   if a concrete need arises.

Phases A–C are all within the existing linear pipeline and require no
architectural changes beyond two well-scoped extensions (Section 7).
Phase D activates the dormant agent pipeline. Phases E–F are
open-ended.

---

## 5. Implementation Phases

### Phase A — Standard Retrieval Baseline (sparse + hybrid)

Goal: bring the harness up to the field's de facto strong retrieval
baseline — dense + sparse + fusion — so that every later experiment
compares against a credible baseline rather than dense-only.

| # | Component | Registry | Interface | Library |
|---|-----------|----------|-----------|---------|
| 1 | BM25 sparse retriever | `bm25` | `BaseRetriever` | `bm25s` |
| 2 | Hybrid retriever (RRF fusion) | `hybrid` | `BaseRetriever` | Custom fusion over dense + sparse |

**BM25 retriever.** Lexical retrieval is "remarkably hard to beat out
of distribution" and fails in different ways than dense retrieval,
which is exactly why combining them helps. `bm25s` is the current
standard Python implementation: pure NumPy/SciPy (no Java/PyTorch),
Apache 2.0, orders of magnitude faster than `rank-bm25`, and it
persists/loads indices to disk — which fits the harness's
`IndexArtifact` caching model.

**Hybrid retriever.** Combines the dense and sparse result lists.
Default fusion is Reciprocal Rank Fusion (RRF) — parameter-light,
robust, and the production default. Score-weighted fusion is an
optional alternative param. The hybrid retriever composes the two
underlying retrievers rather than reimplementing them.

**Architectural prerequisite:** resolve the auxiliary-index gap
(Section 7.1) so the BM25 index is built during indexing and cached
alongside the vector index.

Config params (BM25): `k1`, `b`, `method` (`"lucene"`/`"robertson"`/
`"bm25+"`), `stemmer`.
Config params (hybrid): `fusion` (`"rrf"`/`"weighted"`),
`rrf_k` (default 60), `dense_weight`/`sparse_weight` for weighted mode,
`top_k_retrieve` per sub-retriever.

**Milestone:** can run dense vs. BM25 vs. hybrid (with and without
cross-encoder reranking) and compare retrieval and answer metrics in
one table. This is the canonical strong-baseline matrix.

### Phase B — Standard Query Optimization

Goal: add the standard pre-retrieval query transformations. All reuse
the existing generator infrastructure and `BaseQueryTransformer`
interface.

| # | Component | Registry | Interface | Notes |
|---|-----------|----------|-----------|-------|
| 3 | HyDE | `hyde` | `BaseQueryTransformer` | LLM generates a hypothetical answer; retrieve on its embedding |
| 4 | Multi-query | `multi_query` | `BaseQueryTransformer` | LLM generates N reformulations; union/fuse results |

HyDE embeds a generated hypothetical document instead of the raw
query, on the rationale that an answer resembles its supporting
passages more than the question does. Multi-query issues several
reformulations in parallel and fuses their results, trading retrieval
cost for recall. Both can derail when the LLM hallucinates — that
trade-off is itself worth measuring on the advising corpus.

Note: multi-query that *fuses* result lists shares fusion logic with
the hybrid retriever (Phase A). Factor RRF into a small shared helper
so both use it.

**Milestone:** passthrough vs. contextualizer vs. HyDE vs. multi-query,
each measured on the hybrid baseline.

### Phase C — Standard Chunking Alternatives

Goal: add the two standard alternatives to fixed-size recursive
chunking. The review notes that beyond a competent embedder, chunk
*size* often matters more than chunking *strategy* — so chunk-size
sweeps on the existing recursive chunker are part of this phase, not
just new chunkers.

| # | Component | Registry | Interface | Library |
|---|-----------|----------|-----------|---------|
| 5 | Semantic chunker | `semantic` | `BaseChunker` | `langchain-experimental` SemanticChunker |
| 6 | Contextual chunk enricher | `contextual` | `BaseChunkEnricher` | Custom (LLM prepends situating context per chunk) |

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

| # | Component | Location | Notes |
|---|-----------|----------|-------|
| 7 | ReAct agent loop | `pipeline/agent.py` | Reason → act (tool) → observe → repeat |
| 8 | RAG tool wrapper | `components/tools.py` | Wraps a QueryPipeline as a callable tool |
| 9 | Web search tool | `components/tools.py` | `BaseTool`; external search via API |

The agent treats retrieval as one tool among several, deciding for
itself when it has gathered enough evidence. The RAG tool wraps a
`QueryPipeline` (full hybrid + rerank stack) and produces a
`ToolResult` with `retrieved_chunks` populated, so the evaluator can
still assess retrieval quality in agent mode. The loop runs up to
`agent.max_iterations` steps. Chief risk to measure: error propagation
from a bad early step.

#### Phase D2 — Multi-Agent Supervisor

| # | Component | Location | Notes |
|---|-----------|----------|-------|
| 10 | Supervisor routing | `pipeline/agent.py` | LLM routes queries to specialized agents |
| 11 | Agent roster | `pipeline/agent.py` | Per-agent retrieval filters + prompts |

A supervisor LLM routes each query to a specialized agent; each agent
has its own retrieval filters and prompt (e.g., a handbook-only agent
vs. a deadlines agent). Resolves GAP 1.

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
| Chunking strategy | `indexing.chunking.type` | recursive done; **semantic, contextual** pending |
| Chunk size / overlap | `indexing.chunking.params` | done |
| Embedding model | `indexing.embedding` | done (4 providers) |
| Vectorstore | `indexing.vectorstore.type` | done (FAISS, Chroma) |

### 6.2 Query Variables

| Variable | Config Location | Status |
|----------|----------------|--------|
| Query transform | `query.query_transform.type` | passthrough, contextualizer done; **HyDE, multi-query** pending |
| Retrieval method | `query.retrieval.type` | dense done; **BM25, hybrid** pending |
| Fusion strategy | `query.retrieval.params.fusion` | **pending (Phase A)** |
| Retrieval depth | `query.retrieval.top_k_retrieve` | done |
| Final chunk count | `query.retrieval.top_k_final` | done |
| Reranker | `query.reranking.type` | none, cross_encoder done |
| Generator / LLM | `query.generation`, `query.generation_llm` | done (4 providers) |
| Prompt strategy | `query.prompt` | done (CoT, citation, context format) |

### 6.3 Agent Variables (Phase D)

| Variable | Config Location | Status |
|----------|----------------|--------|
| Pipeline mode | `pipeline_mode` | linear done; **agent** pending |
| Agent mode | `agent.mode` | **pending (single/multi)** |
| Max iterations | `agent.max_iterations` | **pending** |
| Tool roster | `agent.tools` | **pending** |
| Memory | `agent.memory.type` | none, buffer_window done (unused until agent) |

### 6.4 Evaluation Variables

| Variable | Config Location | Status |
|----------|----------------|--------|
| Dataset | `evaluation.dataset` | done |
| Mode | `evaluation.mode` | full, retrieval_only, no-score smoke done |
| Metrics | `evaluation.metrics` | done |
| Number of runs | `evaluation.num_runs` | done |
| Evaluator LLM | `evaluation.evaluator_llm` | done (standardized) |
| Evaluator embedding | `evaluation.evaluator_embedding` | done (fixed bge-m3) |
| Run config | `evaluation.run_config` | done |

---

## 7. Architectural Extensions Required

Two well-scoped changes unblock Phases A and C. Both are additive and
backward-compatible.

### 7.1 Auxiliary indexes in `IndexArtifact` (unblocks Phase A)

**Current:** `IndexArtifact` carries one `vectorstore` plus the
`embedder`. A hybrid/sparse retriever needs a BM25 index built and
cached alongside the vector index.

**Plan:** add an optional `auxiliary_stores: dict[str, Any]` field to
`IndexArtifact`. The indexing pipeline builds the BM25 index when a
sparse or hybrid retriever is configured and stores it under a known
key (e.g., `"bm25"`). The retriever reads it from there. The fingerprint
must incorporate sparse-index parameters so cache invalidation stays
correct. Dense-only experiments are unaffected — the dict is empty.

### 7.2 Chunk-enricher stage (unblocks contextual retrieval in Phase C)

**Current:** chunks are embedded in isolation; there is no place for a
post-chunking, pre-embedding transformation.

**Plan:** introduce a `BaseChunkEnricher` stage between chunking and
embedding in the indexing pipeline. Default is a no-op passthrough
(registered as `none`), so existing configs are unchanged. The
contextual enricher (Phase C) implements this interface. This is the
clean home for any future pre-embedding transform (late chunking,
proposition extraction, title-prepending).

### Lower-priority gaps (unchanged)

- **GAP — Embedder re-instantiation on cache load (LOW):** loading a
  cached index still instantiates the embedding model for query-time
  embedding. Performance only, no correctness issue. Could lazy-load
  `HuggingFaceEmbedder` if startup becomes a bottleneck.

---

## 8. Dependencies by Phase

### Current

```
pyyaml                    Config loading
faiss-cpu                 Vector store
numpy                     Array operations
sentence-transformers     Embeddings + cross-encoder reranking
pymupdf                   PDF parsing
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

### Phase A additions

```
bm25s                     Fast sparse BM25 retrieval (NumPy/SciPy)
PyStemmer                 Optional stemming for BM25 (recommended)
```

### Phase C additions

```
langchain-experimental    Semantic chunker
```

(HyDE, multi-query, hybrid fusion, the chunk enricher, and the agent
loop are custom — no new dependencies.)

---

## 9. Experiment Matrix

Experiments are organized by what each isolates. All inherit from
`base.yaml`; all formal runs use the standardized evaluator
(GPT-4o-mini via EdenAI, fixed bge-m3 evaluator embedding).

### Retrieval (Phase A) — the core matrix

| Experiment | Isolates | Compare against |
|-----------|----------|-----------------|
| `dense` (baseline) | dense retrieval | — |
| `bm25` | sparse retrieval | dense |
| `hybrid_rrf` | dense + sparse, RRF fusion | dense, bm25 |
| `hybrid_rrf_rerank` | hybrid + cross-encoder | hybrid_rrf |
| `hybrid_weighted` | weighted vs. RRF fusion | hybrid_rrf |

### Reranking (Phase 3C, done)

| Experiment | Isolates |
|-----------|----------|
| `cross_encoder` | reranking on/off |
| `cross_encoder_topk{3,5,10}` | final-chunk count under reranking |

### Query Optimization (Phase B)

| Experiment | Isolates |
|-----------|----------|
| `qt_hyde` | HyDE vs. passthrough |
| `qt_multi_query` | multi-query vs. passthrough |
| `qt_contextualizer` | existing contextualizer |

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
