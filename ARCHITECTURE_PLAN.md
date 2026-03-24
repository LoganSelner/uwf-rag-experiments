# Architecture Plan

> Forward-looking plan for the argobot-bench experimentation framework.
> For the system as currently built, see [ARCHITECTURE.md](ARCHITECTURE.md).
>
> **Last updated:** 2026-03-23

---

## 1. Goals

This framework exists to answer: **which combination of RAG
components produces the best academic advising chatbot?**

The concrete research objectives are:

1. **Replicate v1** (Retrieval-based ARGObot from the ACMSE paper)
   using identical components on our evaluation infrastructure,
   and reproduce the published RAGAS scores.

2. **Replicate v2** (Agent-based ARGObot from the ACMSE paper)
   with the single-agent ReAct architecture, and compare it
   against v1 under controlled conditions.

3. **Improve on both** by systematically testing alternative
   components (chunking strategies, embedding models, rerankers,
   query transformers, prompts) and identifying configurations
   that raise RAGAS metrics — particularly Context Entity Recall,
   which scored low (~0.27–0.29) across all published versions.

4. **Support v3** (Multi-agent from SURP 2025) as a future
   controlled comparison point.

---

## 2. Design Principles

These are documented in ARCHITECTURE.md and apply to all future
work. One additional principle governs Phase 3+ implementation:

### Framework Wrapper Strategy

**Use framework libraries when the component is a commodity.
Write custom when the component IS the experiment.**

Chunking, PDF parsing, embedding, and reranking are commodities —
use proven libraries (`langchain-text-splitters`,
`sentence-transformers`, `rank-bm25`). Novel chunking strategies
tuned for policy handbooks, or domain-specific query transformers,
are research contributions — implement those directly.

**Never let framework dependencies leak past the interface
boundary.** A LangChain-backed chunker imports
`RecursiveCharacterTextSplitter` inside its class, but produces
`list[Chunk]` — our type. The pipeline never touches LangChain
types. If the library is swapped, nothing changes outside that
one file.

This is already the pattern for embeddings (`sentence-transformers`
behind `BaseEmbedder`), vectorstore (`faiss-cpu` behind
`BaseVectorStore`), and PDF parsing (`pymupdf` behind
`BaseIngestor`). Extend it consistently.

---

## 3. What Exists Today

### Completed (Phases 1–2)

| Category | Component | Registry Name | Notes |
|----------|-----------|---------------|-------|
| Ingest | `PDFIngestor` | `pdf` | PyMuPDF |
| Chunking | `RecursiveChunker` | `recursive_custom` | Custom 131-line implementation |
| Chunking | `LangChainRecursiveChunker` | `recursive_langchain` | `langchain-text-splitters` wrapper |
| Embedding | `HuggingFaceEmbedder` | `huggingface` | sentence-transformers, bge-m3 default |
| Embedding | `GoogleEmbedder` | `google` | `google-genai` SDK, gemini-embedding-001 default |
| Vectorstore | `FAISSVectorStore` | `faiss` | Cosine + L2, metadata post-filtering |
| Vectorstore | `ChromaVectorStore` | `chroma` | ChromaDB, native metadata filtering |
| Retrieval | `DenseRetriever` | `dense` | Single-vector similarity search |
| Generation | `OllamaGenerator` | `ollama` | Local LLM via Ollama |
| Generation | `EdenAIGenerator` | `edenai` | Cloud LLM via Eden AI gateway |
| Generation | `GoogleGenerator` | `google` | Gemini via `google-genai` SDK |
| Prompts | `ChatPromptTemplate` | `chat` | Numbered/plain context, CoT, citations |
| Query Transform | `PassthroughQueryTransformer` | `passthrough` | Returns query unchanged |
| Query Transform | `ContextualizerQueryTransformer` | `contextualizer` | LLM reformulation with history |
| Reranking | `NoOpReranker` | `none` | Passes through, truncates to top_k |
| Memory | `NoMemory` | `none` | No-op |
| Memory | `BufferWindowMemory` | `buffer_window` | Last N turns |

### Infrastructure Completed

- Config system with YAML inheritance, cycle detection, index
  fingerprinting, and upfront validation
- Linear query pipeline with prompt → generator wiring
- Index caching by SHA-256 fingerprint
- RAGAS evaluation with multi-run aggregation (mean ± std)
- Configurable RAGAS execution settings (`EvalRunConfig`:
  timeout, retries, workers)
- Experiment result saving (summary.json + config.yaml + JSONL)
- Cross-experiment comparison with config diffing
- CI pipeline (ruff + mypy + pytest), pre-commit, Makefile
- Agent pipeline stub (raises `NotImplementedError`)
- Git SHA tracking for reproducibility
- Centralized `.env` loading at application boundary
  (`scripts/run_experiment.py`)

---

## 4. ARGObot Version Specifications

Extracted from the ACMSE 2025 paper (DOI: 10.1145/3696673.3723065)
and the SURP 2025 poster. These define the target configurations
for replication experiments.

### v1 — Retrieval-based (ACMSE paper, Section 3.1)

| Component | Specification |
|-----------|--------------|
| LLM | Gemini 1.0 Pro |
| Embedding | Google Generative AI (models/embedding-001) |
| Vectorstore | ChromaDB |
| Chunking | 1,200 tokens, 200 overlap |
| Retrieval | Top-3 chunks |
| Architecture | Linear RAG chain |
| Query handling | Contextualization chain reformulates query using chat history before retrieval |
| Response strategy | Verbatim quotes with citations; responds "I don't know" when context insufficient |
| Framework | LangChain, deployed on HuggingFace via Streamlit |

Published scores (18q dataset, 3 runs):

| Ans.Corr | Ctx.Prec | Faith. | CER | Ans.Rel |
|----------|----------|--------|-------|---------|
| 0.815 | 0.883 | 0.906 | 0.291 | 0.867 |

### v2 — Single-Agent (ACMSE paper, Section 3.2)

| Component | Specification |
|-----------|--------------|
| LLM | GPT-4 |
| Embedding | text-embedding-ada-002 (OpenAI) |
| Vectorstore | ChromaDB |
| Chunking | Same document, same handbook source |
| Agent | Single ReAct agent choosing between 3 tools |
| Tool: RAG | Retrieves from Student Handbook vectorstore |
| Tool: Search | Google Search via Serper API; disclaims accuracy |
| Tool: Email | Gmail API to schedule advisor appointment (HITL) |
| Memory | Conversation Buffered Window, 5 turns |
| Prompt | Role assignment, Chain-of-Thought prompting |

Published scores (18q dataset, 3 runs):

| Ans.Corr | Ctx.Prec | Faith. | CER | Ans.Rel |
|----------|----------|--------|-------|---------|
| 0.878 | 0.937 | 0.927 | 0.272 | 0.894 |

### v3 — Multi-Agent (SURP 2025 poster)

| Component | Specification |
|-----------|--------------|
| LLM | Qwen2.5 32B |
| Embedding | bge-m3:567m |
| Vectorstore | FAISS |
| Chunking | 1,000 tokens, 100 overlap, k=3 |
| Architecture | Multi-agent with Supervisor routing |
| Agents | Handbook_agent, Knowledge_Base_agent (source-filtered) |
| Tools | EmailSender, WebSearcher |
| Integration | Qwen2.5 integration count: 3 |

Poster scores (19q dataset):

| Ans.Corr | Ctx.Prec | Faith. | CER | Ans.Rel |
|----------|----------|--------|-------|---------|
| 0.3111 | 0.1316 | 0.7632 | 0.2963 | 0.2437 |

Note: MA model scored considerably lower. The poster identifies
this as an incomplete comparison — modifications to the SA model
and further evaluation are planned.

---

## 5. Replication Configs

These YAML configs recreate each version using our framework.
Components marked *(Phase 3B)* or *(Phase 4A)* require
implementation in the corresponding phase. All v1 components
are implemented.

### v1 Config

All components available now. Config at
`configs/experiments/v1_replication.yaml`.

```yaml
pipeline_mode: "linear"
indexing:
  sources:
    - name: "student_handbook"
      path: "data/sources/student_handbook.pdf"
      ingest: { type: "pdf" }
  chunking:
    type: "recursive_langchain"
    params: { chunk_size: 1200, chunk_overlap: 200 }
  embedding:
    type: "google"
    params: { model_name: "models/embedding-001" }
  vectorstore:
    type: "chroma"
query:
  query_transform:
    type: "contextualizer"
    params:
      generator_type: "google"
      llm: { model_name: "gemini-1.0-pro", temperature: 0.0 }
  retrieval:
    type: "dense"
    top_k_retrieve: 3
    top_k_final: 3
  generation:
    type: "google"
  generation_llm:
    provider: "google"
    model_name: "gemini-1.0-pro"
  prompt:
    type: "chat"
    system_template: |
      Answer using ONLY verbatim quotes from the provided context.
      Include the citation source for each quote.
      If the context does not contain the answer, say "I don't know".
    citation_style: "verbatim"
evaluation:
  dataset: "data/datasets/18q_handbook.jsonl"
```

### v2 Config

Requires Phase 3B (OpenAI/EdenAI embedder, tools) and
Phase 4A (agent pipeline).

```yaml
pipeline_mode: "agent"
indexing:
  sources:
    - name: "student_handbook"
      path: "data/sources/student_handbook.pdf"
      ingest: { type: "pdf" }
  chunking:
    type: "recursive_langchain"
    params: { chunk_size: 1200, chunk_overlap: 200 }
  embedding:
    type: "openai"                             # (Phase 3B)
    params: { model_name: "text-embedding-ada-002" }
    # Alternative for EdenAI users:
    # type: "edenai"                           # (Phase 3B)
    # params: { provider: "openai" }
  vectorstore:
    type: "chroma"
agent:
  mode: "single"
  llm:
    provider: "openai"
    model_name: "gpt-4"
  memory:
    type: "buffer_window"
    window_size: 5
  agents:
    - name: "rag_handbook"
      type: "rag"
      retrieval: { type: "dense", top_k_retrieve: 3, top_k_final: 3 }
      prompt:
        type: "chat"
        use_chain_of_thought: true
  tools:
    - { name: "web_search", type: "tool", tool: "serper_search" }  # (Phase 3B)
    - { name: "email", type: "tool", tool: "gmail_mock" }          # (Phase 3B)
evaluation:
  dataset: "data/datasets/18q_handbook.jsonl"
```

### Controlled Comparison

The key insight: v1 and v2 use different LLMs and embeddings.
To isolate the effect of the agent architecture, create a
variant where both use the same indexing:

```yaml
# v1_controlled.yaml — linear, GPT-4, ada-002
pipeline_mode: "linear"
# (same indexing as v2)
query:
  generation_llm: { provider: "openai", model_name: "gpt-4" }

# v2_controlled.yaml — agent, GPT-4, ada-002
pipeline_mode: "agent"
# (same indexing as v1_controlled)
agent: { mode: "single", ... }
```

Both configs produce the same index fingerprint. The only
variable is `pipeline_mode` and the agent config.

---

## 6. Implementation Phases

Each phase unlocks a meaningful set of experiments that couldn't
be run before. Phases are ordered by research value.

### Housekeeping (before Phase 3) — COMPLETED

- [x] Add `data/sources/README.md` and `data/datasets/README.md`
      explaining expected file formats and placement
- [x] Update `.gitignore` to track directory READMEs while
      ignoring contents
- [x] Evaluated moving presentation concerns from
      `src/evaluation/comparison.py` to `scripts/compare.py`;
      decided to keep current separation — the library module
      provides both data access and formatting for multiple
      consumers (CLI, notebooks, future tools)
- [x] Add LangChain-backed chunker as `"recursive_langchain"`;
      custom kept as `"recursive_custom"` for A/B comparison

### Phase 3A — v1 Replication Components

Goal: Run the v1 configuration on our linear pipeline and
reproduce the paper's RAGAS scores.

| # | Component | Registry | Interface | Library | Status |
|---|-----------|----------|-----------|---------|--------|
| 1 | ChromaDB vectorstore | `chroma` | `BaseVectorStore` | `chromadb` | Done |
| 2 | Google AI embeddings | `google` | `BaseEmbedder` | `google-genai` | Done |
| 3 | Gemini generator | `google` | `BaseGenerator` | `google-genai` | Done |
| 4 | Contextualizing query transformer | `contextualizer` | `BaseQueryTransformer` | Custom (LLM call) | Done |
| 5 | LangChain recursive chunker | `recursive_langchain` | `BaseChunker` | `langchain-text-splitters` | Done |

**Milestone:** Can run `configs/experiments/v1_replication.yaml`
and compare scores against the paper's Table 2.

### Evaluator Improvements (before Phase 3B) — COMPLETED

Goal: Complete the evaluator's provider support and add a
dedicated evaluation embedder so cross-experiment comparisons
use a consistent measurement instrument.

| # | Item | Details | Status |
|---|------|---------|--------|
| E1 | Add `"google"` provider to `_build_evaluator_llm()` | Uses `langchain-google-genai` `ChatGoogleGenerativeAI`. Completes the Google stack — experiments can run 100% Google with no Ollama dependency. | Done |
| E2 | Add required `evaluator_embedding` config to `EvaluationConfig` | A `ComponentConfig` block (type + params) that builds a dedicated embedder for RAGAS via the registry. Decouples evaluation embeddings from pipeline embeddings. | Done |
| E3 | Build evaluator embedder from registry in `Evaluator` | Replace the current `embedder=` passthrough with registry-based construction from `evaluator_embedding` config. The evaluator builds its own embedder instance, independent of the pipeline. | Done |
| E4 | Add evaluator provider validation to `validate_config()` | Check that `evaluator_llm.provider` is one of the supported values (`"ollama"`, `"edenai"`, `"google"`) and that `evaluator_embedding.type` is registered. | Done |

**Why a required evaluator embedder:** RAGAS uses embeddings for
metrics like `answer_similarity` and `answer_relevancy`. When
comparing experiments that use different pipeline embedders
(bge-m3 vs ada-002 vs Google embeddings), a fixed evaluation
embedder eliminates a confounding variable. Every experiment is
measured with the same yardstick.

**Config shape:**

```yaml
evaluation:
  evaluator_llm:
    provider: "ollama"
    model_name: "qwen3:32b"
  evaluator_embedding:
    type: "huggingface"
    params:
      model_name: "BAAI/bge-m3"
      normalize: true
```

**Milestone:** All current providers (Ollama, EdenAI, Google)
are supported for evaluation LLM. Evaluation embeddings are
explicitly configured and consistent across experiments.

### Phase 3B — v2 Component Prerequisites

Goal: Build the tools and model integrations v2 needs, testable
independently before the agent loop is implemented.

| # | Component | Registry | Interface | Library |
|---|-----------|----------|-----------|---------|
| 6 | OpenAI embeddings | `openai` | `BaseEmbedder` | `openai` |
| 7 | EdenAI embeddings | `edenai` | `BaseEmbedder` | `langchain-community` `EdenAiEmbeddings` |
| 8 | OpenAI generator | `openai` | `BaseGenerator` | `openai` |
| 9 | Serper web search tool | `serper_search` | `BaseTool` | `requests` |
| 10 | Gmail mock tool | `gmail_mock` | `BaseTool` | Custom (logs, no real send) |

The EdenAI embedder provides access to OpenAI embeddings
(and other providers) through the EdenAI gateway. Users
with direct OpenAI API keys use the `"openai"` embedder;
users with only an EdenAI key use `"edenai"` with
`provider: "openai"` to get the same ada-002 vectors.

**Milestone:** All v2 model/tool components are registered,
tested, and usable in linear pipeline experiments. OpenAI
embeddings + generator can run via the linear pipeline for
standalone quality benchmarking.

### Phase 3C — Experimentation Components

Goal: Add the knobs that let us try to beat the paper's scores.
All work within the existing linear pipeline.

| # | Component | Registry | Interface | Library |
|---|-----------|----------|-----------|---------|
| 11 | Cross-encoder reranker | `cross_encoder` | `BaseReranker` | `sentence-transformers` CrossEncoder |
| 12 | HyDE query transformer | `hyde` | `BaseQueryTransformer` | Custom (LLM generates hypothetical doc) |
| 13 | Multi-query transformer | `multi_query` | `BaseQueryTransformer` | Custom (LLM generates N query variants) |
| 14 | Semantic chunker | `semantic` | `BaseChunker` | `langchain-experimental` SemanticChunker |

**Milestone:** Can run a matrix of experiments: baseline ×
{with/without reranker} × {passthrough/HyDE/multi-query} ×
{recursive/semantic chunking} and compare all results in one
table.

### Phase 4A — Single-Agent Pipeline

Goal: Implement the ReAct loop in `AgentPipeline` for
`mode: "single"`. Replicate v2.

| # | Component | Location | Notes |
|---|-----------|----------|-------|
| 15 | ReAct agent loop | `pipeline/agent.py` | Single LLM decides tool → executes → observes → decides again |
| 16 | RAG tool wrapper | `pipeline/agent.py` or `components/tools.py` | Wraps a mini QueryPipeline as a tool |

The agent receives a query, decides which tool to use (RAG,
web search, email), executes it, observes the result, and either
calls another tool or formulates a final answer. The loop runs
for up to `agent.supervisor.max_iterations` steps.

The RAG tool wraps a `QueryPipeline` instance with the agent's
per-agent retrieval config and prompt. It produces a `ToolResult`
with `retrieved_chunks` populated — these flow into the final
`GenerationResult` so the evaluator can assess retrieval quality
even in agent mode.

**Milestone:** Can run `configs/experiments/v2_replication.yaml`
and compare agent vs linear performance with identical indexing.

### Phase 4B — Multi-Agent Supervisor

Goal: Implement supervisor routing for `mode: "multi"`.
Support v3 comparison.

| # | Component | Location | Notes |
|---|-----------|----------|-------|
| 17 | Supervisor routing | `pipeline/agent.py` | LLM-based routing to specialized agents |
| 18 | Agent roster management | `pipeline/agent.py` | Per-agent QueryPipelines with source filters |

The supervisor uses its own LLM (`agent.supervisor.llm`) to
decide which agent handles each query. Each agent has its own
retrieval filters and prompt template, enabling source-specialized
responses (e.g., handbook_agent only searches the handbook
vectorstore partition).

**Milestone:** Can run controlled v1 vs v2 vs v3 comparison
with identical indexing.

### Phase 5 — Advanced

Goal: Research extensions beyond replication.

| # | Component | Notes |
|---|-----------|-------|
| 19 | BM25 retriever | `rank-bm25`, for hybrid dense+sparse retrieval |
| 20 | Hybrid retriever | Combines dense + BM25 with score fusion |
| 21 | ColBERT reranker | Late interaction model |
| 22 | Multi-turn session evaluator | Exercises memory across ordered turn sequences |
| 23 | Additional embedding models | nomic-embed, GTE, OpenAI ada-002 variants |

---

## 7. Variable Coverage Matrix

Every testable variable from the research factor analysis, mapped
to its config location. All are independently isolatable.

### 7.1 Indexing Variables

| Variable | Config Location |
|----------|----------------|
| Document parser | `indexing.sources[].ingest.type` |
| Number of sources | `indexing.sources` list length |
| Source selection | `indexing.sources` list contents |
| Chunking strategy | `indexing.chunking.type` |
| Chunk size | `indexing.chunking.params.chunk_size` |
| Chunk overlap | `indexing.chunking.params.chunk_overlap` |
| Separators | `indexing.chunking.params.separators` |
| Embedding model | `indexing.embedding.params.model_name` |
| Embedding normalization | `indexing.embedding.params.normalize` |
| Vectorstore backend | `indexing.vectorstore.type` |
| Distance metric | `indexing.vectorstore.params.metric` |

### 7.2 Query Variables

| Variable | Config Location |
|----------|----------------|
| Query transform technique | `query.query_transform.type` |
| Retrieval method | `query.retrieval.type` |
| top_k_retrieve | `query.retrieval.top_k_retrieve` |
| top_k_final | `query.retrieval.top_k_final` |
| Source filters | `query.retrieval.filters` |
| Reranking method | `query.reranking.type` |
| Reranking model | `query.reranking.params.model_name` |
| Generator backend | `query.generation.type` |
| LLM model | `query.generation_llm.model_name` |
| LLM temperature | `query.generation_llm.temperature` |
| LLM max tokens | `query.generation_llm.max_tokens` |
| System prompt text | `query.prompt.system_template` |
| Chain-of-thought | `query.prompt.use_chain_of_thought` |
| Citation style | `query.prompt.citation_style` |
| Few-shot examples | `query.prompt.few_shot_examples` |
| Context format | `query.prompt.context_format` |

### 7.3 Agent Variables

| Variable | Config Location |
|----------|----------------|
| Pipeline mode | `pipeline_mode` |
| Agent mode | `agent.mode` |
| Supervisor LLM | `agent.supervisor.llm` |
| Routing strategy | `agent.supervisor.routing` |
| Max iterations | `agent.supervisor.max_iterations` |
| Memory type | `agent.memory.type` |
| Memory window size | `agent.memory.window_size` |
| Agent roster | `agent.agents` list |
| Per-agent source filter | `agent.agents[].retrieval.filters` |
| Per-agent prompt | `agent.agents[].prompt` |
| Available tools | `agent.tools` list |

### 7.4 Evaluation Variables

| Variable | Config Location |
|----------|----------------|
| Dataset | `evaluation.dataset` |
| Evaluation mode | `evaluation.mode` |
| Metric selection | `evaluation.metrics` |
| Number of runs | `evaluation.num_runs` |
| Evaluator LLM | `evaluation.evaluator_llm` |
| Evaluator embedding | `evaluation.evaluator_embedding` |
| Eval timeout | `evaluation.run_config.timeout` |
| Eval workers | `evaluation.run_config.max_workers` |

---

## 8. Known Gaps

### RESOLVED: Config validation

Previously GAP 4. Now implemented as `validate_config()` in
`core/config.py`, called in `scripts/run_experiment.py` before
pipeline construction.

### GAP 1: Agent supervisor lacks its own generator (MEDIUM)

**Problem:** The AgentPipeline is a stub. The supervisor needs to
make LLM calls for routing and synthesis.

**Plan:** Phase 4A (single-agent) and 4B (multi-agent) implement
the real loops. The config already carries `agent.supervisor.llm`.

### GAP 2: Hybrid retrieval has no BM25 index cache (LOW)

**Problem:** A hybrid retriever needs a keyword index alongside
the vector index. The current `IndexArtifact` carries only one
vectorstore.

**Plan:** When hybrid retrieval is implemented (Phase 5), extend
`IndexArtifact` with an optional `auxiliary_stores` dict. The
hybrid retriever stores/loads its BM25 index in the same cache
directory.

### GAP 3: No multi-turn session evaluation (LOW)

**Problem:** The evaluator runs each query independently. Memory
exists in the agent pipeline but the evaluator doesn't exercise it.

**Plan:** Add a `SessionEvaluator` mode (Phase 5) when multi-turn
datasets are available.

### GAP 4: Contextual chunking has no independent stage (LOW)

**Problem:** Contextual chunking (prepending section titles before
embedding) is a post-chunking, pre-embedding transformation.
Currently baked into chunker implementations.

**Plan:** If this becomes a priority, add a `BaseChunkEnricher`
stage between chunking and embedding. For now, chunker
implementations can include it as a boolean param.

### GAP 5: Embedder re-instantiation on cache load (LOW)

**Problem:** Loading an index from cache still instantiates the
full embedding model (needed for `embed_query()` at query time).
For large models like bge-m3, this takes several seconds even
when the index itself loads instantly.

**Impact:** Performance cost only — no correctness issue. The
model is needed regardless; the question is whether it could be
lazy-loaded.

**Plan:** Acceptable for now. Could add lazy loading to
`HuggingFaceEmbedder` if startup time becomes a bottleneck
during rapid iteration.

---

## 9. Dependencies by Phase

### Current (Phases 1–3A)

```
pyyaml                    Config loading
faiss-cpu                 Vector store
numpy                     Array operations
sentence-transformers     Embedding models
pymupdf                   PDF parsing
ragas                     Evaluation metrics (pinned 0.4.x)
datasets                  RAGAS transitive dependency
python-dotenv             API key management
rich                      CLI table rendering
langchain-community       EdenAI LLM/embeddings wrapper
langchain-ollama          Ollama LLM wrapper
tenacity                  Retry logic for API calls
chromadb                  ChromaDB vectorstore
google-genai              Gemini generation + Google embeddings
langchain-text-splitters  LangChain recursive chunker
```

### Evaluator improvements additions

```
langchain-google-genai    Google evaluator LLM (ChatGoogleGenerativeAI)
```

### Phase 3B additions

```
openai                    OpenAI embeddings + generation
```

### Phase 3C additions

```
langchain-experimental    Semantic chunker
```

### Phase 5 additions

```
rank-bm25                 BM25 retrieval for hybrid search
```

---

## 10. Experiment Matrix

The core experiments, organized by what each isolates.

### Replication Experiments

| Experiment | What It Tests | Requires |
|-----------|---------------|----------|
| `v1_replication` | Reproduce paper Table 2, R-ARGObot row | Phase 3A |
| `v2_replication` | Reproduce paper Table 2, A-ARGObot row | Phase 4A |
| `v1_vs_v2_controlled` | Agent vs linear with identical indexing | Phase 4A |

### Component Isolation Experiments

| Experiment | Variable Isolated | Baseline |
|-----------|-------------------|----------|
| `chunking_256` / `_512` / `_1000` / `_1200` | Chunk size | `baseline` |
| `overlap_0` / `_50` / `_100` / `_200` | Chunk overlap | `baseline` |
| `embedding_bge_m3` / `_ada002` / `_nomic` | Embedding model | `baseline` |
| `vectorstore_faiss` / `_chroma` | Vectorstore backend | `baseline` |
| `reranker_none` / `_cross_encoder` | Reranking | `baseline` |
| `qt_passthrough` / `_hyde` / `_multi_query` | Query transformation | `baseline` |
| `prompt_verbatim` / `_cot` / `_numbered` | Prompt strategy | `baseline` |
| `chunker_recursive` / `_semantic` | Chunking strategy | `baseline` |

### CER-Focused Experiments

Context Entity Recall is the weakest metric across all published
versions (~0.27–0.29). The following experiments specifically
target CER improvement:

| Experiment | Hypothesis |
|-----------|-----------|
| Smaller chunks (256) | More focused chunks improve entity matching |
| Larger top_k (10, 15) | More retrieved chunks cover more entities |
| Cross-encoder reranking | Better-ranked chunks contain more relevant entities |
| Semantic chunking | Entity-preserving boundaries improve recall |
| Different embedding model | Better semantic alignment captures entities |
