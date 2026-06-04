"""Tool implementations for agent pipelines.

Tools are the actions an agent pipeline can take. Each wraps a capability behind
``BaseTool.execute(query) -> ToolResult`` and is advertised to the tool-calling
LLM via a uniform single-``query``-string schema (see :func:`tool_to_spec`).

Implemented:
- ``rag`` — searches the indexed knowledge base, returning the raw retrieved
  chunks as the observation so the agent's own LLM synthesizes the final answer.
- ``web_search`` — searches the public web (Tavily, or keyless DuckDuckGo); a
  first-class tool kept out of the core reproducible matrix as its own arm.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from ragbench.components.base import BaseTool, ComponentParams
from ragbench.core.config import QueryTransformConfig
from ragbench.core.registry import registry
from ragbench.core.types import RetrievedChunk, ToolResult, ToolSpec

if TYPE_CHECKING:
    from ragbench.components.build import BuildContext
    from ragbench.core.config import RetrievalConfig, ToolConfig, ValidateContext

logger = logging.getLogger(__name__)

_DEFAULT_RAG_DESCRIPTION = (
    "Search the knowledge base of indexed source documents for passages "
    "relevant to a question. Use this whenever you need factual information "
    "from the documents to answer. Input: a natural-language search query."
)


def tool_to_spec(tool: BaseTool, description: str | None = None) -> ToolSpec:
    """Advertise a tool to a tool-calling LLM with a single ``query`` arg.

    Every tool in this harness takes one natural-language query string
    (``execute(query)``), so the JSON schema is uniform; only the name and
    description vary. Keeps JSON-schema authoring out of ``BaseTool``.
    """
    return ToolSpec(
        name=tool.name,
        description=tool.description if description is None else description,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language search query.",
                }
            },
            "required": ["query"],
        },
    )


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered observation for the agent."""
    if not chunks:
        return "No relevant passages were found in the knowledge base."
    return "\n\n".join(f"[{i}] {rc.chunk.content}" for i, rc in enumerate(chunks, 1))


@registry.register("tool", "rag")
class RAGSearchTool(BaseTool):
    """Searches the indexed knowledge base via a retrieval-only QueryPipeline.

    Wraps the experiment's query retrieval + rerank stack so the agent searches
    the *same* index and stack as the linear baseline — keeping linear-vs-agent
    comparable. Returns the top chunks verbatim as the observation and populates
    ``ToolResult.retrieved_chunks`` so the evaluator can still score retrieval in
    agent mode.

    The query transform is forced to ``passthrough``: the agent has already
    decided what to search for, so re-running the linear stack's transform
    (contextualizer/HyDE/...) inside the tool would double-transform and add a
    hidden LLM call. The wrapped pipeline is injected (duck-typed: anything with
    ``run(query) -> GenerationResult``), so this component never imports the
    pipeline layer at its interface.
    """

    def __init__(
        self,
        pipeline: Any,
        name: str = "knowledge_base",
        description: str = _DEFAULT_RAG_DESCRIPTION,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._name = name
        self._description = description

    @classmethod
    def build(cls, cfg: ToolConfig, ctx: BuildContext) -> RAGSearchTool:
        """Build a retrieval-only QueryPipeline from the active query stack.

        Reuses ``ctx.query`` (the experiment's retrieval + rerank stack) over
        ``ctx.index`` in retrieval-only mode, with the query transform forced to
        passthrough — the agent has already decided what to search for.

        ``cfg.params`` may carry a per-tool retrieval-scope override —
        ``filters`` (merged over the stack's defaults), ``top_k_retrieve``, and
        ``top_k_final`` — so one agent can hold several ``rag`` tools each scoped
        to a different source (the ``agent_multitool`` rung), and a D2
        specialist's source-scoped tool reuses the same path.

        ``QueryPipeline`` is imported lazily so this component module never
        depends on the pipeline layer at import time (keeping the dependency
        graph pointing inward).
        """
        from ragbench.pipeline.query import QueryPipeline

        if ctx.query is None or ctx.index is None:
            raise RuntimeError(
                "RAGSearchTool.build requires both ctx.query (the query stack) "
                "and ctx.index (the populated index) in the BuildContext."
            )
        retrieval = cls._scoped_retrieval(ctx.query.retrieval, cfg.params)
        query_config = ctx.query.model_copy(
            update={
                "query_transform": QueryTransformConfig(type="passthrough"),
                "retrieval": retrieval,
            },
        )
        pipeline = QueryPipeline.from_config(
            query_config, ctx.index, retrieval_only=True
        )
        return cls(
            pipeline=pipeline,
            name=cfg.name or "knowledge_base",
            description=cfg.description or _DEFAULT_RAG_DESCRIPTION,
        )

    @staticmethod
    def _scoped_retrieval(
        base: RetrievalConfig, params: dict[str, Any]
    ) -> RetrievalConfig:
        """Apply an optional per-tool retrieval-scope override to the stack config.

        ``params`` may carry ``filters`` (merged over the stack's defaults so a
        config-level filter and a per-tool scope compose, mirroring the
        retrievers' own ``{**default, **call}`` merge), ``top_k_retrieve``, and
        ``top_k_final``. Absent keys leave the base config untouched.
        """
        update: dict[str, Any] = {}
        extra_filters = params.get("filters")
        if extra_filters:
            update["filters"] = {**base.filters, **extra_filters}
        if "top_k_retrieve" in params:
            update["top_k_retrieve"] = params["top_k_retrieve"]
        if "top_k_final" in params:
            update["top_k_final"] = params["top_k_final"]
        return base.model_copy(update=update) if update else base

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, query: str) -> ToolResult:
        try:
            result = self._pipeline.run(query)
        except Exception as e:
            # A tool failure must become a recoverable observation, not a crash:
            # the agent can re-plan rather than the whole query dying.
            logger.warning(
                "RAG tool '%s' failed for query %r: %s", self._name, query, e
            )
            return ToolResult(
                tool_name=self._name,
                content=f"Knowledge base search failed: {e}",
                success=False,
                retrieved_chunks=[],
            )
        chunks = result.retrieved_chunks
        return ToolResult(
            tool_name=self._name,
            content=_format_chunks(chunks),
            success=True,
            retrieved_chunks=chunks,
            metadata={"num_chunks": len(chunks)},
        )


_DEFAULT_WEBSEARCH_DESCRIPTION = (
    "Search the public web for information not contained in the knowledge base "
    "(current events, external facts, anything outside the indexed documents). "
    "Prefer the knowledge base for in-corpus questions. Input: a "
    "natural-language search query."
)

_SUPPORTED_WEBSEARCH_PROVIDERS: tuple[str, ...] = ("tavily", "duckduckgo")


def _format_web_results(results: list[dict[str, str]]) -> str:
    """Render web results as a numbered observation for the agent."""
    if not results:
        return "No relevant web results were found."
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("url") or "result"
        body = "\n".join(
            part for part in (r.get("content", ""), r.get("url", "")) if part
        )
        blocks.append(f"[{i}] {title}\n{body}".strip())
    return "\n\n".join(blocks)


@registry.register("tool", "web_search")
class WebSearchTool(BaseTool):
    """Searches the public web via Tavily (default) or keyless DuckDuckGo.

    A first-class agent tool, but deliberately kept OUT of the core reproducible
    experiment matrix: live web results vary over time and inject open-web
    knowledge into a closed-domain evaluation, so web-search configs are run as
    their own ablation arm. Web hits are not corpus chunks, so
    ``retrieved_chunks`` stays empty — they never enter the RAGAS
    context-precision denominator (keeping the web arm's retrieval metrics
    honest).

    Providers (``params.provider``):
    - ``tavily`` (default): LLM-native results; needs an API key named by
      ``params.api_key_env`` (default ``TAVILY_API_KEY``).
    - ``duckduckgo``: keyless (DuckDuckGo Instant Answer API), for free smoke
      runs — no dependency, no key.

    Both providers use stdlib HTTP (mirroring the Ollama generator), so the tool
    adds no new dependency.
    """

    class Params(ComponentParams):
        provider: str = "tavily"
        max_results: int = 5
        api_key_env: str = "TAVILY_API_KEY"
        timeout: float = 15.0

    def __init__(
        self,
        params: WebSearchTool.Params,
        name: str = "web_search",
        description: str = _DEFAULT_WEBSEARCH_DESCRIPTION,
    ) -> None:
        super().__init__()
        self._p = params
        self._name = name
        self._description = description

    @classmethod
    def build(cls, cfg: ToolConfig, ctx: BuildContext) -> WebSearchTool:
        return cls(
            params=cls.Params.model_validate(cfg.params),
            name=cfg.name or "web_search",
            description=cfg.description or _DEFAULT_WEBSEARCH_DESCRIPTION,
        )

    @classmethod
    def validate_params(cls, params: dict[str, Any], ctx: ValidateContext) -> list[str]:
        try:
            p = cls.Params.model_validate(params)
        except ValidationError as e:
            return [f"web_search tool params are invalid: {e}"]
        errors: list[str] = []
        if p.provider not in _SUPPORTED_WEBSEARCH_PROVIDERS:
            errors.append(
                f"web_search params.provider: '{p.provider}' is not supported. "
                f"Supported: {list(_SUPPORTED_WEBSEARCH_PROVIDERS)}"
            )
        if p.max_results <= 0:
            errors.append(
                f"web_search params.max_results must be > 0 (got {p.max_results})"
            )
        return errors

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, query: str) -> ToolResult:
        try:
            if self._p.provider == "tavily":
                results = self._search_tavily(query)
            elif self._p.provider == "duckduckgo":
                results = self._search_duckduckgo(query)
            else:
                raise ValueError(f"Unknown web_search provider '{self._p.provider}'")
        except Exception as e:
            # A tool failure becomes a recoverable observation, not a crash.
            logger.warning(
                "Web search '%s' failed for query %r: %s", self._name, query, e
            )
            return ToolResult(
                tool_name=self._name,
                content=f"Web search failed: {e}",
                success=False,
                retrieved_chunks=[],
            )
        return ToolResult(
            tool_name=self._name,
            content=_format_web_results(results),
            success=True,
            retrieved_chunks=[],  # web hits are not corpus chunks (eval-safe)
            metadata={"provider": self._p.provider, "num_results": len(results)},
        )

    def _search_tavily(self, query: str) -> list[dict[str, str]]:
        api_key = os.environ.get(self._p.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"{self._p.api_key_env} is not set — required for the Tavily "
                "web_search provider. Set it in .env or use provider "
                "'duckduckgo' (keyless) for smoke runs."
            )
        payload = json.dumps(
            {"query": query, "max_results": self._p.max_results}
        ).encode()
        req = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urlopen(req, timeout=self._p.timeout) as resp:
            body = json.loads(resp.read().decode())
        return [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "content": str(r.get("content", "")),
            }
            for r in body.get("results", [])[: self._p.max_results]
        ]

    def _search_duckduckgo(self, query: str) -> list[dict[str, str]]:
        params = urlencode(
            {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        )
        req = Request(
            f"https://api.duckduckgo.com/?{params}",
            headers={"User-Agent": "ragbench-websearch/1.0"},
            method="GET",
        )
        with urlopen(req, timeout=self._p.timeout) as resp:
            body = json.loads(resp.read().decode())
        out: list[dict[str, str]] = []
        if body.get("AbstractText"):
            out.append(
                {
                    "title": str(body.get("Heading", "")),
                    "url": str(body.get("AbstractURL", "")),
                    "content": str(body["AbstractText"]),
                }
            )
        for topic in body.get("RelatedTopics", []):
            if len(out) >= self._p.max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                out.append(
                    {
                        "title": "",
                        "url": str(topic.get("FirstURL", "")),
                        "content": str(topic["Text"]),
                    }
                )
        return out


class SubAgentTool(BaseTool):
    """Wraps a specialist agent pipeline as a tool for a multi-agent supervisor.

    The supervisor (an ``AgentPipeline`` running the ordinary ReAct loop) reasons
    over specialists the same way it would over plain tools: each specialist is
    advertised via its ``name``/``description`` (so the description IS the
    supervisor's routing basis) and invoked with a single query string.
    ``execute`` runs the specialist and returns its synthesized ``answer`` as the
    observation, while surfacing the specialist's ``retrieved_chunks`` upward so
    the supervisor's chunk aggregation (union → dedup → cap at ``top_k_final``)
    keeps agent-mode retrieval metrics comparable to the linear baseline.

    The specialist is **injected** (duck-typed: anything with
    ``run(query) -> GenerationResult``) and is constructed by the pipeline layer,
    so this component imports nothing from ``pipeline`` and is never built through
    the tool registry — hence :meth:`build` is unsupported.
    """

    def __init__(self, agent: Any, name: str, description: str) -> None:
        super().__init__()
        self._agent = agent
        self._name = name
        self._description = description

    @classmethod
    def build(cls, cfg: ToolConfig, ctx: BuildContext) -> SubAgentTool:
        raise NotImplementedError(
            "SubAgentTool is constructed programmatically by the multi-agent "
            "supervisor (AgentPipeline, mode='multi'), not via the tool registry."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, query: str) -> ToolResult:
        try:
            result = self._agent.run(query)
        except Exception as e:
            # A specialist failure becomes a recoverable observation: the
            # supervisor can re-plan rather than the whole query dying.
            logger.warning(
                "Sub-agent '%s' failed for query %r: %s", self._name, query, e
            )
            return ToolResult(
                tool_name=self._name,
                content=f"Specialist '{self._name}' failed: {e}",
                success=False,
                retrieved_chunks=[],
            )
        chunks = result.retrieved_chunks
        return ToolResult(
            tool_name=self._name,
            content=result.answer or "(the specialist returned no answer)",
            success=True,
            retrieved_chunks=chunks,
            metadata={
                "sub_agent": self._name,
                "iterations": result.metadata.get("iterations"),
                "retrieved_chunk_count": len(chunks),
            },
        )
