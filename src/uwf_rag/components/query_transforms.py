"""Query transformer implementations.

Query transformers rewrite or expand a user query before retrieval.
The contextualizer uses an LLM to reformulate follow-up questions
into standalone queries when conversation history is present.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from uwf_rag.components.base import BaseQueryTransformer
from uwf_rag.core.registry import registry
from uwf_rag.core.types import TransformedQuery

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can "
    "be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


@registry.register("query_transform", "contextualizer")
class ContextualizerQueryTransformer(BaseQueryTransformer):
    """Reformulates follow-up questions into standalone queries.

    Uses any registered generator backend (Google, Ollama, EdenAI, etc.)
    to reformulate the query when conversation history is present.
    When there is no history, the query passes through unchanged
    (no LLM call).

    Config params:
        generator_type: Registry name of the generator to use (required).
        llm: LLM config dict passed to the generator constructor.
        system_prompt: Override the default reformulation prompt.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        generator_type = self.config.get("generator_type")
        if not generator_type:
            raise ValueError(
                "ContextualizerQueryTransformer requires 'generator_type' "
                "in config (e.g. 'google', 'ollama'). Set it via "
                "query_transform.params.generator_type in YAML."
            )

        # Pass through all config except query-transform-specific keys,
        # so generator-specific params (e.g. sub_provider, base_url) are preserved.
        _QT_KEYS = {"generator_type", "system_prompt"}
        gen_config = {k: v for k, v in self.config.items() if k not in _QT_KEYS}
        gen_cls = registry.get("generation", generator_type)
        self._generator = gen_cls(config=gen_config)

        self._system_prompt: str = self.config.get(
            "system_prompt", _DEFAULT_SYSTEM_PROMPT
        )

    def transform(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[TransformedQuery]:
        if not history:
            return [TransformedQuery(text=query)]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {"role": "user", "content": query},
        ]

        result = self._generator.generate(messages)
        reformulated = result.answer.strip()
        logger.info("Contextualized query: %r -> %r", query, reformulated)
        return [TransformedQuery(text=reformulated)]


_DEFAULT_HYDE_SYSTEM_PROMPT = (
    "Write a concise, factual passage that directly answers the following "
    "question, as if you were an authoritative source. Do not hedge, "
    "qualify, or say 'the answer is' — just write the passage. Keep it "
    "to a few sentences."
)


@registry.register("query_transform", "hyde")
class HyDEQueryTransformer(BaseQueryTransformer):
    """Hypothetical Document Embeddings (HyDE) transformer.

    Asks the LLM to write a hypothetical passage that would answer the
    question, then uses that passage as the retrieval query. The
    underlying rationale (Gao et al., 2022): an answer-shaped passage
    resembles supporting documents more closely than a question does, so
    the dense embedding similarity improves.

    Default behavior (paper-canonical): emit exactly one hypothetical
    document and use it in place of the original query. By default the
    hypothetical broadcasts to every retriever (``branch=None``, like all
    other transformers). For the canonical HyDE+hybrid recipe, opt in to
    routing: set ``branch="dense"`` so the hypothetical only flows into
    the dense child, plus ``include_original=True`` with
    ``original_branch="bm25"`` so the original question flows into the
    BM25 child.

    Branch hints are silently ignored by non-hybrid retrievers, so the
    same config works in both dense-only and hybrid setups.

    Config params:
        generator_type: Registry name of the generator (required).
        llm: LLM config dict for the generator.
        system_prompt: Override the default HyDE instruction.
        num_hypotheticals: How many hypothetical docs to generate
            (default 1). Note: at ``llm.temperature=0`` (the default in
            base.yaml), repeated calls produce identical text — bump
            temperature for meaningful ``num_hypotheticals > 1``.
        include_original: If True, also emit a TransformedQuery for the
            original user question (default False).
        branch: Branch hint attached to each hypothetical
            TransformedQuery (default None — broadcast to all retrievers).
            Set to a hybrid child name (e.g. "dense") to route there.
        original_branch: Branch hint for the original query when
            ``include_original=True`` (default None — broadcasts to all
            branches; set to ``"bm25"`` for the canonical HyDE+hybrid
            recipe).
    """

    _OWN_KEYS = frozenset(
        {
            "generator_type",
            "system_prompt",
            "num_hypotheticals",
            "include_original",
            "branch",
            "original_branch",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        generator_type = self.config.get("generator_type")
        if not generator_type:
            raise ValueError(
                "HyDEQueryTransformer requires 'generator_type' in config "
                "(e.g. 'edenai', 'google', 'ollama'). Set it via "
                "query_transform.params.generator_type in YAML."
            )

        num_hyp = int(self.config.get("num_hypotheticals", 1))
        if num_hyp < 1:
            raise ValueError(
                f"HyDEQueryTransformer 'num_hypotheticals' must be >= 1 (got {num_hyp})"
            )
        self._num_hypotheticals: int = num_hyp
        self._include_original: bool = bool(self.config.get("include_original", False))
        self._branch: str | None = self.config.get("branch")
        self._original_branch: str | None = self.config.get("original_branch")

        gen_config = {k: v for k, v in self.config.items() if k not in self._OWN_KEYS}
        gen_cls = registry.get("generation", generator_type)
        self._generator = gen_cls(config=gen_config)

        self._system_prompt: str = self.config.get(
            "system_prompt", _DEFAULT_HYDE_SYSTEM_PROMPT
        )

    def transform(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[TransformedQuery]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        hypotheticals: list[str] = []
        for _ in range(self._num_hypotheticals):
            result = self._generator.generate(messages)
            text = result.answer.strip()
            if text:
                hypotheticals.append(text)

        if not hypotheticals:
            logger.warning(
                "HyDE produced no hypothetical documents for query %r; "
                "falling back to original query",
                query,
            )
            return [TransformedQuery(text=query)]

        out: list[TransformedQuery] = []
        if self._include_original:
            out.append(TransformedQuery(text=query, branch=self._original_branch))
        out.extend(TransformedQuery(text=h, branch=self._branch) for h in hypotheticals)

        logger.info(
            "HyDE produced %d hypothetical(s) (include_original=%s, branch=%r)",
            len(hypotheticals),
            self._include_original,
            self._branch,
        )
        return out


_DEFAULT_MULTI_QUERY_SYSTEM_PROMPT = (
    "Generate {num_queries} different reformulations of the user's question. "
    "Each reformulation should explore a different aspect, phrasing, or "
    "interpretation of the original question to help retrieve diverse "
    "relevant documents. Output one reformulation per line, prefixed with "
    "its number and a period (e.g. '1. ...'). Do not include explanations, "
    "headers, or any text other than the numbered reformulations."
)

# Tolerant matcher for LLM-emitted numbered/bulleted lines. Accepts
# `1.`, `1)`, `1:`, `-`, `*` as prefixes; trims surrounding whitespace
# and stray surrounding quotes from the captured content.
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+[\.\)\:]|\-|\*)\s*(.+?)\s*$")


def _parse_numbered_list(text: str, expected: int) -> list[str]:
    """Extract up to ``expected`` queries from a numbered-list LLM response.

    Strategy:
    1. Match lines against the tolerant numbered/bulleted regex.
    2. If no lines matched, fall back to each non-empty stripped line
       (Ollama qwen3 occasionally drops the numbering prefix).
    3. Truncate to ``expected``. Log a warning if fewer than ``expected``
       reformulations were parsed — the multi-query path still works
       with fewer, just with reduced query diversity.

    Surrounding ``"`` or ``'`` quotes are stripped from each match;
    blank captures are dropped.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    parsed: list[str] = []
    for ln in lines:
        m = _NUMBERED_LINE_RE.match(ln)
        if m:
            cleaned = m.group(1).strip().strip("\"'")
            if cleaned:
                parsed.append(cleaned)

    if not parsed:
        # Fallback: treat each non-empty line as a query.
        parsed = [ln.strip() for ln in lines if ln.strip()]

    if len(parsed) < expected:
        logger.warning(
            "Multi-query parser extracted %d of %d expected reformulations "
            "from LLM output; proceeding with what we have",
            len(parsed),
            expected,
        )
    return parsed[:expected]


@registry.register("query_transform", "multi_query")
class MultiQueryQueryTransformer(BaseQueryTransformer):
    """Multi-query expansion (RAG-Fusion style).

    Asks an LLM to generate ``num_queries`` reformulations of the user's
    question, then returns each as a separate :class:`TransformedQuery`
    for the retriever to fan out over. Combined with the pipeline's RRF
    fusion (``query_transform.fusion="rrf"``), this is the canonical
    RAG-Fusion recipe (Raudaschl, 2024).

    All emitted queries default to ``branch=None``, so they broadcast to
    every hybrid child — pairs naturally with both dense-only and
    hybrid+BM25 setups. Hybrid + multi-query + RRF is reported in the
    literature as the strongest pre-rerank stack.

    Config params:
        generator_type: Registry name of the generator (required).
        llm: LLM config dict for the generator.
        system_prompt: Override the default reformulation prompt. The
            template variable ``{num_queries}`` is replaced with the
            configured count.
        num_queries: How many reformulations to request (default 4 —
            the RAG-Fusion paper default).
        include_original: If True, prepend a TransformedQuery for the
            original user question (default True — keeps the literal
            query in the retrieval pool).
        branch: Branch hint attached to all emitted TransformedQueries
            (default None — broadcast to all hybrid children).
    """

    _OWN_KEYS = frozenset(
        {
            "generator_type",
            "system_prompt",
            "num_queries",
            "include_original",
            "branch",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        generator_type = self.config.get("generator_type")
        if not generator_type:
            raise ValueError(
                "MultiQueryQueryTransformer requires 'generator_type' in "
                "config (e.g. 'edenai', 'google', 'ollama'). Set it via "
                "query_transform.params.generator_type in YAML."
            )

        num_queries = int(self.config.get("num_queries", 4))
        if num_queries < 1:
            raise ValueError(
                f"MultiQueryQueryTransformer 'num_queries' must be >= 1 "
                f"(got {num_queries})"
            )
        self._num_queries: int = num_queries
        self._include_original: bool = bool(self.config.get("include_original", True))
        self._branch: str | None = self.config.get("branch")

        gen_config = {k: v for k, v in self.config.items() if k not in self._OWN_KEYS}
        gen_cls = registry.get("generation", generator_type)
        self._generator = gen_cls(config=gen_config)

        self._system_prompt: str = self.config.get(
            "system_prompt", _DEFAULT_MULTI_QUERY_SYSTEM_PROMPT
        )

    def transform(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[TransformedQuery]:
        system_prompt = self._system_prompt.replace(
            "{num_queries}", str(self._num_queries)
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        result = self._generator.generate(messages)
        reformulations = _parse_numbered_list(result.answer, self._num_queries)

        if not reformulations:
            logger.warning(
                "Multi-query produced no reformulations for query %r; "
                "falling back to original query",
                query,
            )
            return [TransformedQuery(text=query, branch=self._branch)]

        out: list[TransformedQuery] = []
        if self._include_original:
            out.append(TransformedQuery(text=query, branch=self._branch))
        out.extend(
            TransformedQuery(text=q, branch=self._branch) for q in reformulations
        )

        logger.info(
            "Multi-query produced %d reformulation(s) (include_original=%s, branch=%r)",
            len(reformulations),
            self._include_original,
            self._branch,
        )
        return out
