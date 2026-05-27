"""Query transformer implementations.

Query transformers rewrite or expand a user query before retrieval.
The contextualizer uses an LLM to reformulate follow-up questions
into standalone queries when conversation history is present.
"""

from __future__ import annotations

import logging
from typing import Any

from components.base import BaseQueryTransformer
from core.registry import registry
from core.types import TransformedQuery

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
    document and use it in place of the original query. With hybrid
    retrieval, set ``branch="dense"`` so the hypothetical only flows
    into the dense child; setting ``include_original=True`` with
    ``original_branch="bm25"`` lets the original query flow into the
    BM25 child — the canonical HyDE+hybrid recipe.

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
            TransformedQuery (default "dense" — pairs naturally with a
            hybrid child named "dense"; ignored otherwise).
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
        self._branch: str | None = self.config.get("branch", "dense")
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
