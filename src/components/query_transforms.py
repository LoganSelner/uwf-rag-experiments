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
