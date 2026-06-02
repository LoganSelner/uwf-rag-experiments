"""Prompt template implementations.

Each prompt template formats a query, retrieved chunks, and optional
conversation history into a prompt consumable by a generator.
"""

from __future__ import annotations

from typing import Any

from uwf_rag.components.base import BasePromptTemplate
from uwf_rag.core.registry import registry
from uwf_rag.core.types import RetrievedChunk


def _format_chunks_numbered(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as a numbered list."""
    parts: list[str] = []
    for i, rc in enumerate(chunks, 1):
        parts.append(f"[{i}] {rc.chunk.content}")
    return "\n\n".join(parts)


def _format_chunks_plain(chunks: list[RetrievedChunk]) -> str:
    """Format chunks separated by dividers."""
    return "\n\n---\n\n".join(rc.chunk.content for rc in chunks)


_CONTEXT_FORMATTERS = {
    "numbered": _format_chunks_numbered,
    "plain": _format_chunks_plain,
}


@registry.register("prompts", "chat")
class ChatPromptTemplate(BasePromptTemplate):
    """Formats query + chunks into a chat-style message list.

    Config params (from PromptConfig):
        system_template: System message text.
        context_format: How to format chunks ("numbered" or "plain").
        use_chain_of_thought: Whether to add CoT instruction.
        citation_style: Citation instruction ("none", "inline", "verbatim").
        few_shot_examples: List of {query, answer} dicts.
        max_context_tokens: Optional limit (not enforced here — future).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._system_template: str = self.config.get("system_template", "")
        self._context_format: str = self.config.get("context_format", "numbered")
        self._use_cot: bool = self.config.get("use_chain_of_thought", False)
        self._citation_style: str = self.config.get("citation_style", "none")
        self._few_shot: list[dict[str, str]] = self.config.get("few_shot_examples", [])

    def format(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build a chat message list: system + history + user."""
        messages: list[dict[str, str]] = []

        # System message
        system = self._build_system_message(chunks)
        if system:
            messages.append({"role": "system", "content": system})

        # Few-shot examples
        for example in self._few_shot:
            messages.append(
                {
                    "role": "user",
                    "content": example.get("query", ""),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": example.get("answer", ""),
                }
            )

        # Conversation history
        if history:
            messages.extend(history)

        # User message with query
        messages.append({"role": "user", "content": query})

        return messages

    def _build_system_message(self, chunks: list[RetrievedChunk]) -> str:
        """Assemble the system message with context and instructions."""
        parts: list[str] = []

        if self._system_template:
            parts.append(self._system_template)

        # Format and inject context
        if chunks:
            formatter = _CONTEXT_FORMATTERS.get(
                self._context_format, _format_chunks_numbered
            )
            context_text = formatter(chunks)
            parts.append(f"CONTEXT:\n{context_text}")

        # Citation instruction
        if self._citation_style == "inline":
            parts.append(
                "Cite your sources using [1], [2], etc. "
                "corresponding to the context numbers above."
            )
        elif self._citation_style == "verbatim":
            parts.append(
                "Use ONLY verbatim quotes from the context above. "
                "Do not paraphrase or add information."
            )

        # Chain-of-thought
        if self._use_cot:
            parts.append("Think step by step before answering. Show your reasoning.")

        return "\n\n".join(parts)
