"""Shared base for SDK/HTTP-backed generators + the generator factory.

Each generator wraps an LLM backend and produces a ``GenerationResult`` from a
formatted prompt; generators are pure LLM wrappers and never see raw chunks (the
pipeline calls ``PromptTemplate.format()`` first). ``_SDKGenerator`` factors the
parts the four providers share — the ``llm`` param block + field unpacking, the
API-key env helper, and the ``GenerationResult`` assembly — so each provider
module is left with only its genuinely provider-specific message translation,
SDK/HTTP call, and response parsing.

The provider ``generate()`` orchestration is deliberately *not* templated: tool
binding happens at different points per provider (a payload key for Ollama, a
``GenerateContentConfig`` for Gemini, ``bind_tools`` for EdenAI, a ``create``
kwarg for OpenAI) and the error context differs, so a shared *assembly* helper
is the cleaner seam than a forced reason→act template.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from ragbench.components.base import BaseGenerator, ComponentParams
from ragbench.core.config import LLMConfig
from ragbench.core.registry import registry
from ragbench.core.types import GenerationResult, ToolCall


class _SDKGenerator(BaseGenerator):
    """Skeleton for SDK/HTTP-backed generators.

    Subclasses set ``provider_label``, extend ``Params`` with any provider
    extras, set up their client in ``__init__`` (after ``super().__init__``),
    and implement ``generate``. They reuse :meth:`_require_env` for key lookup
    and :meth:`_assemble` for the uniform result shape.
    """

    provider_label: str = ""

    _model: str
    _temperature: float
    _max_tokens: int | None

    class Params(ComponentParams):
        llm: LLMConfig = Field(default_factory=LLMConfig)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        llm = self.Params.model_validate(self.config).llm
        self._model = llm.model_name
        self._temperature = llm.temperature
        self._max_tokens = llm.max_tokens

    @staticmethod
    def _require_env(*names: str) -> str:
        """Return the first present env var among ``names``, else raise ValueError."""
        for name in names:
            value = os.environ.get(name, "")
            if value:
                return value
        joined = " or ".join(names)
        raise ValueError(
            f"{joined} environment variable is required. "
            "Set it in .env or your shell environment."
        )

    def _assemble(
        self,
        answer: str,
        tool_calls: list[ToolCall],
        *,
        finish_reason: str,
        **extra: Any,
    ) -> GenerationResult:
        """Build the GenerationResult with the shared model/provider metadata.

        The pipeline fills in ``query`` afterwards; provider-specific metadata
        (token counts, durations, sub_provider) is passed via ``extra``.
        """
        return GenerationResult(
            query="",  # Pipeline fills this in
            answer=answer,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            metadata={"model": self._model, "provider": self.provider_label, **extra},
        )


def build_generator(llm: LLMConfig) -> BaseGenerator:
    """Construct a generator from an :class:`LLMConfig` via the registry.

    The single construction path for every generator in the harness — used by
    the linear pipeline, the agent's reasoning model, and any component that
    needs an LLM (query transformers, the contextual enricher). ``llm.provider``
    is the ``generation`` registry name; ``llm.params`` carries provider-specific
    extras (EdenAI ``sub_provider``, Ollama ``base_url``) that land at the top
    level of the generator config, alongside the nested ``llm`` block each
    generator reads.
    """
    gen_cls = registry.get("generation", llm.provider)
    config: dict[str, Any] = {
        **llm.params,
        "llm": {
            "provider": llm.provider,
            "model_name": llm.model_name,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
        },
    }
    return gen_cls(config=config)
