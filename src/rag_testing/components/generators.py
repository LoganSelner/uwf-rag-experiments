"""Generator implementations.

Each class satisfies the ``Generator`` Protocol defined in ``.base``.

Implementations
---------------
StuffGenerator — concatenates all context chunks into one prompt (current baseline)

To add a new generator (e.g. MapReduceGenerator, RefineGenerator):
  1. Add a class here satisfying the ``Generator`` Protocol (``generate`` method) and
     add a ``from_settings(cls, settings: Settings, llm: BaseChatModel) -> <class>``
     classmethod.
  2. Export it from ``__init__.py``.
  3. Add a branch in ``pipeline._build_generator()`` for the new ``generator_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from langchain_core.language_models import BaseChatModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings

logger = logging.getLogger(__name__)

# Default prompt template used by StuffGenerator.
# Callers can supply a custom template via the prompt_template field.
# The template must contain exactly two placeholders: {context} and {question}.
STUFF_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say so concisely.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# Exceptions that clearly indicate programming bugs — never retry these.
_NON_RETRYABLE = (TypeError, ValueError, KeyError, AttributeError, SyntaxError)

# Network / transient exceptions that should always be retried.
_RETRYABLE = (ConnectionError, TimeoutError, OSError)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that are worth retrying.

    Known network exceptions are always retried. Deterministic programming
    errors (TypeError, ValueError, etc.) are never retried. Everything else
    (including EdenAI's bare ``Exception`` for provider errors) is retried
    as a conservative default — the run-loop safety net handles persistent
    failures.
    """
    if isinstance(exc, _NON_RETRYABLE):
        return False
    return True


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm_with_retry(llm: BaseChatModel, prompt: str) -> str:
    """Invoke the LLM with automatic retry on transient failures."""
    response = llm.invoke(prompt)
    return str(getattr(response, "content", response))


@dataclass(frozen=True, eq=False)  # eq=False: BaseChatModel is not hashable
class StuffGenerator:
    """Concatenates all context chunks into a single prompt (the 'stuff' strategy).

    prompt_template must contain ``{context}`` and ``{question}`` placeholders.
    Defaults to ``STUFF_PROMPT``; pass a custom string to experiment with
    different system instructions or output formats without creating a new class.
    """

    llm: BaseChatModel
    prompt_template: str = STUFF_PROMPT

    def generate(self, question: str, contexts: list[str]) -> str:
        stuffed = "\n\n---\n\n".join(contexts)
        prompt = self.prompt_template.format(context=stuffed, question=question)
        return _invoke_llm_with_retry(self.llm, prompt)

    @classmethod
    def from_settings(cls, settings: Settings, llm: BaseChatModel) -> StuffGenerator:
        if settings.eval.prompt_template:
            return cls(llm=llm, prompt_template=settings.eval.prompt_template)
        return cls(llm=llm)
