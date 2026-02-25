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

from langchain_core.language_models import BaseChatModel

from ..config import Settings

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
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response))

    @classmethod
    def from_settings(cls, settings: Settings, llm: BaseChatModel) -> StuffGenerator:
        if settings.eval.prompt_template:
            return cls(llm=llm, prompt_template=settings.eval.prompt_template)
        return cls(llm=llm)
