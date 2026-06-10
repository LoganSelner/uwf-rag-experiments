"""Abstention scoring (Phase E, item 13).

Classifies whether a model RESPONSE *abstained* — refused, deferred, or said it
doesn't have the information — so the evaluator can report the abstention
confusion matrix: False Refusal Rate (abstained on an *answerable* question) and
Missed Refusal Rate (answered an *unanswerable* one). The two are always read
together; collapsing them to one number hides the over-refusal failure mode.

Two classifiers behind one :class:`AbstentionClassifier` Protocol:

- :class:`PhraseAbstentionClassifier` — deterministic phrase matcher. Free and
  fully reproducible (no judge), the CI/smoke floor. High-precision but
  low-recall against *indirect* refusals, which is why it is not the formal
  default.
- :class:`LLMAbstentionClassifier` — an ``AspectCritic`` judge (via the
  quarantined ragas adapter) whose definition explicitly counts indirect/hedged
  refusals. The formal default; judge-agnostic (works with any configured
  evaluator LLM, including the local Ollama path).
"""

from __future__ import annotations

from typing import Any, Protocol

from ragbench.core.types import EvalSample
from ragbench.evaluation import _ragas_adapter

ABSTENTION_METRIC_NAME = "abstained"

# The AspectCritic judge instruction. It deliberately counts INDIRECT and hedged
# refusals (the OR-Bench correction to naive refusal classifiers) — essential
# here because the gold abstentions redirect the user ("check MyUWF") rather than
# saying "I refuse", so a literal-minded judge would miss them.
ABSTENTION_DEFINITION = (
    "Determine whether the RESPONSE declines to directly and substantively "
    "answer the question. Return 1 if the response is a refusal, abstention, or "
    "deferral — INCLUDING indirect or hedged refusals such as: stating it lacks "
    "access to the information, saying the information is unavailable or outside "
    "its knowledge, redirecting the user to another office, system, or website, "
    "or otherwise not committing to a concrete answer. Return 0 only if the "
    "response makes a direct, substantive attempt to answer the question with "
    "specific information. A response that both partially answers AND redirects "
    "the user elsewhere for the actual answer counts as 1."
)

# Phrase cues for the deterministic floor: access-denial + redirection, lowercase
# and matched case-insensitively. Kept general (not over-fit to the dataset's
# exact gold phrasings).
_PHRASE_CUES: tuple[str, ...] = (
    "i don't have access",
    "i do not have access",
    "i can't provide",
    "i cannot provide",
    "i don't have that information",
    "i do not have that information",
    "i don't have",
    "i don't know",
    "i'm not able to",
    "i am not able to",
    "i can't look up",
    "i cannot look up",
    "i can't see",
    "i cannot see",
    "i'm unable to",
    "i am unable to",
    "isn't covered",
    "is not covered",
    "not available in",
    "outside the scope",
    "no information",
    "that's outside",
    "that is outside",
    "please contact",
    "please check",
)


class AbstentionClassifier(Protocol):
    """Scores each sample's response for abstention."""

    def classify(self, samples: list[EvalSample]) -> list[float | None]:
        """Return one signal per sample: ``1.0`` if the response abstained,
        ``0.0`` if it answered, ``None`` if the verdict is unknown (excluded from
        the slice denominator)."""
        ...


class PhraseAbstentionClassifier:
    """Deterministic, judge-free abstention detector (the CI/smoke floor)."""

    def __init__(self, cues: tuple[str, ...] = _PHRASE_CUES) -> None:
        self._cues = tuple(c.lower() for c in cues)

    def classify(self, samples: list[EvalSample]) -> list[float | None]:
        return [
            1.0 if any(cue in (s.response or "").lower() for cue in self._cues) else 0.0
            for s in samples
        ]


class LLMAbstentionClassifier:
    """AspectCritic-backed abstention judge (the formal default).

    Delegates to :func:`_ragas_adapter.run_aspect_critics`, so it needs a
    ragas-ready judge ``llm`` and ``run_config`` (the same objects RAGAS uses —
    handed in by the evaluator). A NaN/None verdict stays ``None`` (unknown), so a
    sample the judge couldn't classify is dropped from the slice rather than
    miscounted.
    """

    def __init__(
        self,
        *,
        llm: Any,
        run_config: Any,
        definition: str = ABSTENTION_DEFINITION,
    ) -> None:
        self._llm = llm
        self._run_config = run_config
        self._spec = _ragas_adapter.AspectCriticSpec(
            name=ABSTENTION_METRIC_NAME, definition=definition
        )

    def classify(self, samples: list[EvalSample]) -> list[float | None]:
        if not samples:
            return []
        rows = _ragas_adapter.run_aspect_critics(
            samples,
            [self._spec],
            llm=self._llm,
            run_config=self._run_config,
        )
        return [row.get(ABSTENTION_METRIC_NAME) for row in rows]


def build_abstention_classifier(
    classifier: str,
    *,
    llm: Any | None = None,
    run_config: Any | None = None,
) -> AbstentionClassifier:
    """Construct the configured classifier (``evaluation.abstention.classifier``).

    ``"phrase"`` needs nothing; ``"llm"`` requires a ragas-ready judge ``llm``.
    """
    if classifier == "phrase":
        return PhraseAbstentionClassifier()
    if classifier == "llm":
        if llm is None:
            raise ValueError(
                "The 'llm' abstention classifier requires a judge LLM; configure "
                "evaluation.evaluator_llm (or use classifier 'phrase')."
            )
        return LLMAbstentionClassifier(llm=llm, run_config=run_config)
    raise ValueError(
        f"Unknown abstention classifier: '{classifier}' (expected 'llm' or 'phrase')"
    )
