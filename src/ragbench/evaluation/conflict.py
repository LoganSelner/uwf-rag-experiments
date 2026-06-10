"""Knowledge-conflict probe scoring (Phase E, item 15).

The probe injects a contradicting passage into the generation context
(``QueryPipeline.run(injected_contexts=...)``) and measures, with two
``AspectCritic`` judges over the conflict slice:

- ``corpus_preference`` — did the answer assert the authoritative corpus fact
  rather than repeat the injected falsehood? (the RGB counterfactual-robustness
  signal).
- ``error_detection`` — did the answer explicitly flag the contradiction?

Plus a confound check (:func:`corpus_fact_retrieved`): was the corpus fact
actually present in the retrieved context? — so a low ``corpus_preference`` isn't
conflated with "the truth was never retrieved".
"""

from __future__ import annotations

import re

from ragbench.evaluation._ragas_adapter import AspectCriticSpec

CORPUS_PREFERENCE_NAME = "corpus_preference"
ERROR_DETECTION_NAME = "error_detection"

# The judge is given REFERENCE = the authoritative corpus_fact (the evaluator
# passes it as the per-sample reference override to run_aspect_critics).
CORPUS_PREFERENCE_DEFINITION = (
    "You are given a QUESTION, the authoritative correct fact from the knowledge "
    "base as the REFERENCE, and a RESPONSE. Return 1 if the response's answer "
    "AGREES with the REFERENCE (states the authoritative value). Return 0 if it "
    "instead asserts a different or contradicting value, or gives no value at "
    "all. A response that hedges but ultimately states the reference value "
    "counts as 1."
)
ERROR_DETECTION_DEFINITION = (
    "Return 1 if the RESPONSE explicitly flags, questions, or notes a "
    "contradiction or inconsistency between its sources (for example 'the "
    "documents disagree', or 'one source says X but the official policy is Y'). "
    "Return 0 if it presents a single answer with no acknowledgment of any "
    "conflict."
)


def conflict_specs() -> list[AspectCriticSpec]:
    """The two AspectCritic metrics for the knowledge-conflict probe."""
    return [
        AspectCriticSpec(CORPUS_PREFERENCE_NAME, CORPUS_PREFERENCE_DEFINITION),
        AspectCriticSpec(ERROR_DETECTION_NAME, ERROR_DETECTION_DEFINITION),
    ]


# Common words ignored when checking whether the corpus fact was retrieved.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "is",
        "are",
        "be",
        "with",
        "that",
        "this",
        "your",
        "you",
        "may",
        "can",
        "will",
        "from",
        "as",
        "it",
        "within",
        "each",
        "only",
        "not",
        "no",
        "any",
        "all",
        "into",
        "per",
        "up",
        "do",
        "does",
        "if",
        "but",
        "must",
        "have",
    }
)


def corpus_fact_retrieved(contexts: list[str], corpus_fact: str) -> bool | None:
    """Best-effort confound check: did retrieval surface the corpus fact?

    Flags ``True`` when some retrieved context contains a majority (>= 60%) of the
    fact's distinctive content words (length >= 4, non-stopword). Returns ``None``
    when the fact has no distinctive words to match. A heuristic for
    interpretation, not a scored metric: a ``False`` means a low
    ``corpus_preference`` might reflect "the truth wasn't retrieved" rather than
    "the model trusted the injected lie".
    """
    words = {
        w
        for w in re.findall(r"[a-z0-9]{4,}", corpus_fact.lower())
        if w not in _STOPWORDS
    }
    if not words:
        return None
    blob = " ".join(contexts).lower()
    hits = sum(1 for w in words if w in blob)
    return hits / len(words) >= 0.6
