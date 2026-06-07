"""Anti-corruption layer: the ONLY module that imports ragas.

Quarantines the entire ragas surface — including the deprecated *private* metric
classes (``ragas.metrics._*``) imported to dodge the 0.4.x ``ragas.metrics``
deprecation warnings — behind a small, stable internal interface, so a ragas
version bump touches this one file instead of the evaluator.

**Why the legacy ``evaluate()`` path, not the new collections API.** ragas 0.4
ships a ``ragas.metrics.collections`` API (the non-deprecated replacement), but
its metrics take an ``InstructorBaseRagasLLM`` at construction — built from an
instructor/litellm-supported provider. That cannot wrap this project's EdenAI
judge, which only exists as a ``langchain-community`` ``ChatEdenAI`` and reaches
ragas via ``LangchainLLMWrapper``. ``evaluate(llm=LangchainLLMWrapper(...))`` is
the only path that supports EdenAI, so the harness stays on the legacy metrics +
``evaluate()`` (pinned ``ragas<0.5``) and isolates them here. Revisit only if
ragas grows an EdenAI-capable structured LLM, or EdenAI is dropped as a judge.

All ragas imports are deferred inside the functions so importing this module is
cheap and ragas-version problems surface only when an evaluation actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Any

from ragbench.core.types import EvalSample

# The metric names this adapter can build. Declared statically (no ragas import)
# so callers can filter their configured metrics without importing ragas; the
# lazy builder below asserts its keys match this set so the two can't drift.
KNOWN_METRICS: frozenset[str] = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "answer_similarity",
        "factual_correctness",
        "context_precision",
        "context_recall",
        "context_entity_recall",
    }
)


@functools.cache
def _metric_map() -> dict[str, Any]:
    """Build ragas metric instances (cached).

    Imports from the private ``ragas.metrics._*`` submodules to avoid the
    deprecation warnings that ``ragas.metrics`` top-level imports trigger in
    ragas 0.4.x. This is the single place that fragile coupling lives.
    """
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._answer_similarity import AnswerSimilarity
    from ragas.metrics._context_entities_recall import ContextEntityRecall
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.metrics._faithfulness import Faithfulness

    mapping: dict[str, Any] = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "answer_correctness": AnswerCorrectness(),
        "answer_similarity": AnswerSimilarity(),
        "factual_correctness": FactualCorrectness(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "context_entity_recall": ContextEntityRecall(),
    }
    if set(mapping) != KNOWN_METRICS:
        raise RuntimeError(
            "ragas adapter metric map "
            f"{sorted(mapping)} drifted from KNOWN_METRICS "
            f"{sorted(KNOWN_METRICS)}."
        )
    return mapping


def make_run_config(
    *, timeout: int, max_retries: int, max_wait: int, max_workers: int
) -> Any:
    """Build a ragas ``RunConfig`` from the harness's evaluation settings."""
    from ragas import RunConfig

    return RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=max_wait,
        max_workers=max_workers,
    )


def wrap_langchain_llm(chat_model: Any) -> Any:
    """Wrap a LangChain chat model in the shape ragas's ``evaluate()`` expects.

    The judge chat models are built per-provider in the evaluator (LangChain,
    not ragas); this is the one ragas-side wrap, kept here with the rest of the
    ragas surface.
    """
    from ragas.llms.base import LangchainLLMWrapper

    return LangchainLLMWrapper(chat_model)


def wrap_embedder(embedder: Any) -> Any:
    """Adapt a ``BaseEmbedder`` to the ragas-native ``BaseRagasEmbedding`` API.

    Bridges the evaluator's dedicated embedding model (built from
    ``evaluator_embedding`` config via the component registry) to the interface
    ragas ``evaluate()`` expects. The adapter class is defined inside the
    function so the ``ragas.embeddings`` import stays deferred.
    """
    from ragas.embeddings import BaseRagasEmbedding

    class _EmbedderAdapter(BaseRagasEmbedding):
        def __init__(self, base_embedder: Any) -> None:
            super().__init__()
            self._embedder = base_embedder

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self._embedder.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self._embedder.embed_query(text)

        # Legacy LangChain interface — ragas 0.4.x metrics like
        # answer_relevancy still call embed_query/embed_documents internally
        # instead of the modern embed_text/embed_texts API.
        def embed_query(self, text: str) -> list[float]:
            return self._embedder.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embedder.embed_query(t) for t in texts]

    return _EmbedderAdapter(embedder)


def run_eval(
    samples: list[EvalSample],
    metric_names: list[str],
    *,
    llm: Any | None,
    embeddings: Any | None,
    run_config: Any,
) -> list[dict[str, Any]]:
    """Score ``samples`` with the named metrics via ragas ``evaluate()``.

    Builds the ragas dataset + metric objects internally and returns the
    per-sample score dicts (``ragas_result.scores``), so the evaluator never
    touches a ragas type. Unknown metric names are dropped (already filtered by
    the caller against :data:`KNOWN_METRICS`).
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample

    metric_map = _metric_map()
    metrics = [metric_map[n] for n in metric_names if n in metric_map]
    if not metrics:
        return [{} for _ in samples]

    ragas_samples = [
        SingleTurnSample(
            user_input=s.query,
            response=s.response,
            retrieved_contexts=s.retrieved_contexts,
            reference=s.reference,
        )
        for s in samples
    ]

    eval_kwargs: dict[str, Any] = {"run_config": run_config}
    if llm is not None:
        eval_kwargs["llm"] = llm
    if embeddings is not None:
        eval_kwargs["embeddings"] = embeddings

    ragas_result: Any = evaluate(
        dataset=EvaluationDataset(samples=ragas_samples),  # type: ignore[arg-type]
        metrics=metrics,
        **eval_kwargs,
    )
    scores: list[dict[str, Any]] = ragas_result.scores
    return scores


@dataclass(frozen=True)
class AspectCriticSpec:
    """A binary, free-form LLM-judge metric declared by the evaluator.

    ``name`` is the output column; ``definition`` is the natural-language judge
    instruction (e.g. "return 1 if the response refuses to answer …"). A plain
    dataclass with no ragas import, so the evaluator builds specs from strings
    while every ragas type stays quarantined in this module. Powers the Phase E
    abstention and knowledge-conflict signals.
    """

    name: str
    definition: str


def run_aspect_critics(
    samples: list[EvalSample],
    specs: list[AspectCriticSpec],
    *,
    llm: Any,
    run_config: Any,
    references: list[str] | None = None,
) -> list[dict[str, float | None]]:
    """Score ``samples`` with AspectCritic metrics via the legacy ``evaluate()``.

    The binary-judge sibling of :func:`run_eval`: builds one ``AspectCritic`` per
    spec (private import, the same anti-deprecation style as :func:`_metric_map`)
    and one ``SingleTurnSample`` per sample, runs
    ``evaluate(llm=…, metrics=[…])``, and returns a per-sample
    ``{spec.name: 0.0 | 1.0 | None}`` dict (``None`` when the judge produced no
    parseable verdict). AspectCritic needs no embedder.

    ``llm`` must be ragas-ready (a :func:`wrap_langchain_llm` result), exactly as
    :func:`run_eval` receives it — so the EdenAI judge path works unchanged.
    ``references`` optionally overrides each sample's ``reference`` for the judge
    (the conflict probe compares the response to the authoritative ``corpus_fact``
    rather than the dataset gold); its length must match ``samples``.
    """
    if not specs or not samples:
        return [{} for _ in samples]
    if references is not None and len(references) != len(samples):
        raise ValueError(
            f"references length {len(references)} != samples length {len(samples)}"
        )

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._aspect_critic import AspectCritic

    metrics = [
        AspectCritic(name=spec.name, definition=spec.definition, llm=llm)
        for spec in specs
    ]
    ragas_samples = [
        SingleTurnSample(
            user_input=s.query,
            response=s.response,
            retrieved_contexts=s.retrieved_contexts,
            reference=(references[i] if references is not None else s.reference),
        )
        for i, s in enumerate(samples)
    ]

    ragas_result: Any = evaluate(
        dataset=EvaluationDataset(samples=ragas_samples),  # type: ignore[arg-type]
        metrics=metrics,
        llm=llm,
        run_config=run_config,
    )

    out: list[dict[str, float | None]] = []
    for row in ragas_result.scores:
        cleaned: dict[str, float | None] = {}
        for spec in specs:
            val = row.get(spec.name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                cleaned[spec.name] = None
            else:
                cleaned[spec.name] = float(val)
        out.append(cleaned)
    return out
