"""Construction contract for components.

The registry resolves a component *class* by name; this module supplies the
*construction contract* — a small, typed context (:class:`BuildContext`) carrying
the runtime dependencies a component may need (the populated index, a generator
factory, the active query stack) plus the registry handle for recursive builds.

Each component exposes a polymorphic ``build(...)`` classmethod (see
``components.base``); the pipeline resolves the class from the registry and calls
``build`` with a :class:`BuildContext`, so the pipeline never branches on a
concrete implementation name. Components that compose others (the hybrid
retriever; the Phase D2 supervisor) recurse through ``ctx.registry`` + ``build``,
which is the substrate multi-agent construction leans on.

This module lives in the *components* layer and references the populated index
structurally via :class:`IndexHandle` (a ``Protocol``) so components never import
the ``pipeline`` layer — keeping the dependency graph pointing inward.
``IndexArtifact`` (in ``pipeline/indexing.py``) satisfies the protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ragbench.components.generators import build_generator

if TYPE_CHECKING:
    from ragbench.components.base import (
        BaseEmbedder,
        BaseGenerator,
        BaseLexicalIndex,
        BaseVectorStore,
    )
    from ragbench.core.config import LLMConfig, QueryConfig
    from ragbench.core.registry import RegistryLike

# Key under which the BM25 lexical index lives inside an index's
# ``auxiliary_stores``. Defined here (not in the pipeline layer) so a retriever's
# ``build`` can pull its sparse index from the ``IndexHandle`` without importing
# ``pipeline``. Names the implementation so a future SPLADE store can coexist
# under its own key.
BM25_AUX_KEY = "bm25"


class IndexHandle(Protocol):
    """Structural view of a populated index, as seen by a retriever's ``build``.

    ``IndexArtifact`` satisfies this. Declaring it as a ``Protocol`` lets the
    components layer reference the index without importing the pipeline layer.
    """

    vectorstore: BaseVectorStore
    embedder: BaseEmbedder
    auxiliary_stores: dict[str, BaseLexicalIndex | BaseVectorStore]


@dataclass
class BuildContext:
    """Runtime dependencies + handles threaded through component ``build`` calls.

    Args:
        registry: The component registry, for recursive child construction
            (hybrid retriever, future multi-agent supervisor).
        index: The populated index (vector store + embedder + auxiliary
            stores), or ``None`` for stages built before/without an index.
        query: The active query stack, reused by the agent's RAG tool to
            build a retrieval-only sub-pipeline over the same index.
        make_generator: The single generator factory; injected so LLM-backed
            components (query transformers, tools) don't each reach for
            ``build_generator`` directly.
    """

    registry: RegistryLike
    index: IndexHandle | None = None
    query: QueryConfig | None = None
    make_generator: Callable[[LLMConfig], BaseGenerator] = build_generator
