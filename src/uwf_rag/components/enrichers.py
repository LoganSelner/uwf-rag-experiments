"""Chunk enricher implementations.

Enrichers run between chunking and embedding/indexing. They may set a
chunk's ``index_text`` (the text that gets embedded + lexically indexed)
without touching ``content`` (what is stored, generated from, and scored).
This isolates a pre-embedding transform's effect to retrieval only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

from uwf_rag.components.base import BaseChunkEnricher
from uwf_rag.components.generators import build_generator
from uwf_rag.core.config import LLMConfig
from uwf_rag.core.registry import registry
from uwf_rag.core.types import Chunk, Document

logger = logging.getLogger(__name__)

# Anthropic contextual-retrieval prompt, ported verbatim as defaults.
_DEFAULT_DOCUMENT_PROMPT = "<document>\n{doc_content}\n</document>"
_DEFAULT_CHUNK_PROMPT = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk_content}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the "
    "chunk. Answer only with the succinct context and nothing else."
)

_VALID_SCOPES = frozenset({"document", "page", "window"})


@registry.register("chunk_enricher", "contextual")
class ContextualChunkEnricher(BaseChunkEnricher):
    """Prepends a model-generated situating context to each chunk's
    ``index_text`` (the Anthropic contextual-retrieval recipe).

    For each chunk an LLM is shown a surrounding "document" unit and asked
    for a short blurb situating the chunk; the blurb is prepended to the
    text used for embedding + BM25 only. ``content`` is untouched, so the
    generator and RAGAS still see the original chunk — the enrichment's
    effect is isolated to retrieval.

    Config params:
        generator: An LLMConfig (provider + model_name [+ params]) for the
            situating LLM (required). Built lazily via ``build_generator`` on
            the first ``enrich()`` call, so constructing the enricher on the
            cache-load path never needs the generator's API key.
        context_scope: "document" (default), "page", or "window" — the unit
            of surrounding text shown to the LLM. ``document`` = the whole
            source; ``page`` = the chunk's own page; ``window`` = the
            chunk's page +/- ``window_size`` neighbor pages.
        window_size: Neighbor pages per side for ``window`` scope (default 1).
        max_workers: Concurrency for the per-chunk LLM calls (default 8).
        document_prompt / chunk_prompt: Override the default templates
            (must contain ``{doc_content}`` / ``{chunk_content}``).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        gen_cfg = self.config.get("generator")
        if not gen_cfg:
            raise ValueError(
                "ContextualChunkEnricher requires a 'generator' config (an "
                "LLMConfig: provider + model_name [+ params]). Set it via "
                "indexing.chunk_enricher.params.generator in YAML."
            )
        self._generator_llm = LLMConfig.from_dict(gen_cfg)

        self._scope: str = self.config.get("context_scope", "document")
        if self._scope not in _VALID_SCOPES:
            raise ValueError(
                f"ContextualChunkEnricher context_scope must be one of "
                f"{sorted(_VALID_SCOPES)}, got '{self._scope}'"
            )

        self._window_size: int = int(self.config.get("window_size", 1))
        if self._window_size < 0:
            raise ValueError(
                f"ContextualChunkEnricher window_size must be >= 0 "
                f"(got {self._window_size})"
            )

        self._max_workers: int = int(self.config.get("max_workers", 8))
        if self._max_workers < 1:
            raise ValueError(
                f"ContextualChunkEnricher max_workers must be >= 1 "
                f"(got {self._max_workers})"
            )

        self._document_prompt: str = self.config.get(
            "document_prompt", _DEFAULT_DOCUMENT_PROMPT
        )
        self._chunk_prompt: str = self.config.get("chunk_prompt", _DEFAULT_CHUNK_PROMPT)

        # Built lazily on first enrich() so constructing the enricher on the
        # cache-load path never requires the generator's API key.
        self._generator: Any = None

    def _build_generator(self) -> Any:
        return build_generator(self._generator_llm)

    def enrich(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        if self._generator is None:
            self._generator = self._build_generator()

        pages_by_source, source_full = self._build_doc_maps(documents)

        def _situate(chunk: Chunk) -> None:
            doc_text = self._scope_text(chunk, pages_by_source, source_full)
            if not doc_text:
                return  # no surrounding unit resolvable → leave index_text unset
            try:
                context = self._call_llm(doc_text, chunk.content)
            except Exception as e:
                # One bad chunk must not fail a multi-thousand-call build.
                logger.warning(
                    "Contextual enrichment failed for chunk %s: %s; "
                    "falling back to raw content",
                    chunk.chunk_id,
                    e,
                )
                return
            if context:
                chunk.index_text = f"{context}\n\n{chunk.content}"

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            list(pool.map(_situate, chunks))

        enriched = sum(1 for c in chunks if c.index_text is not None)
        logger.info(
            "Contextual enrichment set index_text on %d/%d chunks (scope=%s)",
            enriched,
            len(chunks),
            self._scope,
        )
        return chunks

    def _call_llm(self, doc_text: str, chunk_text: str) -> str:
        message = (
            self._document_prompt.format(doc_content=doc_text)
            + "\n"
            + self._chunk_prompt.format(chunk_content=chunk_text)
        )
        result = self._generator.generate([{"role": "user", "content": message}])
        return str(result.answer).strip()

    # --- scope resolution -------------------------------------------------

    @staticmethod
    def _build_doc_maps(
        documents: list[Document],
    ) -> tuple[dict[str, dict[Any, str]], dict[str, str]]:
        """Index document text by source for scope resolution.

        Returns ``(pages_by_source, source_full)`` where
        ``pages_by_source[source_path][page_number]`` is a page's content and
        ``source_full[source_path]`` is all of a source's pages joined in
        page order.
        """
        pages_by_source: dict[str, dict[Any, str]] = {}
        for doc in documents:
            sp = doc.metadata.get("source_path", "")
            pn = doc.metadata.get("page_number")
            pages_by_source.setdefault(sp, {})[pn] = doc.content

        source_full: dict[str, str] = {}
        for sp, pages in pages_by_source.items():
            try:
                ordered = [pages[k] for k in sorted(pages)]
            except TypeError:  # mixed/None keys aren't sortable
                ordered = list(pages.values())
            source_full[sp] = "\n\n".join(ordered)
        return pages_by_source, source_full

    def _scope_text(
        self,
        chunk: Chunk,
        pages_by_source: dict[str, dict[Any, str]],
        source_full: dict[str, str],
    ) -> str:
        sp = chunk.metadata.get("source_path", "")
        pn = chunk.metadata.get("page_number")

        if self._scope == "document":
            return source_full.get(sp, "")

        pages = pages_by_source.get(sp, {})
        if self._scope == "page":
            return pages.get(pn) or source_full.get(sp, "")

        # window: the chunk's page +/- window_size neighbors (numeric pages).
        if isinstance(pn, int):
            lo, hi = pn - self._window_size, pn + self._window_size
            window = [
                pages[k] for k in sorted(pages) if isinstance(k, int) and lo <= k <= hi
            ]
            if window:
                return "\n\n".join(window)
        return source_full.get(sp, "")
