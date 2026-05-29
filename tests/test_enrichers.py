"""Tests for src/components/enrichers.py and the no-op chunk enricher."""

from __future__ import annotations

from typing import Any

import pytest

from components.base import BaseGenerator
from components.defaults import NoOpChunkEnricher
from components.enrichers import ContextualChunkEnricher
from core.registry import registry
from core.types import Chunk, Document, GenerationResult

# --- fake generators registered for offline enrichment tests ---------------


@registry.register("generation", "_fake_ctx_gen")
class _FakeContextGenerator(BaseGenerator):
    """Returns a constant situating blurb, ignoring the prompt."""

    def generate(self, prompt: str | list[dict[str, str]]) -> GenerationResult:
        return GenerationResult(query="", answer="CTX")


@registry.register("generation", "_failing_gen")
class _FailingGenerator(BaseGenerator):
    """Always raises, to exercise per-chunk graceful fallback."""

    def generate(self, prompt: str | list[dict[str, str]]) -> GenerationResult:
        raise RuntimeError("boom")


def _pages() -> list[Document]:
    return [
        Document(
            content=f"PAGE {n}.", metadata={"source_path": "a.pdf", "page_number": n}
        )
        for n in range(1, 6)
    ]


def _chunk_on(page: int, content: str = "the chunk") -> Chunk:
    return Chunk(
        content=content,
        chunk_id=f"c{page}",
        metadata={"source_path": "a.pdf", "page_number": page},
    )


def _enricher(**params: Any) -> ContextualChunkEnricher:
    params.setdefault("generator_type", "_fake_ctx_gen")
    return ContextualChunkEnricher(config=params)


# --- NoOpChunkEnricher ------------------------------------------------------


class TestNoOpChunkEnricher:
    def test_passthrough_leaves_index_text_unset(self) -> None:
        chunks = [_chunk_on(1)]
        out = NoOpChunkEnricher().enrich(_pages(), chunks)
        assert out is chunks
        assert out[0].index_text is None


# --- ContextualChunkEnricher: enrichment ------------------------------------


class TestContextualEnrichment:
    def test_prepends_context_to_index_text_only(self) -> None:
        chunks = [_chunk_on(2, "alpha"), _chunk_on(3, "beta")]
        out = _enricher(context_scope="document").enrich(_pages(), chunks)
        for c in out:
            assert c.index_text == f"CTX\n\n{c.content}"
        # content is preserved (generator/RAGAS see the original).
        assert [c.content for c in out] == ["alpha", "beta"]

    def test_graceful_fallback_on_generator_error(self) -> None:
        chunks = [_chunk_on(2, "alpha")]
        out = _enricher(generator_type="_failing_gen", context_scope="document").enrich(
            _pages(), chunks
        )
        # A failing call leaves index_text unset → text_for_index == content.
        assert out[0].index_text is None
        assert out[0].text_for_index == "alpha"

    def test_empty_chunks_noop(self) -> None:
        assert _enricher().enrich(_pages(), []) == []


# --- ContextualChunkEnricher: scope resolution ------------------------------


class TestScopeResolution:
    def _resolve(self, scope: str, page: int, **params: Any) -> str:
        enr = _enricher(context_scope=scope, **params)
        pages_by_source, source_full = enr._build_doc_maps(_pages())
        return enr._scope_text(_chunk_on(page), pages_by_source, source_full)

    def test_page_scope(self) -> None:
        assert self._resolve("page", 2) == "PAGE 2."

    def test_window_scope_includes_neighbors(self) -> None:
        # page 3, window 1 → pages 2,3,4
        assert (
            self._resolve("window", 3, window_size=1) == "PAGE 2.\n\nPAGE 3.\n\nPAGE 4."
        )

    def test_document_scope_includes_all_pages(self) -> None:
        assert self._resolve("document", 3) == (
            "PAGE 1.\n\nPAGE 2.\n\nPAGE 3.\n\nPAGE 4.\n\nPAGE 5."
        )

    def test_unresolvable_source_skips(self) -> None:
        enr = _enricher(context_scope="document")
        pages_by_source, source_full = enr._build_doc_maps(_pages())
        orphan = Chunk(content="x", chunk_id="o", metadata={"source_path": "other.pdf"})
        assert enr._scope_text(orphan, pages_by_source, source_full) == ""


# --- ContextualChunkEnricher: construction validation -----------------------


class TestContextualConstruction:
    def test_requires_generator_type(self) -> None:
        with pytest.raises(ValueError, match="generator_type"):
            ContextualChunkEnricher(config={})

    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="context_scope"):
            ContextualChunkEnricher(
                config={"generator_type": "_fake_ctx_gen", "context_scope": "bogus"}
            )

    def test_invalid_max_workers_raises(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            ContextualChunkEnricher(
                config={"generator_type": "_fake_ctx_gen", "max_workers": 0}
            )
