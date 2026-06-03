"""Tests for src/core/registry.py — component registration system."""

from __future__ import annotations

import pytest

from ragbench.core.registry import ComponentRegistry, registry


@pytest.fixture()
def clean_registry() -> ComponentRegistry:
    """A fresh, empty registry for isolated tests."""
    return ComponentRegistry()


class TestComponentRegistry:
    def test_register_and_get(self, clean_registry: ComponentRegistry) -> None:
        @clean_registry.register("chunking", "test")
        class TestChunker:
            pass

        assert clean_registry.get("chunking", "test") is TestChunker

    def test_decorator_returns_class(self, clean_registry: ComponentRegistry) -> None:
        @clean_registry.register("chunking", "test")
        class TestChunker:
            pass

        assert TestChunker.__name__ == "TestChunker"

    def test_get_missing_raises_keyerror(
        self, clean_registry: ComponentRegistry
    ) -> None:
        with pytest.raises(KeyError):
            clean_registry.get("chunking", "nonexistent")

    def test_get_missing_lists_available(
        self, clean_registry: ComponentRegistry
    ) -> None:
        @clean_registry.register("chunking", "recursive_langchain")
        class Rec:
            pass

        with pytest.raises(KeyError, match="recursive_langchain"):
            clean_registry.get("chunking", "bogus")

    def test_duplicate_registration_raises(
        self, clean_registry: ComponentRegistry
    ) -> None:
        @clean_registry.register("chunking", "dup")
        class First:
            pass

        with pytest.raises(ValueError):

            @clean_registry.register("chunking", "dup")
            class Second:
                pass

    def test_list_category(self, clean_registry: ComponentRegistry) -> None:
        @clean_registry.register("retrieval", "dense")
        class D:
            pass

        @clean_registry.register("retrieval", "hybrid")
        class H:
            pass

        names = clean_registry.list_category("retrieval")
        assert set(names) == {"dense", "hybrid"}

    def test_list_category_empty(self, clean_registry: ComponentRegistry) -> None:
        assert clean_registry.list_category("unknown") == []

    def test_is_registered_true(self, clean_registry: ComponentRegistry) -> None:
        @clean_registry.register("gen", "ollama")
        class OllamaGen:
            pass

        assert clean_registry.is_registered("gen", "ollama") is True

    def test_is_registered_false(self, clean_registry: ComponentRegistry) -> None:
        assert clean_registry.is_registered("gen", "ollama") is False

    def test_clear(self, clean_registry: ComponentRegistry) -> None:
        @clean_registry.register("gen", "test")
        class T:
            pass

        clean_registry.clear()
        assert clean_registry.is_registered("gen", "test") is False


# -----------------------------------------------------------------------
# Registration inventory — verifies that `import components` (in
# conftest.py) registers all expected types. If this fails after
# adding a new component, ensure the implementation file is imported
# in components/__init__.py.
# -----------------------------------------------------------------------


class TestRegistrationInventory:
    # Indexing
    def test_ingest_pdf(self) -> None:
        assert registry.is_registered("ingest", "pdf")

    def test_chunking_recursive_langchain(self) -> None:
        assert registry.is_registered("chunking", "recursive_langchain")

    def test_chunking_recursive_custom(self) -> None:
        assert registry.is_registered("chunking", "recursive_custom")

    def test_chunking_semantic(self) -> None:
        assert registry.is_registered("chunking", "semantic")

    def test_chunk_enricher_none(self) -> None:
        assert registry.is_registered("chunk_enricher", "none")

    def test_chunk_enricher_contextual(self) -> None:
        assert registry.is_registered("chunk_enricher", "contextual")

    def test_embedding_huggingface(self) -> None:
        assert registry.is_registered("embedding", "huggingface")

    def test_vectorstore_faiss(self) -> None:
        assert registry.is_registered("vectorstore", "faiss")

    def test_vectorstore_chroma(self) -> None:
        assert registry.is_registered("vectorstore", "chroma")

    def test_embedding_google(self) -> None:
        assert registry.is_registered("embedding", "google")

    def test_embedding_openai(self) -> None:
        assert registry.is_registered("embedding", "openai")

    def test_embedding_edenai(self) -> None:
        assert registry.is_registered("embedding", "edenai")

    # Sparse / lexical indexes
    def test_sparse_index_bm25(self) -> None:
        assert registry.is_registered("sparse_index", "bm25")

    # Query
    def test_retrieval_dense(self) -> None:
        assert registry.is_registered("retrieval", "dense")

    def test_retrieval_bm25(self) -> None:
        assert registry.is_registered("retrieval", "bm25")

    def test_retrieval_hybrid(self) -> None:
        assert registry.is_registered("retrieval", "hybrid")

    def test_generation_ollama(self) -> None:
        assert registry.is_registered("generation", "ollama")

    def test_generation_edenai(self) -> None:
        assert registry.is_registered("generation", "edenai")

    def test_generation_google(self) -> None:
        assert registry.is_registered("generation", "google")

    def test_generation_openai(self) -> None:
        assert registry.is_registered("generation", "openai")

    def test_prompts_chat(self) -> None:
        assert registry.is_registered("prompts", "chat")

    # Defaults
    def test_query_transform_passthrough(self) -> None:
        assert registry.is_registered("query_transform", "passthrough")

    def test_query_transform_contextualizer(self) -> None:
        assert registry.is_registered("query_transform", "contextualizer")

    def test_query_transform_hyde(self) -> None:
        assert registry.is_registered("query_transform", "hyde")

    def test_query_transform_multi_query(self) -> None:
        assert registry.is_registered("query_transform", "multi_query")

    def test_reranking_none(self) -> None:
        assert registry.is_registered("reranking", "none")

    # Agent tools
    def test_tool_rag(self) -> None:
        assert registry.is_registered("tool", "rag")

    def test_memory_none(self) -> None:
        assert registry.is_registered("memory", "none")

    def test_memory_buffer_window(self) -> None:
        assert registry.is_registered("memory", "buffer_window")
