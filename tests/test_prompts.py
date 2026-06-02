"""Tests for src/components/prompts.py — ChatPromptTemplate."""

from __future__ import annotations

from uwf_rag.components.prompts import (
    ChatPromptTemplate,
    _format_chunks_numbered,
    _format_chunks_plain,
)
from uwf_rag.core.types import Chunk, RetrievedChunk


def _rc(content: str, score: float = 0.9) -> RetrievedChunk:
    """Shorthand helper to build a RetrievedChunk."""
    return RetrievedChunk(
        chunk=Chunk(content=content, chunk_id="id", metadata={}),
        score=score,
    )


# -----------------------------------------------------------------------
# Context formatters
# -----------------------------------------------------------------------


class TestFormatChunksNumbered:
    def test_single_chunk(self) -> None:
        result = _format_chunks_numbered([_rc("Hello")])
        assert result == "[1] Hello"

    def test_multiple_chunks(self) -> None:
        result = _format_chunks_numbered([_rc("A"), _rc("B"), _rc("C")])
        assert "[1] A" in result
        assert "[2] B" in result
        assert "[3] C" in result


class TestFormatChunksPlain:
    def test_single_chunk(self) -> None:
        result = _format_chunks_plain([_rc("Hello")])
        assert result == "Hello"

    def test_multiple_chunks_separated(self) -> None:
        result = _format_chunks_plain([_rc("A"), _rc("B")])
        assert "---" in result
        assert "A" in result
        assert "B" in result


# -----------------------------------------------------------------------
# ChatPromptTemplate.format
# -----------------------------------------------------------------------


class TestChatPromptTemplate:
    def test_basic_format(self) -> None:
        tpl = ChatPromptTemplate({"system_template": "You are a helpful assistant."})
        msgs = tpl.format("What is X?", [_rc("Context about X")])
        assert msgs[0]["role"] == "system"
        assert "helpful assistant" in msgs[0]["content"]
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "What is X?"

    def test_numbered_context_default(self) -> None:
        tpl = ChatPromptTemplate()
        msgs = tpl.format("Q?", [_rc("chunk1"), _rc("chunk2")])
        system = msgs[0]["content"]
        assert "[1]" in system
        assert "[2]" in system

    def test_plain_context(self) -> None:
        tpl = ChatPromptTemplate({"context_format": "plain"})
        msgs = tpl.format("Q?", [_rc("chunk1"), _rc("chunk2")])
        system = msgs[0]["content"]
        assert "---" in system
        assert "[1]" not in system

    def test_chain_of_thought(self) -> None:
        tpl = ChatPromptTemplate({"use_chain_of_thought": True})
        msgs = tpl.format("Q?", [_rc("ctx")])
        system = msgs[0]["content"]
        assert "step by step" in system.lower()

    def test_inline_citation(self) -> None:
        tpl = ChatPromptTemplate({"citation_style": "inline"})
        msgs = tpl.format("Q?", [_rc("ctx")])
        system = msgs[0]["content"]
        assert "[1]" in system or "Cite" in system

    def test_verbatim_citation(self) -> None:
        tpl = ChatPromptTemplate({"citation_style": "verbatim"})
        msgs = tpl.format("Q?", [_rc("ctx")])
        system = msgs[0]["content"]
        assert "verbatim" in system.lower()

    def test_few_shot_examples(self) -> None:
        tpl = ChatPromptTemplate(
            {
                "few_shot_examples": [
                    {"query": "Example Q?", "answer": "Example A."},
                ],
            }
        )
        msgs = tpl.format("Real Q?", [])
        # few-shot user/assistant pair + final user message
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles
        assert msgs[-1]["content"] == "Real Q?"

    def test_history_included(self) -> None:
        tpl = ChatPromptTemplate({"system_template": "Sys"})
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        msgs = tpl.format("Follow-up?", [], history=history)
        contents = [m["content"] for m in msgs]
        assert "Hi" in contents
        assert "Hello!" in contents
        assert msgs[-1]["content"] == "Follow-up?"

    def test_no_system_template_no_chunks(self) -> None:
        tpl = ChatPromptTemplate()
        msgs = tpl.format("Q?", [])
        # No system message when template is empty and no chunks
        assert all(m["role"] != "system" for m in msgs)
        assert msgs[0]["role"] == "user"

    def test_no_chunks_with_system_template(self) -> None:
        tpl = ChatPromptTemplate({"system_template": "Be concise."})
        msgs = tpl.format("Q?", [])
        assert msgs[0]["role"] == "system"
        assert "CONTEXT" not in msgs[0]["content"]
