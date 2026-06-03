"""Tests for src/components/_tool_protocol.py — pure tool-calling helpers."""

from __future__ import annotations

from ragbench.components._tool_protocol import (
    dump_arguments,
    parse_arguments,
    synthesize_call_id,
    to_openai_tools,
)
from ragbench.core.types import ToolSpec


class TestToOpenAITools:
    def test_shape(self) -> None:
        specs = [
            ToolSpec(
                name="knowledge_base",
                description="Search the KB.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
        out = to_openai_tools(specs)
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "knowledge_base",
                    "description": "Search the KB.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    def test_empty_parameters_defaults_to_object(self) -> None:
        out = to_openai_tools([ToolSpec(name="t", description="d")])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


class TestSynthesizeCallId:
    def test_stable_and_unique(self) -> None:
        assert synthesize_call_id(0, 0) == "call_0_0"
        assert synthesize_call_id(2, 1) == "call_2_1"
        assert synthesize_call_id(0, 0) != synthesize_call_id(0, 1)


class TestParseArguments:
    def test_json_string(self) -> None:
        assert parse_arguments('{"query": "hi"}') == {"query": "hi"}

    def test_already_dict(self) -> None:
        assert parse_arguments({"query": "hi"}) == {"query": "hi"}

    def test_empty_string(self) -> None:
        assert parse_arguments("") == {}
        assert parse_arguments("   ") == {}

    def test_malformed_json_degrades_to_empty(self) -> None:
        assert parse_arguments("not json") == {}

    def test_non_object_json_degrades_to_empty(self) -> None:
        # A JSON array/scalar isn't a valid argument mapping.
        assert parse_arguments("[1, 2]") == {}
        assert parse_arguments("42") == {}

    def test_none_degrades_to_empty(self) -> None:
        assert parse_arguments(None) == {}


class TestDumpArguments:
    def test_roundtrip(self) -> None:
        assert parse_arguments(dump_arguments({"query": "hi"})) == {"query": "hi"}
