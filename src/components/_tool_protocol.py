"""Pure helpers for the native tool-calling protocol.

Shared by the generators so the four providers don't drift on the fiddly
bits: building the OpenAI-style tool schema, synthesizing call ids for
providers that omit them (Ollama, Gemini), and normalizing tool-call
arguments between JSON strings (OpenAI/EdenAI wire format) and parsed dicts
(what the agent loop and ``ToolCall.arguments`` use).

No provider SDKs are imported here — these are plain data transforms and are
unit-tested in isolation.
"""

from __future__ import annotations

import json
from typing import Any

from core.types import ToolSpec


def to_openai_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Render tool specs into the OpenAI ``tools`` array.

    This shape is consumed directly by the OpenAI SDK and the Ollama
    ``/api/chat`` payload, and is accepted by ``ChatEdenAI.bind_tools`` (which
    runs it through ``convert_to_openai_tool``). Gemini uses a different shape
    and builds its own function declarations.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters or {"type": "object", "properties": {}},
            },
        }
        for spec in specs
    ]


def synthesize_call_id(step: int, idx: int) -> str:
    """Build a stable call id for providers that don't return one.

    Ollama and Gemini omit tool-call ids; the agent loop still needs a handle
    to correlate each call with its result turn. ``step`` is the loop
    iteration and ``idx`` the call's position within that turn, so the id is
    unique and reproducible.
    """
    return f"call_{step}_{idx}"


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Normalize provider tool-call arguments to a parsed dict.

    OpenAI/EdenAI return a JSON **string**; Ollama returns an already-parsed
    object. Anything unparseable degrades to an empty dict rather than raising
    — a malformed tool call should become a recoverable observation, not a
    crash.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def dump_arguments(arguments: dict[str, Any]) -> str:
    """Serialize a parsed argument dict back to the JSON string wire format.

    Used when re-serializing an assistant-with-tool_calls turn for providers
    (OpenAI) whose message schema expects ``function.arguments`` as a string.
    """
    return json.dumps(arguments)
