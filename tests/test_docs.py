"""Docs-as-code drift guards.

The ARCHITECTURE.md "Registered Implementations" table is the one component
inventory worth keeping in prose (it's a quick reference). This test binds it to
the live registry so the table can't silently drift when a component is added,
removed, or renamed — the failure mode the rest of the cleanup explicitly fought.
"""

from __future__ import annotations

from pathlib import Path
import re

# Populate the registry with the real components (side-effect import).
import ragbench.components  # noqa: F401
from ragbench.core.registry import registry

_ARCHITECTURE = Path(__file__).resolve().parent.parent / "ARCHITECTURE.md"

# A table row: | `category` | `name` | `Class` | `file` |  — first two cells only.
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


def _documented_pairs() -> set[tuple[str, str]]:
    """Parse the (category, name) pairs from the Registered Implementations table."""
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    marker = "### Registered Implementations"
    assert marker in text, f"{_ARCHITECTURE.name} lost its '{marker}' section"
    section = text[text.index(marker) :]
    # Stop at the next top-level (##) section so we only read this table.
    nxt = section.find("\n## ")
    if nxt != -1:
        section = section[:nxt]
    return {
        (m.group(1), m.group(2))
        for line in section.splitlines()
        if (m := _ROW.match(line))
    }


def test_architecture_registry_table_in_sync() -> None:
    documented = _documented_pairs()
    # Drop underscore-prefixed test doubles (e.g. test_enrichers' fake generators)
    # that other test modules register on the global registry at import time.
    live = {
        (cat, name) for cat, name in registry.registered() if not name.startswith("_")
    }

    missing_from_docs = sorted(live - documented)
    stale_in_docs = sorted(documented - live)
    assert live == documented, (
        "ARCHITECTURE.md 'Registered Implementations' table is out of sync with the "
        "live registry.\n"
        f"  Registered but undocumented: {missing_from_docs}\n"
        f"  Documented but not registered: {stale_in_docs}"
    )
