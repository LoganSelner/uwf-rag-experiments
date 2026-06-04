"""Docs-as-code drift guards.

The ARCHITECTURE.md "Registered Implementations" table is the one component
inventory worth keeping in prose (it's a quick reference). These tests bind every
cell of it — `(category, name, class, file)` — to the live registry, so the table
can't silently drift when a component is added, removed, renamed, or **moved to a
different file** (the failure mode that let the generators-package split leave
stale `generators.py` cells behind).
"""

from __future__ import annotations

from pathlib import Path
import re

# Populate the registry with the real components (side-effect import).
import ragbench.components  # noqa: F401
from ragbench.core.registry import registry

_ARCHITECTURE = Path(__file__).resolve().parent.parent / "ARCHITECTURE.md"
_COMPONENTS_PREFIX = "ragbench.components."

# A full table row: | `category` | `name` | `Class` | `file` |
_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|"
)


def _documented_rows() -> list[tuple[str, str, str, str]]:
    """Parse (category, name, class, file) rows from the Registered table."""
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    marker = "### Registered Implementations"
    assert marker in text, f"{_ARCHITECTURE.name} lost its '{marker}' section"
    section = text[text.index(marker) :]
    # Stop at the next top-level (##) section so we only read this table.
    nxt = section.find("\n## ")
    if nxt != -1:
        section = section[:nxt]
    rows = [m.groups() for line in section.splitlines() if (m := _ROW.match(line))]
    assert rows, "parsed no rows from the Registered Implementations table"
    return rows  # type: ignore[return-value]


def _module_to_file(module: str) -> str:
    """``ragbench.components.generators.ollama`` -> ``generators/ollama.py``."""
    assert module.startswith(_COMPONENTS_PREFIX), module
    return module[len(_COMPONENTS_PREFIX) :].replace(".", "/") + ".py"


def test_architecture_registry_table_in_sync() -> None:
    """Category/name pairs in the table exactly match the live registry."""
    documented = {(cat, name) for cat, name, _, _ in _documented_rows()}
    # Drop underscore-prefixed test doubles (e.g. test_enrichers' fake generators)
    # that other test modules register on the global registry at import time.
    live = {
        (cat, name) for cat, name in registry.registered() if not name.startswith("_")
    }
    assert live == documented, (
        "ARCHITECTURE.md 'Registered Implementations' table is out of sync with the "
        "live registry.\n"
        f"  Registered but undocumented: {sorted(live - documented)}\n"
        f"  Documented but not registered: {sorted(documented - live)}"
    )


def test_architecture_registry_table_class_and_file_match() -> None:
    """Each row's Class + File cells match the registered class's real module."""
    mismatches: list[str] = []
    for category, name, doc_class, doc_file in _documented_rows():
        if not registry.is_registered(category, name):
            continue  # covered by the pairs test above
        cls = registry.get(category, name)
        actual_file = _module_to_file(cls.__module__)
        if cls.__name__ != doc_class:
            mismatches.append(
                f"({category}/{name}) class: doc={doc_class!r} actual={cls.__name__!r}"
            )
        if actual_file != doc_file:
            mismatches.append(
                f"({category}/{name}) file: doc={doc_file!r} actual={actual_file!r}"
            )
    assert not mismatches, (
        "ARCHITECTURE table class/file cells drifted:\n  " + "\n  ".join(mismatches)
    )
