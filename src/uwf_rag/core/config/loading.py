"""YAML loading with ``extends:`` inheritance.

Deep-merge semantics: nested dicts merge recursively, lists replace, scalars
replace, and a changed implementation selector (``type``/``provider``) replaces
the whole component subtree (params are private to the implementation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from uwf_rag.core.config.errors import ConfigValidationError

# Keys that select a component implementation. A component's other settings
# (notably ``params``) are scoped to the chosen implementation, so when a child
# config changes one of these the whole subtree is *replaced* rather than
# deep-merged — otherwise the parent's implementation-specific params (e.g.
# EdenAI's ``provider`` / ``sub_provider``) would leak onto a different
# implementation (e.g. a HuggingFace embedder or an Ollama generator). This
# mirrors Hydra's config-group *selection* and the general rule that a swapped
# component's config is private to its kind.
_IMPL_SELECTOR_KEYS = ("type", "provider")


def _selects_different_impl(base: dict[str, Any], override: dict[str, Any]) -> bool:
    """True if *base* and *override* pick different component implementations."""
    return any(
        key in base and key in override and base[key] != override[key]
        for key in _IMPL_SELECTOR_KEYS
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*.

    Rules (from ROADMAP.md §6.4):
    - Nested dicts: merge recursively (override only specified keys)
    - Lists: full replacement (override provides the entire list)
    - Scalars: override replaces base
    - Component dicts whose implementation selector (``type``/``provider``)
      changes: full replacement (params are private to the implementation)
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
            and not _selects_different_impl(merged[key], value)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_inheritance(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends:`` inheritance.

    The ``extends`` field is relative to the directory containing the
    child config file. Multiple levels of inheritance are supported.

    Raises:
        ConfigValidationError: If a circular inheritance chain is detected.
    """
    return _load_yaml_recursive(path, _visited=None)


def _load_yaml_recursive(
    path: str | Path,
    *,
    _visited: set[Path] | None,
) -> dict[str, Any]:
    """Internal: load YAML with cycle detection."""
    path = Path(path).resolve()

    if _visited is None:
        _visited = set()

    if path in _visited:
        raise ConfigValidationError([f"Circular YAML inheritance detected: {path}"])
    _visited.add(path)

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    parent_path = (path.parent / extends).resolve()
    parent = _load_yaml_recursive(parent_path, _visited=_visited)
    return _deep_merge(parent, raw)
