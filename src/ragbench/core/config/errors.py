"""Config error type.

Leaf module with no internal imports, so both :mod:`loading` (cycle detection)
and :mod:`validation` can raise it without creating an import cycle.
"""

from __future__ import annotations


class ConfigValidationError(Exception):
    """Raised when experiment config validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(
            f"Config validation failed with {len(errors)} error(s):\n{bullet_list}"
        )
