"""Component registry for argobot-bench.

Maps (category, type_name) pairs to component classes. Implementations
register themselves with the @register decorator, and the pipeline
resolves them at runtime from config values.

Usage:

    from uwf_rag.core.registry import registry

    @registry.register("chunking", "recursive_langchain")
    class RecursiveChunker(BaseChunker):
        ...

    # Later, in pipeline construction:
    cls = registry.get("chunking", "recursive_langchain")
    instance = cls(config=params)
"""

from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """Global registry mapping (category, type_name) → component class."""

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], type[Any]] = {}

    def register(self, category: str, type_name: str) -> Any:
        """Decorator that registers a component class.

        Args:
            category: The component category (e.g. "chunking", "embedding").
            type_name: The name used in YAML configs
                (e.g. "recursive_langchain", "bge").
        """

        def decorator(cls: type[Any]) -> type[Any]:
            key = (category, type_name)
            if key in self._registry:
                existing = self._registry[key]
                raise ValueError(
                    f"Duplicate registration for {key}: "
                    f"{cls.__name__} conflicts with {existing.__name__}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, category: str, type_name: str) -> type[Any]:
        """Look up a registered component class.

        Raises:
            KeyError: If (category, type_name) is not registered.
        """
        key = (category, type_name)
        if key not in self._registry:
            available = [name for cat, name in self._registry if cat == category]
            raise KeyError(
                f"No component registered for {key}. "
                f"Available types in '{category}': {available}"
            )
        return self._registry[key]

    def list_category(self, category: str) -> list[str]:
        """Return all registered type names for a category."""
        return [name for cat, name in self._registry if cat == category]

    def is_registered(self, category: str, type_name: str) -> bool:
        """Check if a (category, type_name) pair is registered."""
        return (category, type_name) in self._registry

    def clear(self) -> None:
        """Remove all registrations. Useful for testing."""
        self._registry.clear()


# Singleton instance — import this, not the class.
registry = ComponentRegistry()
