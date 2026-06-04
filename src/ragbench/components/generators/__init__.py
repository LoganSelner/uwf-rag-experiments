"""Generator implementations.

Importing this package registers all four generators (via their ``@register``
decorators) and re-exports the public surface, so existing call sites
(``from ragbench.components.generators import build_generator`` / a concrete
generator class) keep working unchanged after the split from a single module.
"""

from ragbench.components.generators.base import build_generator
from ragbench.components.generators.edenai import EdenAIGenerator
from ragbench.components.generators.google import GoogleGenerator
from ragbench.components.generators.ollama import OllamaGenerator
from ragbench.components.generators.openai import OpenAIGenerator

__all__ = [
    "EdenAIGenerator",
    "GoogleGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "build_generator",
]
