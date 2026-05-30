"""Components package.

Importing this package triggers registration of all implementations.
Each implementation file decorates its classes with @registry.register(),
so importing the file is sufficient to make them available.

Import this package at application entry points to populate the registry
before any pipeline code calls registry.get(). For example::

    import components  # noqa: F401

Current entry points: scripts/run_experiment.py, tests/conftest.py.
"""

from components import (  # noqa: F401
    chunkers,
    defaults,
    embedders,
    enrichers,
    generators,
    ingestors,
    lexical_indexes,
    prompts,
    query_transforms,
    rerankers,
    retrievers,
    vectorstores,
)

# Stub files (no registrations yet — implementations in future branches):
# from components import tools
