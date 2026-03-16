"""Components package.

Importing this package triggers registration of all implementations.
Each implementation file decorates its classes with @registry.register(),
so importing the file is sufficient to make them available.
"""

from components import (  # noqa: F401
    chunkers,
    defaults,
    embedders,
    generators,
    ingestors,
    prompts,
    retrievers,
    vectorstores,
)

# Stub files (no registrations yet — implementations in future branches):
# from components import rerankers
# from components import tools
