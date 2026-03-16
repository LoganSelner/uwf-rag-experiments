"""Components package.

Importing this package triggers registration of all implementations.
Each implementation file decorates its classes with @registry.register(),
so importing the file is sufficient to make them available.
"""

from components import defaults  # noqa: F401

# Future implementation imports will go here:
# from components import ingestors
# from components import chunkers
# from components import embedders
# from components import vectorstores
# from components import retrievers
# from components import rerankers
# from components import generators
# from components import prompts
# from components import tools
