"""Metadata-filter matching shared by the post-retrieval filter paths.

FAISS and BM25 have no native metadata filtering, so both over-retrieve and
post-filter in Python. This module is the single definition of what a filter
dict *means*, so dense and sparse retrieval interpret one identically — a filter
that scopes the dense branch scopes the BM25 branch the same way (keeping
retrieval method a freely swappable config variable). ChromaDB filters natively
via its ``where`` syntax (built in ``vectorstores.py``); it mirrors these same
semantics rather than calling this helper.

No business logic beyond the match itself lives here.
"""

from __future__ import annotations

from typing import Any


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if ``metadata`` satisfies every criterion in ``filters``.

    A scalar filter value requires equality; a list/tuple/set value matches by
    membership (``$in`` semantics), so a single filter can scope to a *group* of
    values — e.g. ``{"source_name": ["knowledge_base", "student_handbook"]}``
    selects either source, identically on dense and sparse retrieval.
    """
    for key, value in filters.items():
        actual = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True
