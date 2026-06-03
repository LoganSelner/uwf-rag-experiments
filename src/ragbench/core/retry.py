"""Shared retry policy for transient external-service failures.

One definition reused by every SDK/HTTP client wrapper (generators, embedders)
so the policy can't drift between modules. The retry predicate is deliberately
**type-based and provider-agnostic**: it retries anything that is not an obvious
programming error (``TypeError``/``ValueError``/...). This keeps the module in
``core`` with no provider-SDK imports — teaching it to recognize provider auth
exceptions (to *not* retry a 401/4xx) would invert the dependency rule. The
consequence is a known, accepted minor inefficiency: an auth/4xx failure retries
the full budget before surfacing.
"""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Programming errors that will not recover on retry — re-raised immediately.
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    SyntaxError,
)

# Decorator applied to a single external call (one HTTP/SDK request). Exponential
# backoff, 4 attempts, re-raises the last exception so callers wrap it in a
# domain error.
retry_transient = retry(
    retry=retry_if_exception(lambda e: not isinstance(e, NON_RETRYABLE)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
