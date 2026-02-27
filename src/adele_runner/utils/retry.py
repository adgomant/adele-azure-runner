"""Retry / backoff helpers built on tenacity."""

from __future__ import annotations

import logging
from collections.abc import Callable

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Python logic errors that should never be retried.
_NON_RETRYABLE_LOGIC_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

# HTTP status codes that are never retryable (auth / validation).
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 422})

# HTTP status codes that are always retryable (rate-limit / server errors).
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _get_status_code(exc: BaseException) -> int | None:
    """Try to extract an HTTP status code from a variety of exception types."""

    # --- httpx ---
    try:
        import httpx  # noqa: WPS433

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return None  # connection / timeout -> retryable (handled below)
    except ImportError:
        pass

    # --- openai ---
    try:
        import openai  # noqa: WPS433

        if isinstance(exc, openai.APIStatusError):
            return exc.status_code
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return None  # connection / timeout -> retryable
    except ImportError:
        pass

    # --- Azure SDK (azure.core.exceptions) ---
    try:
        from azure.core.exceptions import HttpResponseError  # noqa: WPS433

        if isinstance(exc, HttpResponseError):
            return exc.status_code
    except ImportError:
        pass

    # --- Generic fallback: look for common attributes ---
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    return None


def _is_connection_or_timeout(exc: BaseException) -> bool:
    """Return True when *exc* looks like a connection or timeout error."""

    # Explicit library types checked first (already handled in
    # ``_get_status_code`` returning ``None``), but we also do a
    # string-based heuristic for libraries we cannot import.
    try:
        import httpx  # noqa: WPS433

        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass

    try:
        import openai  # noqa: WPS433

        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
    except ImportError:
        pass

    # String-based heuristic for other libraries.
    exc_type_name = type(exc).__name__.lower()
    exc_str = str(exc).lower()
    connection_keywords = ("timeout", "timed out", "connection", "connect", "reset", "broken pipe")
    return any(kw in exc_type_name or kw in exc_str for kw in connection_keywords)


def is_retryable(exc: BaseException) -> bool:
    """Decide whether *exc* should be retried.

    Rules (evaluated in order):
    1. Python logic errors (``ValueError``, ``TypeError``, …) -> **not** retryable.
    2. Connection / timeout errors -> **retryable**.
    3. HTTP 429 or 5xx -> **retryable**.
    4. HTTP 400, 401, 403, 404, 422 -> **not** retryable.
    5. Unknown exception types -> **retryable** (fail open).
    """

    # 1. Never retry obvious programming mistakes.
    if isinstance(exc, _NON_RETRYABLE_LOGIC_ERRORS):
        return False

    # 2. Connection / timeout -> always retry.
    if _is_connection_or_timeout(exc):
        return True

    # 3 & 4. Check HTTP status code if available.
    status = _get_status_code(exc)
    if status is not None:
        if status in _RETRYABLE_STATUS_CODES or status >= 500:
            return True
        if status in _NON_RETRYABLE_STATUS_CODES or 400 <= status < 500:
            return False

    # 5. Unknown -> retry by default.
    return True


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        "Retry attempt %d after error: %s",
        state.attempt_number,
        exc,
    )


def make_retry_decorator(
    max_retries: int = 6,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    retry_filter: Callable[[BaseException], bool] | None = None,
) -> Callable:
    """Return a tenacity ``retry`` decorator configured for the given params.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts (not counting the initial call).
    backoff_base:
        Multiplier for the exponential back-off.
    backoff_max:
        Upper bound (seconds) for the back-off wait.
    retry_exceptions:
        Tuple of exception types that are eligible for retrying.  This acts as
        a coarse pre-filter; the *retry_filter* callable further narrows which
        exceptions are actually retried.
    retry_filter:
        A callable ``(BaseException) -> bool`` that decides whether a given
        exception should be retried.  Defaults to :func:`is_retryable`.
    """
    if retry_filter is None:
        retry_filter = is_retryable

    return retry(
        retry=retry_if_exception_type(retry_exceptions) & retry_if_exception(retry_filter),
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=backoff_base, max=backoff_max),
        before_sleep=_log_retry,
        reraise=True,
    )
