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
from tenacity.wait import wait_base

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


# ---------------------------------------------------------------------------
# Retry-After–aware wait strategy
# ---------------------------------------------------------------------------


def get_retry_after(exc: BaseException) -> float | None:
    """Extract a ``Retry-After`` wait time (seconds) from exception headers.

    Checks ``Retry-After`` (standard HTTP) and ``x-ratelimit-reset-tokens``
    (Azure-specific).  Returns ``None`` when no header is found.
    """
    headers = _get_response_headers(exc)
    if headers is None:
        return None

    # Standard Retry-After (seconds or HTTP-date; we only handle seconds)
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass

    # Azure-specific: x-ratelimit-reset-tokens (seconds as string)
    reset_tokens = headers.get("x-ratelimit-reset-tokens") or headers.get(
        "X-Ratelimit-Reset-Tokens"
    )
    if reset_tokens is not None:
        try:
            # May be "Ns" (e.g., "6s") or just a number
            cleaned = str(reset_tokens).rstrip("s").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            pass

    return None


def _get_response_headers(exc: BaseException) -> dict | None:
    """Try to extract HTTP response headers from various exception types."""
    # Azure SDK: HttpResponseError.response.headers
    try:
        from azure.core.exceptions import HttpResponseError  # noqa: WPS433

        if isinstance(exc, HttpResponseError) and exc.response is not None:
            return dict(exc.response.headers)
    except ImportError:
        pass

    # openai SDK: APIStatusError.response.headers
    try:
        import openai  # noqa: WPS433

        if isinstance(exc, openai.APIStatusError) and exc.response is not None:
            return dict(exc.response.headers)
    except ImportError:
        pass

    # Generic: look for .response.headers
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                return dict(headers)
            except (TypeError, ValueError):
                pass

    return None


class WaitRateLimitAware(wait_base):
    """Use ``Retry-After`` header when available, else exponential backoff.

    When a *rate_limiter* is provided, 429 responses also trigger a global
    backoff via :meth:`~adele_runner.utils.concurrency.AsyncRateLimiter.signal_backoff`
    so that *all* in-flight dispatches pause — not just the retrying task.
    """

    def __init__(
        self,
        backoff_base: float,
        backoff_max: float,
        rate_limiter: object | None = None,
    ) -> None:
        self._exp = wait_exponential(multiplier=backoff_base, max=backoff_max)
        self._rate_limiter = rate_limiter

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            retry_after = get_retry_after(exc)
            if retry_after is not None:
                wait_s = min(retry_after, 300.0)
                logger.info("Using Retry-After header: waiting %.1fs", wait_s)
                if self._rate_limiter is not None:
                    self._rate_limiter.signal_backoff(wait_s)  # type: ignore[attr-defined]
                return wait_s
        return self._exp(retry_state)


def make_retry_decorator(
    max_retries: int = 6,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    retry_filter: Callable[[BaseException], bool] | None = None,
    rate_limiter: object | None = None,
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
    rate_limiter:
        An optional :class:`~adele_runner.utils.concurrency.AsyncRateLimiter`.
        When provided, 429 Retry-After triggers a global backoff that pauses
        all in-flight dispatches.
    """
    if retry_filter is None:
        retry_filter = is_retryable

    return retry(
        retry=retry_if_exception_type(retry_exceptions) & retry_if_exception(retry_filter),
        stop=stop_after_attempt(max_retries + 1),
        wait=WaitRateLimitAware(
            backoff_base=backoff_base, backoff_max=backoff_max, rate_limiter=rate_limiter
        ),
        before_sleep=_log_retry,
        reraise=True,
    )
