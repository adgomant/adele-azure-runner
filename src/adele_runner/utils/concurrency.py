"""Async bounded concurrency helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TOKEN_SAMPLE_WINDOW = 50  # rolling average window size
_TOKEN_SAMPLE_MIN = 10  # min samples before recomputing base interval


def _parse_int(value: str | None) -> int | None:
    """Parse an integer from a header value, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """Adaptive token-bucket rate limiter with header feedback and token tracking.

    Requests are paced at a base interval (``60 / rpm`` seconds) that is
    dynamically adjusted via two feedback loops:

    1. **Header pressure** — :meth:`update_from_headers` reads
       ``x-ratelimit-remaining-*`` headers and scales the interval up when
       remaining capacity is low (reactive, prevents 429).
    2. **Token tracking** — :meth:`update_token_usage` maintains a rolling
       average of actual tokens per request and recalculates the TPM-implied
       RPM, potentially speeding up the pacing (predictive).

    The effective interval is ``base_interval × pressure_multiplier``.

    When a 429 is received, :meth:`signal_backoff` pauses *all* future
    dispatches for the specified duration.
    """

    def __init__(self, rpm: int, tpm: int | None = None) -> None:
        if rpm < 1:
            raise ValueError(f"rpm must be >= 1, got {rpm}")
        self._rpm = rpm
        self._tpm = tpm
        self._base_interval = 60.0 / rpm
        self._interval = self._base_interval
        self._lock = asyncio.Lock()
        self._next_slot = 0.0  # monotonic timestamp
        self._backoff_until = 0.0  # monotonic timestamp

        # Header-based adaptive feedback
        self._limit_requests: int | None = None
        self._limit_tokens: int | None = None
        self._pressure_multiplier = 1.0

        # Token-usage rolling average
        self._token_samples: deque[int] = deque(maxlen=_TOKEN_SAMPLE_WINDOW)

    async def acquire(self) -> None:
        """Wait until the next available request slot."""
        async with self._lock:
            now = time.monotonic()
            target = max(self._next_slot, self._backoff_until, now)
            self._next_slot = target + self._interval
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def signal_backoff(self, seconds: float) -> None:
        """Pause all future dispatches for *seconds* (called on 429)."""
        deadline = time.monotonic() + seconds
        # Simple float assignment — safe to call from sync tenacity callbacks.
        self._backoff_until = max(self._backoff_until, deadline)

    # -- Adaptive feedback (sync — called from response hooks) -------------

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Adjust pacing based on ``x-ratelimit-remaining-*`` response headers.

        Called synchronously from the Azure SDK ``raw_response_hook``.
        """
        remaining_req = _parse_int(headers.get("x-ratelimit-remaining-requests"))
        remaining_tok = _parse_int(headers.get("x-ratelimit-remaining-tokens"))
        limit_req = _parse_int(headers.get("x-ratelimit-limit-requests"))
        limit_tok = _parse_int(headers.get("x-ratelimit-limit-tokens"))

        # Store limits on first observation
        if limit_req is not None:
            self._limit_requests = limit_req
        if limit_tok is not None:
            self._limit_tokens = limit_tok

        # Compute pressure ratio — tighter of request and token remaining
        ratios: list[float] = []
        if remaining_req is not None and self._limit_requests:
            ratios.append(remaining_req / self._limit_requests)
        if remaining_tok is not None and self._limit_tokens:
            ratios.append(remaining_tok / self._limit_tokens)

        if not ratios:
            return

        pressure = min(ratios)
        if pressure < 0.05:
            new_mult = 5.0
        elif pressure < 0.15:
            new_mult = 3.0
        elif pressure < 0.30:
            new_mult = 1.5
        else:
            new_mult = 1.0

        if new_mult != self._pressure_multiplier:
            self._pressure_multiplier = new_mult
            self._interval = self._base_interval * self._pressure_multiplier
            if new_mult > 1.0:
                logger.info(
                    "Rate limiter pressure: remaining=%.0f%% → interval ×%.1f",
                    pressure * 100,
                    new_mult,
                )

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record actual token usage and refine the base pacing interval.

        After enough samples, recomputes ``effective_rpm`` from the rolling
        average of actual tokens per request vs. the TPM budget.  The base
        interval is only loosened (never below ``60 / rpm``).
        """
        total = prompt_tokens + completion_tokens
        if total <= 0:
            return
        self._token_samples.append(total)

        if len(self._token_samples) < _TOKEN_SAMPLE_MIN or self._tpm is None:
            return

        avg_tokens = sum(self._token_samples) / len(self._token_samples)
        rpm_from_tpm = self._tpm / max(1.0, avg_tokens)
        new_effective_rpm = max(1, int(min(self._rpm, rpm_from_tpm)))
        new_base = 60.0 / new_effective_rpm

        if new_base != self._base_interval:
            logger.info(
                "Rate limiter adjusted: avg_tokens=%.0f effective_rpm=%d (was %d)",
                avg_tokens,
                new_effective_rpm,
                max(1, int(60.0 / self._base_interval)),
            )
            self._base_interval = new_base
            self._interval = self._base_interval * self._pressure_multiplier


# ---------------------------------------------------------------------------
# Bounded gather
# ---------------------------------------------------------------------------


async def bounded_gather(
    tasks: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 16,
    rate_limiter: AsyncRateLimiter | None = None,
) -> list[T | BaseException]:
    """Run coroutines with bounded concurrency.

    When *rate_limiter* is provided, each task acquires a rate-limit token
    **before** entering the semaphore so that dispatch is paced over time.

    Returns results in the same order as ``tasks``.  Exceptions are returned
    as values (not raised) so the caller can decide how to handle them.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[T | BaseException] = [None] * len(tasks)  # type: ignore[list-item]

    async def _run(idx: int, coro: Coroutine[Any, Any, T]) -> None:
        if rate_limiter is not None:
            await rate_limiter.acquire()
        async with semaphore:
            try:
                results[idx] = await coro
            except Exception as exc:  # noqa: BLE001
                logger.error("Task %d failed: %s", idx, exc)
                results[idx] = exc

    await asyncio.gather(*(_run(i, t) for i, t in enumerate(tasks)))
    return results
