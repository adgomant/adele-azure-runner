"""Async bounded concurrency helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from adele_runner.runtime.provider_limits import RequestBudget

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TOKEN_SAMPLE_WINDOW = 50
_TOKEN_SAMPLE_MIN = 10


def _parse_int(value: str | None) -> int | None:
    """Parse an integer from a header value, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class AsyncRateLimiter:
    """Adaptive limiter with provider header feedback and day-window accounting."""

    def __init__(self, budget_or_rpm: RequestBudget | int, tpm: int | None = None) -> None:
        if isinstance(budget_or_rpm, RequestBudget):
            self._budget = budget_or_rpm
            configured_rpm = budget_or_rpm.requests_per_minute
            configured_tpm = budget_or_rpm.tokens_per_minute
        else:
            if budget_or_rpm < 1:
                raise ValueError(f"rpm must be >= 1, got {budget_or_rpm}")
            self._budget = RequestBudget(
                provider_kind="azure_openai",
                requests_per_minute=budget_or_rpm,
                tokens_per_minute=tpm,
                estimated_input_tokens=1,
                estimated_output_tokens=1,
                enable_header_feedback=True,
            )
            configured_rpm = budget_or_rpm
            configured_tpm = tpm

        self._rpm = configured_rpm or 0
        self._tpm = configured_tpm
        self._token_samples: deque[int] = deque(maxlen=_TOKEN_SAMPLE_WINDOW)
        self._input_token_samples: deque[int] = deque(maxlen=_TOKEN_SAMPLE_WINDOW)
        self._output_token_samples: deque[int] = deque(maxlen=_TOKEN_SAMPLE_WINDOW)
        self._base_interval = self._compute_base_interval()
        self._interval = self._base_interval
        self._lock = asyncio.Lock()
        self._next_slot = 0.0
        self._backoff_until = 0.0

        self._limit_requests: int | None = None
        self._limit_tokens: int | None = None
        self._limit_input_tokens: int | None = None
        self._limit_output_tokens: int | None = None
        self._pressure_multiplier = 1.0

        self._day_requests_used = 0
        self._day_tokens_used = 0
        self._day_window_key = self._current_day_key()

    def _now_utc(self) -> datetime:
        return datetime.now(UTC)

    def _daily_tz_name(self) -> str | None:
        if self._budget.daily_reset_timezone is not None:
            return self._budget.daily_reset_timezone
        if self._budget.requests_per_day is not None or self._budget.tokens_per_day is not None:
            return "UTC"
        return None

    def _current_day_key(self) -> datetime.date | None:
        tz_name = self._daily_tz_name()
        if tz_name is None:
            return None
        return self._now_utc().astimezone(ZoneInfo(tz_name)).date()

    def _next_day_reset(self) -> datetime | None:
        tz_name = self._daily_tz_name()
        if tz_name is None:
            return None
        local_now = self._now_utc().astimezone(ZoneInfo(tz_name))
        next_midnight = datetime.combine(
            local_now.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=local_now.tzinfo,
        )
        return next_midnight.astimezone(UTC)

    def _refresh_day_window(self) -> None:
        day_key = self._current_day_key()
        if day_key is None:
            return
        if self._day_window_key != day_key:
            self._day_window_key = day_key
            self._day_requests_used = 0
            self._day_tokens_used = 0

    def _compute_base_interval(self) -> float:
        rpm_candidates: list[float] = []
        est_total = max(1, self._average_total_tokens())
        est_input = max(1, self._average_input_tokens())
        est_output = max(1, self._average_output_tokens())

        if self._budget.requests_per_minute is not None:
            rpm_candidates.append(float(self._budget.requests_per_minute))
        if self._budget.tokens_per_minute is not None:
            rpm_candidates.append(self._budget.tokens_per_minute / est_total)
        if self._budget.input_tokens_per_minute is not None:
            rpm_candidates.append(self._budget.input_tokens_per_minute / est_input)
        if self._budget.output_tokens_per_minute is not None:
            rpm_candidates.append(self._budget.output_tokens_per_minute / est_output)
        if self._budget.requests_per_day is not None:
            rpm_candidates.append(self._budget.requests_per_day / 1440.0)
        if self._budget.tokens_per_day is not None:
            rpm_candidates.append(self._budget.tokens_per_day / (est_total * 1440.0))

        if not rpm_candidates:
            return 0.0
        effective_rpm = max(1.0, min(rpm_candidates))
        return 60.0 / effective_rpm

    def _average_total_tokens(self) -> float:
        if self._token_samples:
            return sum(self._token_samples) / len(self._token_samples)
        return max(1, self._budget.estimated_total_tokens)

    def _average_input_tokens(self) -> float:
        if self._input_token_samples:
            return sum(self._input_token_samples) / len(self._input_token_samples)
        return max(1, self._budget.estimated_input_tokens)

    def _average_output_tokens(self) -> float:
        if self._output_token_samples:
            return sum(self._output_token_samples) / len(self._output_token_samples)
        return max(1, self._budget.estimated_output_tokens)

    def _daily_delay_seconds(self) -> float:
        self._refresh_day_window()
        request_limit = self._budget.requests_per_day
        token_limit = self._budget.tokens_per_day
        next_reset = self._next_day_reset()
        if next_reset is None:
            return 0.0

        reserve_requests = 1
        reserve_tokens = self._budget.estimated_total_tokens
        if request_limit is not None and self._day_requests_used + reserve_requests > request_limit:
            return max(0.0, (next_reset - self._now_utc()).total_seconds())
        if token_limit is not None and self._day_tokens_used + reserve_tokens > token_limit:
            return max(0.0, (next_reset - self._now_utc()).total_seconds())
        return 0.0

    def _reserve_daily_usage(self) -> None:
        self._refresh_day_window()
        if self._budget.requests_per_day is not None:
            self._day_requests_used += 1
        if self._budget.tokens_per_day is not None:
            self._day_tokens_used += self._budget.estimated_total_tokens

    async def acquire(self) -> None:
        """Wait until the next available request slot."""
        while True:
            async with self._lock:
                now = time.monotonic()
                daily_delay = self._daily_delay_seconds()
                if daily_delay > 0:
                    target = max(self._backoff_until, now + daily_delay)
                    delay = target - time.monotonic()
                else:
                    target = max(self._next_slot, self._backoff_until, now)
                    self._next_slot = target + self._interval
                    self._reserve_daily_usage()
                    delay = target - time.monotonic()
                    break
            if delay > 0:
                await asyncio.sleep(delay)

        if delay > 0:
            await asyncio.sleep(delay)

    def signal_backoff(self, seconds: float) -> None:
        """Pause all future dispatches for *seconds* (called on 429)."""
        deadline = time.monotonic() + seconds
        self._backoff_until = max(self._backoff_until, deadline)

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Adjust pacing based on provider response headers."""
        normalized = {key.lower(): value for key, value in headers.items()}

        remaining_pairs = [
            (
                _parse_int(normalized.get("x-ratelimit-remaining-requests")),
                _parse_int(normalized.get("x-ratelimit-limit-requests")),
            ),
            (
                _parse_int(normalized.get("x-ratelimit-remaining-tokens")),
                _parse_int(normalized.get("x-ratelimit-limit-tokens")),
            ),
            (
                _parse_int(normalized.get("anthropic-ratelimit-requests-remaining")),
                _parse_int(normalized.get("anthropic-ratelimit-requests-limit")),
            ),
            (
                _parse_int(normalized.get("anthropic-ratelimit-tokens-remaining")),
                _parse_int(normalized.get("anthropic-ratelimit-tokens-limit")),
            ),
            (
                _parse_int(normalized.get("anthropic-ratelimit-input-tokens-remaining")),
                _parse_int(normalized.get("anthropic-ratelimit-input-tokens-limit")),
            ),
            (
                _parse_int(normalized.get("anthropic-ratelimit-output-tokens-remaining")),
                _parse_int(normalized.get("anthropic-ratelimit-output-tokens-limit")),
            ),
        ]

        ratios = [
            remaining / limit
            for remaining, limit in remaining_pairs
            if remaining is not None and limit not in {None, 0}
        ]

        if ratios:
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
                        "Rate limiter pressure: remaining=%.0f%% -> interval x%.1f",
                        pressure * 100,
                        new_mult,
                    )

        reset_headers = (
            normalized.get("anthropic-ratelimit-requests-reset"),
            normalized.get("anthropic-ratelimit-tokens-reset"),
            normalized.get("anthropic-ratelimit-input-tokens-reset"),
            normalized.get("anthropic-ratelimit-output-tokens-reset"),
        )
        for reset_header in reset_headers:
            reset_at = _parse_rfc3339(reset_header)
            if reset_at is None:
                continue
            wait_s = max(0.0, (reset_at.astimezone(UTC) - self._now_utc()).total_seconds())
            if wait_s > 0:
                self.signal_backoff(min(wait_s, 300.0))
                break

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record actual token usage and refine the base pacing interval."""
        total = prompt_tokens + completion_tokens
        if total <= 0:
            return

        self._token_samples.append(total)
        if prompt_tokens > 0:
            self._input_token_samples.append(prompt_tokens)
        if completion_tokens > 0:
            self._output_token_samples.append(completion_tokens)

        if self._budget.tokens_per_day is not None:
            delta = total - self._budget.estimated_total_tokens
            self._day_tokens_used = max(0, self._day_tokens_used + delta)

        budgeted_dimensions = any(
            value is not None
            for value in (
                self._budget.tokens_per_minute,
                self._budget.input_tokens_per_minute,
                self._budget.output_tokens_per_minute,
                self._budget.tokens_per_day,
            )
        )
        if len(self._token_samples) < _TOKEN_SAMPLE_MIN or not budgeted_dimensions:
            return

        new_base = self._compute_base_interval()
        if new_base != self._base_interval:
            old_effective = 0 if self._base_interval == 0 else max(1, int(60.0 / self._base_interval))
            new_effective = 0 if new_base == 0 else max(1, int(60.0 / new_base))
            logger.info(
                "Rate limiter adjusted: avg_tokens=%.0f effective_rpm=%d (was %d)",
                self._average_total_tokens(),
                new_effective,
                old_effective,
            )
            self._base_interval = new_base
            self._interval = self._base_interval * self._pressure_multiplier


async def bounded_gather(
    tasks: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 16,
    rate_limiter: AsyncRateLimiter | None = None,
) -> list[T | BaseException]:
    """Run coroutines with bounded concurrency."""
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
