"""Reusable execution lanes for request-response and batch adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from adele_runner.config import RateLimitsConfig
from adele_runner.runtime.types import ChatRequest, ChatResponse, ExecutionSettings
from adele_runner.utils.concurrency import AsyncRateLimiter, bounded_gather
from adele_runner.utils.retry import make_retry_decorator

logger = logging.getLogger(__name__)

ExecutionCallback = Callable[[ChatResponse | BaseException], None]


class RequestResponseTransport(Protocol):
    """Minimal request-response transport interface."""

    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse: ...


class BatchTransport(Protocol):
    """Minimal batch transport interface."""

    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
    ) -> list[ChatResponse]: ...


def create_rate_limiter(
    settings: ExecutionSettings,
    rate_limits: RateLimitsConfig | None = None,
) -> AsyncRateLimiter | None:
    """Create a request pacing limiter for a lane if needed."""
    if settings.request_budget is None:
        if settings.effective_rpm is None:
            return None
        return AsyncRateLimiter(
            settings.effective_rpm,
            tpm=rate_limits.tokens_per_minute if rate_limits else None,
        )
    return AsyncRateLimiter(settings.request_budget)


async def _send_with_retry(
    adapter: RequestResponseTransport,
    request: ChatRequest,
    settings: ExecutionSettings,
    retry_decorator,  # noqa: ANN001
) -> ChatResponse:
    @retry_decorator
    async def _call() -> ChatResponse:
        return await adapter.send(request, timeout_s=settings.request_timeout_s)

    return await _call()


class RequestResponseExecutor:
    """Shared executor for async request-response transports."""

    async def execute(
        self,
        *,
        adapter: RequestResponseTransport,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
        rate_limiter: AsyncRateLimiter | None = None,
        on_result: ExecutionCallback | None = None,
    ) -> list[ChatResponse]:
        if not requests:
            return []

        retry_dec = make_retry_decorator(
            max_retries=settings.max_retries,
            backoff_base=settings.backoff_base_s,
            backoff_max=settings.backoff_max_s,
            rate_limiter=rate_limiter,
        )

        completed: list[ChatResponse] = []
        chunk_size = settings.max_in_flight * 4
        for chunk_start in range(0, len(requests), chunk_size):
            chunk = requests[chunk_start : chunk_start + chunk_size]
            tasks = [
                _send_with_retry(adapter, request, settings, retry_dec) for request in chunk
            ]
            results = await bounded_gather(
                tasks,
                max_concurrency=settings.max_in_flight,
                rate_limiter=rate_limiter,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.error("Request-response task failed (skipping): %s", result)
                    if on_result is not None:
                        on_result(result)
                    continue
                completed.append(result)
                if on_result is not None:
                    on_result(result)

        return completed


class BatchExecutor:
    """Shared executor for blocking batch transports."""

    async def execute(
        self,
        *,
        adapter: BatchTransport,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
        on_result: ExecutionCallback | None = None,
    ) -> list[ChatResponse]:
        if not requests:
            return []

        responses = await asyncio.to_thread(adapter.run_batch, requests, run_dir, settings)
        for response in responses:
            if on_result is not None:
                on_result(response)
        return responses
