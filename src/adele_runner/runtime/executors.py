"""Reusable execution lanes for request-response and batch adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from adele_runner.adapters.factory import build_batch_adapter, build_request_response_adapter
from adele_runner.config import AppConfig, RateLimitsConfig
from adele_runner.runtime.types import AdapterKind, ChatRequest, ChatResponse, ExecutionSettings
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

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def execute(
        self,
        *,
        adapter_kind: AdapterKind,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
        rate_limits: RateLimitsConfig | None = None,
        on_result: ExecutionCallback | None = None,
    ) -> list[ChatResponse]:
        if not requests:
            return []

        rate_limiter: AsyncRateLimiter | None = None
        if settings.effective_rpm is not None:
            rate_limiter = AsyncRateLimiter(
                settings.effective_rpm,
                tpm=rate_limits.tokens_per_minute if rate_limits else None,
            )
            logger.info(
                "Rate limiter enabled for %s: %d RPM",
                adapter_kind,
                settings.effective_rpm,
            )

        adapter = build_request_response_adapter(
            adapter_kind,
            self._config,
            rate_limiter=rate_limiter,
            configured_rate_limits=rate_limits,
        )
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

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def execute(
        self,
        *,
        adapter_kind: AdapterKind,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
        on_result: ExecutionCallback | None = None,
    ) -> list[ChatResponse]:
        if not requests:
            return []

        adapter = build_batch_adapter(adapter_kind, self._config)
        responses = await asyncio.to_thread(adapter.run_batch, requests, run_dir, settings)
        for response in responses:
            if on_result is not None:
                on_result(response)
        return responses
