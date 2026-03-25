"""Reusable execution lanes for request-response and batch adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from adele_runner.config import RateLimitsConfig
from adele_runner.runtime.types import ChatRequest, ChatResponse, ExecutionSettings
from adele_runner.utils.concurrency import AsyncRateLimiter
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
        on_chunk_completed: Callable[[list[ChatResponse]], None] | None = None,
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

        results: list[ChatResponse | BaseException | None] = [None] * len(requests)
        queue: asyncio.Queue[tuple[int, ChatRequest]] = asyncio.Queue()
        stop_dispatch = asyncio.Event()
        callback_error: BaseException | None = None

        for idx, request in enumerate(requests):
            queue.put_nowait((idx, request))

        async def _worker() -> None:
            nonlocal callback_error
            while not stop_dispatch.is_set():
                try:
                    idx, request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                try:
                    if rate_limiter is not None:
                        await rate_limiter.acquire()
                    response = await _send_with_retry(adapter, request, settings, retry_dec)
                    results[idx] = response
                    if on_result is not None:
                        try:
                            on_result(response)
                        except BaseException as callback_exc:  # noqa: BLE001
                            callback_error = callback_exc
                            stop_dispatch.set()
                            return
                except Exception as exc:  # noqa: BLE001
                    results[idx] = exc
                    if on_result is not None:
                        try:
                            on_result(exc)
                        except BaseException as callback_exc:  # noqa: BLE001
                            callback_error = callback_exc
                            stop_dispatch.set()
                            return
                    logger.error("Request-response task failed (skipping): %s", exc)
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(_worker(), name=f"request-response-worker-{idx}")
            for idx in range(max(1, settings.max_in_flight))
        ]

        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        for worker_result in worker_results:
            if isinstance(worker_result, BaseException) and callback_error is None:
                callback_error = worker_result

        if callback_error is not None:
            raise callback_error

        completed: list[ChatResponse] = []
        for result in results:
            if isinstance(result, ChatResponse):
                completed.append(result)
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

        def _handle_chunk(chunk: list[ChatResponse]) -> None:
            for response in chunk:
                if on_result is not None:
                    on_result(response)

        try:
            return await asyncio.to_thread(
                adapter.run_batch,
                requests,
                run_dir,
                settings,
                _handle_chunk,
            )
        except TypeError:
            responses = await asyncio.to_thread(adapter.run_batch, requests, run_dir, settings)
            _handle_chunk(responses)
            return responses
