"""Async bounded concurrency helpers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def bounded_gather(
    tasks: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 16,
) -> list[T | BaseException]:
    """Run coroutines with bounded concurrency.

    Returns results in the same order as ``tasks``.  Exceptions are returned
    as values (not raised) so the caller can decide how to handle them.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[T | BaseException] = [None] * len(tasks)  # type: ignore[list-item]

    async def _run(idx: int, coro: Coroutine[Any, Any, T]) -> None:
        async with semaphore:
            try:
                results[idx] = await coro
            except Exception as exc:  # noqa: BLE001
                logger.error("Task %d failed: %s", idx, exc)
                results[idx] = exc

    await asyncio.gather(*(_run(i, t) for i, t in enumerate(tasks)))
    return results
