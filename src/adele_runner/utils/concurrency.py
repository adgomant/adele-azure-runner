from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def bounded_gather(limit: int, factories: list[Callable[[], Awaitable[T]]]) -> list[T]:
    semaphore = asyncio.Semaphore(limit)

    async def wrapped(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks = [asyncio.create_task(wrapped(factory)) for factory in factories]
    return await asyncio.gather(*tasks)
