from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

T = TypeVar("T")


def retry_policy(max_retries: int, backoff_base_s: float, backoff_max_s: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff_base_s, max=backoff_max_s),
        retry=retry_if_exception_type(Exception),
    )
