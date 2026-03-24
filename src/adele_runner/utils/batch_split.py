"""Provider-agnostic batch request splitting."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from adele_runner.runtime.provider_limits import (
    ANTHROPIC_BATCH_MAX_BYTES,
    ANTHROPIC_BATCH_MAX_REQUESTS,
    AZURE_BATCH_MAX_FILE_BYTES,
    AZURE_BATCH_MAX_REQUESTS,
    GOOGLE_GEMINI_BATCH_MAX_BYTES,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_REQUESTS_PER_FILE = AZURE_BATCH_MAX_REQUESTS
DEFAULT_MAX_BYTES_PER_FILE = AZURE_BATCH_MAX_FILE_BYTES


def split_items_by_limits(
    items: list[T],
    *,
    size_getter: Callable[[T], int],
    token_getter: Callable[[T], int] | None = None,
    max_items: int | None = None,
    max_bytes: int | None = None,
    max_tokens: int | None = None,
) -> list[list[T]]:
    """Split items into chunks respecting item-count, byte, and token limits."""
    if not items:
        return []

    chunks: list[list[T]] = []
    current_chunk: list[T] = []
    current_bytes = 0
    current_tokens = 0

    for item in items:
        item_bytes = size_getter(item)
        item_tokens = token_getter(item) if token_getter is not None else 0

        exceeds_items = max_items is not None and len(current_chunk) >= max_items
        exceeds_bytes = max_bytes is not None and current_bytes + item_bytes > max_bytes
        exceeds_tokens = max_tokens is not None and current_tokens + item_tokens > max_tokens
        if current_chunk and (exceeds_items or exceeds_bytes or exceeds_tokens):
            chunks.append(current_chunk)
            current_chunk = []
            current_bytes = 0
            current_tokens = 0

        current_chunk.append(item)
        current_bytes += item_bytes
        current_tokens += item_tokens

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) > 1:
        logger.info(
            "Split %d items into %d batch chunks (max_items=%s, max_bytes=%s, max_tokens=%s)",
            len(items),
            len(chunks),
            max_items,
            max_bytes,
            max_tokens,
        )

    return chunks


def split_batch_requests(
    lines: list[str],
    *,
    max_requests: int = DEFAULT_MAX_REQUESTS_PER_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_FILE,
    token_estimates: list[int] | None = None,
    max_tokens: int | None = None,
) -> list[list[str]]:
    """Split serialized JSONL lines respecting provider limits."""
    estimated_tokens = token_estimates or [0] * len(lines)
    chunks = split_items_by_limits(
        list(zip(lines, estimated_tokens, strict=False)),
        size_getter=lambda item: len(item[0].encode("utf-8")) + 1,
        token_getter=lambda item: item[1],
        max_items=max_requests,
        max_bytes=max_bytes,
        max_tokens=max_tokens,
    )
    return [[line for line, _tokens in chunk] for chunk in chunks]


__all__ = [
    "ANTHROPIC_BATCH_MAX_BYTES",
    "ANTHROPIC_BATCH_MAX_REQUESTS",
    "AZURE_BATCH_MAX_FILE_BYTES",
    "AZURE_BATCH_MAX_REQUESTS",
    "DEFAULT_MAX_BYTES_PER_FILE",
    "DEFAULT_MAX_REQUESTS_PER_FILE",
    "GOOGLE_GEMINI_BATCH_MAX_BYTES",
    "split_batch_requests",
    "split_items_by_limits",
]
