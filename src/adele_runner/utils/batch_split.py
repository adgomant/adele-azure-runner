"""Batch request splitting to respect Azure Batch API limits."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Azure hard limits (absolute maximums enforced by the service).
AZURE_BATCH_MAX_REQUESTS = 100_000
AZURE_BATCH_MAX_FILE_BYTES = 200_000_000  # 200 MB

# Conservative defaults used by config when no override is provided.
DEFAULT_MAX_REQUESTS_PER_FILE = 50_000
DEFAULT_MAX_BYTES_PER_FILE = 100_000_000  # 100 MB


def split_batch_requests(
    lines: list[str],
    *,
    max_requests: int = DEFAULT_MAX_REQUESTS_PER_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_FILE,
) -> list[list[str]]:
    """Split JSONL *lines* into chunks respecting count and size limits.

    Each element of *lines* should be a complete JSONL line (without trailing
    newline).  Returns a list of chunks where each chunk fits within both
    *max_requests* and *max_bytes* (accounting for a newline per line).

    A single oversized line (>= *max_bytes*) is placed in its own chunk so the
    rest of the batch is not blocked.
    """
    if not lines:
        return []

    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_bytes = 0

    for line in lines:
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline

        # Start a new chunk when adding this line would exceed limits
        if current_chunk and (
            len(current_chunk) >= max_requests or current_bytes + line_bytes > max_bytes
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_bytes = 0

        current_chunk.append(line)
        current_bytes += line_bytes

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) > 1:
        logger.info(
            "Split %d requests into %d batch chunks (max_requests=%d, max_bytes=%d)",
            len(lines),
            len(chunks),
            max_requests,
            max_bytes,
        )

    return chunks
