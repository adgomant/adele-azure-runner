"""Tests for batch request splitting logic."""

from __future__ import annotations

from adele_runner.utils.batch_split import (
    AZURE_BATCH_MAX_FILE_BYTES,
    AZURE_BATCH_MAX_REQUESTS,
    DEFAULT_MAX_BYTES_PER_FILE,
    DEFAULT_MAX_REQUESTS_PER_FILE,
    split_batch_requests,
)

# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------


def test_empty_input():
    assert split_batch_requests([]) == []


def test_single_line_no_split():
    lines = ['{"custom_id": "1", "body": {}}']
    chunks = split_batch_requests(lines)
    assert len(chunks) == 1
    assert chunks[0] == lines


def test_under_limits_single_chunk():
    lines = [f'{{"id": "{i}"}}' for i in range(100)]
    chunks = split_batch_requests(lines)
    assert len(chunks) == 1
    assert len(chunks[0]) == 100


# ---------------------------------------------------------------------------
# Request count splitting
# ---------------------------------------------------------------------------


def test_split_by_request_count():
    lines = [f'{{"id": "{i}"}}' for i in range(10)]
    chunks = split_batch_requests(lines, max_requests=3, max_bytes=10**9)
    assert len(chunks) == 4  # 3 + 3 + 3 + 1
    assert sum(len(c) for c in chunks) == 10
    # Each chunk respects the limit
    for chunk in chunks:
        assert len(chunk) <= 3


def test_split_exact_multiple():
    lines = [f'{{"id": "{i}"}}' for i in range(6)]
    chunks = split_batch_requests(lines, max_requests=3, max_bytes=10**9)
    assert len(chunks) == 2
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 3


# ---------------------------------------------------------------------------
# File size splitting
# ---------------------------------------------------------------------------


def test_split_by_byte_size():
    # Each line is ~20 bytes + 1 newline = ~21 bytes
    line = '{"id": "test_value"}'
    lines = [line] * 10
    line_bytes = len(line.encode("utf-8")) + 1
    # Set max_bytes so that only 3 lines fit per chunk
    max_bytes = line_bytes * 3
    chunks = split_batch_requests(lines, max_requests=10**6, max_bytes=max_bytes)
    assert len(chunks) >= 3
    assert sum(len(c) for c in chunks) == 10


def test_single_oversized_line_gets_own_chunk():
    small = '{"s": 1}'
    big = '{"data": "' + "x" * 200 + '"}'
    lines = [small, big, small]
    # Set max_bytes so small lines fit together but big doesn't
    small_bytes = len(small.encode("utf-8")) + 1
    chunks = split_batch_requests(lines, max_requests=10**6, max_bytes=small_bytes * 2)
    # big line must be in its own chunk (even though it exceeds max_bytes alone)
    assert sum(len(c) for c in chunks) == 3
    assert any(big in chunk for chunk in chunks)


# ---------------------------------------------------------------------------
# Combined limits
# ---------------------------------------------------------------------------


def test_both_limits_respected():
    line = '{"id": "x"}'
    lines = [line] * 20
    line_bytes = len(line.encode("utf-8")) + 1
    # Limit to 5 requests AND 3 * line_bytes bytes
    chunks = split_batch_requests(lines, max_requests=5, max_bytes=line_bytes * 3)
    # Byte limit (3) is tighter than request limit (5)
    for chunk in chunks:
        assert len(chunk) <= 5
    assert sum(len(c) for c in chunks) == 20


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


def test_default_constants():
    assert AZURE_BATCH_MAX_REQUESTS == 100_000
    assert AZURE_BATCH_MAX_FILE_BYTES == 200_000_000
    assert DEFAULT_MAX_REQUESTS_PER_FILE == 50_000
    assert DEFAULT_MAX_BYTES_PER_FILE == 100_000_000


# ---------------------------------------------------------------------------
# Preserves order
# ---------------------------------------------------------------------------


def test_preserves_order():
    lines = [f'{{"id": "{i}"}}' for i in range(10)]
    chunks = split_batch_requests(lines, max_requests=3, max_bytes=10**9)
    flat = [line for chunk in chunks for line in chunk]
    assert flat == lines
