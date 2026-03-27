"""Persistent batch job records for resumable batch execution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from adele_runner.runtime.types import ProviderKind
from adele_runner.utils.io import append_jsonl, iter_jsonl

BatchStage = Literal["inference", "judging"]

_CHUNK_ID_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class BatchJobRecord(BaseModel):
    """Append-only snapshot of one remote batch chunk submission."""

    run_id: str
    stage: BatchStage
    provider: ProviderKind
    judge_name: str | None = None
    chunk_id: str
    remote_batch_id: str
    request_ids: list[str] = Field(default_factory=list)
    request_count: int
    submitted_at: datetime | None = None
    last_known_status: str | None = None
    status_checked_at: datetime | None = None
    completed_at: datetime | None = None
    results_downloaded_at: datetime | None = None
    terminal_error: str | None = None
    input_artifact_path: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    is_terminal: bool = False
    is_successful: bool | None = None

    @property
    def needs_recovery(self) -> bool:
        return self.results_downloaded_at is None

    @property
    def reserves_requests(self) -> bool:
        return self.results_downloaded_at is None and not self.is_terminal


def append_batch_job_record(path: Path, record: BatchJobRecord) -> None:
    """Append one batch-job snapshot to disk."""
    append_jsonl(path, record)


def iter_batch_job_records(path: Path) -> Iterable[BatchJobRecord]:
    """Stream batch job snapshots from disk."""
    return iter_jsonl(path, BatchJobRecord)


def latest_batch_job_records(
    path: Path,
    *,
    run_id: str,
    stage: BatchStage,
    provider: ProviderKind | None = None,
    judge_name: str | None = None,
) -> list[BatchJobRecord]:
    """Return the latest snapshot for each chunk matching the given filters."""
    latest: dict[str, BatchJobRecord] = {}
    for record in iter_batch_job_records(path):
        if record.run_id != run_id or record.stage != stage:
            continue
        if provider is not None and record.provider != provider:
            continue
        if judge_name is not None and record.judge_name != judge_name:
            continue
        latest[record.chunk_id] = record
    return list(latest.values())


def reserved_request_ids(records: Iterable[BatchJobRecord]) -> set[str]:
    """Collect request IDs reserved by still-live batch jobs."""
    reserved: set[str] = set()
    for record in records:
        if record.reserves_requests:
            reserved.update(record.request_ids)
    return reserved


def make_chunk_id(
    *,
    stage: BatchStage,
    provider: ProviderKind,
    request_ids: Iterable[str],
    judge_name: str | None = None,
) -> str:
    """Create a deterministic chunk identifier from the request IDs."""
    request_list = sorted(str(request_id) for request_id in request_ids)
    digest = hashlib.sha256("\n".join(request_list).encode("utf-8")).hexdigest()[:16]
    parts = [stage, provider]
    if judge_name:
        label = _CHUNK_ID_ALLOWED_RE.sub("-", judge_name).strip("-")
        if label:
            parts.append(label[:24])
    parts.append(digest)
    return "-".join(parts)
