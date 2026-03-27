from __future__ import annotations

from datetime import datetime
from pathlib import Path

from adele_runner.runtime.batch_jobs import (
    BatchJobRecord,
    append_batch_job_record,
    latest_batch_job_records,
    make_chunk_id,
    reserved_request_ids,
)


def _record(*, chunk_id: str, status: str, terminal: bool = False, downloaded: bool = False) -> BatchJobRecord:
    now = datetime(2024, 1, 1, 0, 0, 0)
    return BatchJobRecord(
        run_id="run-1",
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
        chunk_id=chunk_id,
        remote_batch_id=f"batch-{chunk_id}",
        request_ids=[f"{chunk_id}-r1", f"{chunk_id}-r2"],
        request_count=2,
        submitted_at=now,
        last_known_status=status,
        status_checked_at=now,
        completed_at=now if terminal else None,
        results_downloaded_at=now if downloaded else None,
        is_terminal=terminal,
        is_successful=True if terminal and downloaded else None,
    )


def test_latest_batch_job_records_returns_latest_snapshot_per_chunk(tmp_path: Path):
    path = tmp_path / "batch_jobs.jsonl"
    append_batch_job_record(path, _record(chunk_id="chunk-a", status="in_progress"))
    append_batch_job_record(path, _record(chunk_id="chunk-a", status="completed", terminal=True))
    append_batch_job_record(path, _record(chunk_id="chunk-b", status="in_progress"))

    records = latest_batch_job_records(
        path,
        run_id="run-1",
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
    )

    assert {record.chunk_id for record in records} == {"chunk-a", "chunk-b"}
    latest = {record.chunk_id: record for record in records}
    assert latest["chunk-a"].last_known_status == "completed"


def test_reserved_request_ids_only_tracks_live_jobs(tmp_path: Path):
    path = tmp_path / "batch_jobs.jsonl"
    append_batch_job_record(path, _record(chunk_id="chunk-a", status="in_progress"))
    append_batch_job_record(path, _record(chunk_id="chunk-b", status="failed", terminal=True))
    append_batch_job_record(path, _record(chunk_id="chunk-c", status="completed", terminal=True, downloaded=True))

    records = latest_batch_job_records(
        path,
        run_id="run-1",
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
    )

    assert reserved_request_ids(records) == {"chunk-a-r1", "chunk-a-r2"}


def test_make_chunk_id_is_stable_across_request_order():
    first = make_chunk_id(
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
        request_ids=["r2", "r1"],
    )
    second = make_chunk_id(
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
        request_ids=["r1", "r2"],
    )

    assert first == second
