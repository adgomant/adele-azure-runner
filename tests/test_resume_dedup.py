"""Tests for JSONL resume / dedup logic."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from adele_runner.schemas import InferenceOutput
from adele_runner.utils.io import append_jsonl, build_dedup_index, iter_jsonl


def _make_output(instance_id: str, model_id: str) -> InferenceOutput:
    return InferenceOutput(
        instance_id=instance_id,
        model_id=model_id,
        prompt="test prompt",
        response="test response",
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
    )


def test_append_and_read(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    out = _make_output("id1", "modelA")
    append_jsonl(p, out)
    records = list(iter_jsonl(p, InferenceOutput))
    assert len(records) == 1
    assert records[0].instance_id == "id1"


def test_dedup_index(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    append_jsonl(p, _make_output("id1", "modelA"))
    append_jsonl(p, _make_output("id2", "modelA"))
    append_jsonl(p, _make_output("id1", "modelB"))

    idx = build_dedup_index(p, "instance_id", "model_id")
    assert ("id1", "modelA") in idx
    assert ("id2", "modelA") in idx
    assert ("id1", "modelB") in idx
    assert ("id3", "modelA") not in idx


def test_dedup_skips_completed():
    """Simulate the skip logic used in inference_runner."""
    completed = {("id1", "modelA"), ("id2", "modelA")}
    pending_ids = ["id1", "id2", "id3"]
    model = "modelA"
    pending = [i for i in pending_ids if (i, model) not in completed]
    assert pending == ["id3"]


def test_empty_file_dedup(tmp_path: Path):
    p = tmp_path / "nonexistent.jsonl"
    idx = build_dedup_index(p, "instance_id", "model_id")
    assert idx == set()


def test_append_multiple_reads_all(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    for i in range(5):
        append_jsonl(p, _make_output(f"id{i}", "modelA"))
    records = list(iter_jsonl(p, InferenceOutput))
    assert len(records) == 5
