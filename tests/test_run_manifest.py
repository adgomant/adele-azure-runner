"""Tests for RunManifest creation and update."""

from __future__ import annotations

from datetime import datetime

from adele_runner.schemas import RunManifest


def test_manifest_creation():
    m = RunManifest(
        run_id="test-run",
        dataset_name="adele",
        model_id="gpt-4o",
        total_instances=100,
    )
    assert m.run_id == "test-run"
    assert m.dataset_name == "adele"
    assert m.model_id == "gpt-4o"
    assert m.total_instances == 100
    assert m.completed_instances == 0
    assert m.end_time is None
    assert isinstance(m.start_time, datetime)


def test_manifest_update():
    m = RunManifest(
        run_id="test-run",
        dataset_name="adele",
        model_id="gpt-4o",
        total_instances=100,
    )
    m.end_time = datetime(2024, 1, 1, 12, 0, 0)
    m.completed_instances = 95
    assert m.completed_instances == 95
    assert m.end_time == datetime(2024, 1, 1, 12, 0, 0)


def test_manifest_serialization():
    m = RunManifest(
        run_id="test-run",
        dataset_name="adele",
        model_id="gpt-4o",
        total_instances=50,
    )
    data = m.model_dump(mode="json")
    assert data["run_id"] == "test-run"
    assert data["total_instances"] == 50
    assert data["completed_instances"] == 0


def test_manifest_defaults():
    m = RunManifest(
        run_id="r",
        dataset_name="d",
        model_id="m",
    )
    assert m.code_version == "0.1.0"
    assert m.dataset_revision is None
    assert m.params == {}
    assert m.total_instances == 0
