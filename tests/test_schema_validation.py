"""Tests for internal schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adele_runner.schemas import DatasetItem, InferenceOutput, JudgeOutput


def test_inference_output_valid():
    out = InferenceOutput(
        instance_id="abc",
        model_id="gpt-4o",
        prompt="Hello",
        response="World",
    )
    assert out.instance_id == "abc"


def test_inference_output_json_roundtrip():
    out = InferenceOutput(
        instance_id="abc",
        model_id="gpt-4o",
        prompt="Hello",
        response="World",
    )
    restored = InferenceOutput.model_validate_json(out.model_dump_json())
    assert restored.instance_id == out.instance_id
    assert restored.response == out.response


def test_judge_output_valid():
    j = JudgeOutput(
        instance_id="abc",
        model_id="gpt-4o",
        judge_name="judge1",
        score=4,
        verdict="correct",
        reason="Good.",
        raw_output='{"score":4}',
        judge_prompt="Evaluate...",
    )
    assert j.verdict == "correct"


def test_failed_inference_output_valid():
    out = InferenceOutput(
        instance_id="abc",
        model_id="gpt-4o",
        prompt="Hello",
        response=None,
        status="failed",
        error_message="timeout",
    )
    assert out.status == "failed"


def test_judge_output_none_score_valid():
    j = JudgeOutput(
        instance_id="abc",
        model_id="gpt-4o",
        judge_name="judge1",
        score=None,
        verdict=None,
        reason=None,
        raw_output=None,
        judge_prompt="Evaluate...",
        status="request_failed",
        error_message="timeout",
    )
    assert j.score is None


def test_judge_output_invalid_score():
    with pytest.raises(ValidationError):
        JudgeOutput(
            instance_id="abc",
            model_id="gpt-4o",
            judge_name="judge1",
            score=6,  # invalid: must be 1-5
            verdict="correct",
            reason="Good.",
            raw_output="",
            judge_prompt="",
        )


def test_dataset_item_valid():
    item = DatasetItem(instance_id="x", prompt="Q?", ground_truth="A.")
    assert item.metadata == {}
