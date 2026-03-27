"""Tests for average-based verification score logic."""

from __future__ import annotations

import pytest

from adele_runner.pipeline.metrics import compute_verification_scores
from adele_runner.schemas import InferenceOutput, JudgeOutput


def _judge(instance_id: str, model_id: str, judge_name: str, score: int) -> JudgeOutput:
    """Helper to build a minimal JudgeOutput."""
    return JudgeOutput(
        instance_id=instance_id,
        model_id=model_id,
        judge_name=judge_name,
        score=score,
        verdict="correct",
        reason="test",
        raw_output="{}",
        judge_prompt="",
    )


def _inference(instance_id: str, model_id: str, *, status: str = "success") -> InferenceOutput:
    return InferenceOutput(
        instance_id=instance_id,
        model_id=model_id,
        prompt="Q",
        response="A" if status == "success" else None,
        status=status,
    )


# ---------------------------------------------------------------------------
# compute_verification_scores
# ---------------------------------------------------------------------------


def test_single_judge_score_4():
    records = [_judge("i1", "m1", "j1", 4)]
    scores = compute_verification_scores([_inference("i1", "m1")], records, ["j1"])
    assert scores[("i1", "m1")] == 4.0


def test_single_judge_score_3():
    records = [_judge("i1", "m1", "j1", 3)]
    scores = compute_verification_scores([_inference("i1", "m1")], records, ["j1"])
    assert scores[("i1", "m1")] == 3.0


def test_two_judges_avg_3():
    """Scores [4, 2] → avg 3.0, which should pass (avg >= 3)."""
    records = [
        _judge("i1", "m1", "j1", 4),
        _judge("i1", "m1", "j2", 2),
    ]
    scores = compute_verification_scores([_inference("i1", "m1")], records, ["j1", "j2"])
    assert scores[("i1", "m1")] == 3.0


def test_two_judges_avg_above_3():
    """Scores [4, 3] → avg 3.5, which should pass."""
    records = [
        _judge("i1", "m1", "j1", 4),
        _judge("i1", "m1", "j2", 3),
    ]
    scores = compute_verification_scores([_inference("i1", "m1")], records, ["j1", "j2"])
    assert scores[("i1", "m1")] == 3.5


def test_no_records():
    scores = compute_verification_scores([], [], ["j1"])
    assert scores == {}


def test_missing_judge_yields_none():
    scores = compute_verification_scores([_inference("i1", "m1")], [_judge("i1", "m1", "j1", 4)], ["j1", "j2"])
    assert scores[("i1", "m1")] is None


def test_failed_inference_yields_none():
    scores = compute_verification_scores([_inference("i1", "m1", status="failed")], [], ["j1"])
    assert scores[("i1", "m1")] is None


# ---------------------------------------------------------------------------
# Verification binary: 1 if avg >= 3 else 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "judge_scores, expected_verification",
    [
        ([4], 1),  # avg=4 >= 3 → pass
        ([3], 1),  # avg=3 >= 3 → pass
        ([4, 2], 1),  # avg=3.0 → pass
        ([4, 3], 1),  # avg=3.5 → pass
        ([5, 5], 1),  # avg=5 → pass
        ([1, 1], 0),  # avg=1 → fail
        ([5, 1], 1),  # avg=3 → pass
        ([4, 4, 1], 1),  # avg=3.0 → pass
    ],
)
def test_verification_binary(judge_scores: list[int], expected_verification: int):
    """End-to-end: judge scores → avg → binary verification."""
    records = [_judge("i1", "m1", f"j{i}", s) for i, s in enumerate(judge_scores)]
    avg_scores = compute_verification_scores([_inference("i1", "m1")], records, [f"j{i}" for i, _ in enumerate(judge_scores)])
    avg = avg_scores[("i1", "m1")]

    verification = 1 if avg >= 3 else 0
    assert verification == expected_verification


def test_verification_binary_441():
    """[4, 4, 1] → avg=3.0 → pass (1)."""
    records = [
        _judge("i1", "m1", "j1", 4),
        _judge("i1", "m1", "j2", 4),
        _judge("i1", "m1", "j3", 1),
    ]
    avg_scores = compute_verification_scores([_inference("i1", "m1")], records, ["j1", "j2", "j3"])
    avg = avg_scores[("i1", "m1")]
    assert avg == 3.0
    assert (1 if avg >= 3 else 0) == 1


def test_multiple_instances():
    """Two instances: one passes, one fails."""
    records = [
        _judge("i1", "m1", "j1", 5),  # avg=5 → pass
        _judge("i2", "m1", "j1", 2),  # avg=2 → fail
    ]
    avg_scores = compute_verification_scores([_inference("i1", "m1"), _inference("i2", "m1")], records, ["j1"])
    assert avg_scores[("i1", "m1")] == 5.0
    assert avg_scores[("i2", "m1")] == 2.0

    verification = {k: (1 if v >= 3 else 0) for k, v in avg_scores.items()}
    assert verification[("i1", "m1")] == 1
    assert verification[("i2", "m1")] == 0
