"""Tests for judge JSON parsing."""

from __future__ import annotations

from adele_runner.pipeline.judge_runner import parse_judge_json


def test_parse_clean_json():
    raw = '{"score": 4, "verdict": "correct", "reason": "Looks good."}'
    result = parse_judge_json(raw)
    assert result["score"] == 4
    assert result["verdict"] == "correct"


def test_parse_json_with_preamble():
    raw = 'Sure! Here is my evaluation:\n{"score": 2, "verdict": "incorrect", "reason": "Wrong."}'
    result = parse_judge_json(raw)
    assert result["score"] == 2
    assert result["verdict"] == "incorrect"


def test_parse_regex_fallback():
    raw = 'score: 3, verdict: "partial", reason: "Partially correct"'
    result = parse_judge_json(raw)
    assert 1 <= result["score"] <= 5
    assert result["verdict"] in {"correct", "incorrect", "partial", "unknown"}


def test_parse_invalid_verdict_coerced():
    raw = '{"score": 5, "verdict": "yes", "reason": "Fine."}'
    result = parse_judge_json(raw)
    assert result["verdict"] == "unknown"


def test_parse_score_clamped():
    raw = '{"score": 5, "verdict": "correct", "reason": "Perfect."}'
    result = parse_judge_json(raw)
    assert result["score"] == 5
