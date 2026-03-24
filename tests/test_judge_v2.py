"""Tests for v2 bare-integer judge output parsing."""

from __future__ import annotations

from adele_runner.stages.judging import parse_judge_v2


def test_clean_integer():
    result = parse_judge_v2("4")
    assert result["score"] == 4
    assert result["verdict"] == "unknown"


def test_integer_with_whitespace():
    result = parse_judge_v2("  3  \n")
    assert result["score"] == 3


def test_integer_with_preamble():
    result = parse_judge_v2("Based on my analysis, I would rate this 4 out of 5")
    assert result["score"] == 4


def test_score_1():
    result = parse_judge_v2("1")
    assert result["score"] == 1


def test_score_5():
    result = parse_judge_v2("5")
    assert result["score"] == 5


def test_out_of_range_high_clamped():
    """Score > 5 should be clamped to 5."""
    result = parse_judge_v2("8")
    assert result["score"] == 5


def test_out_of_range_low_clamped():
    """Score < 1 should be clamped to 1."""
    result = parse_judge_v2("0")
    assert result["score"] == 1


def test_no_integer_defaults_to_1():
    result = parse_judge_v2("I cannot evaluate this.")
    assert result["score"] == 1


def test_regex_finds_first_valid_score():
    """When text contains multiple digits, regex finds first 1-5 match."""
    result = parse_judge_v2("The correct answer gets a score of 3 out of 5")
    assert result["score"] == 3


def test_only_non_score_digits():
    """When text has digits but none in 1-5 range, and not directly parseable."""
    result = parse_judge_v2("I looked at 07 items and found 0 matches")
    assert result["score"] == 1  # default


def test_v2_result_structure():
    result = parse_judge_v2("5")
    assert set(result.keys()) == {"score", "verdict", "reason"}
    assert result["reason"] == "Scored via v2 bare-integer prompt"
