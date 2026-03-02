"""Tests for case-insensitive column picking with aliases."""

from __future__ import annotations

from adele_runner.datasets.adele import _GT_COLS, _ID_COLS, _PROMPT_COLS, _pick_col

# ---------------------------------------------------------------------------
# Exact matches (case-insensitive)
# ---------------------------------------------------------------------------


def test_exact_match_lower():
    assert _pick_col(["prompt", "other"], _PROMPT_COLS) == "prompt"


def test_exact_match_upper():
    assert _pick_col(["PROMPT", "other"], _PROMPT_COLS) == "PROMPT"


def test_exact_match_mixed_case():
    assert _pick_col(["Prompt", "other"], _PROMPT_COLS) == "Prompt"


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def test_alias_question():
    assert _pick_col(["question", "extra"], _PROMPT_COLS) == "question"


def test_alias_input():
    assert _pick_col(["Input", "extra"], _PROMPT_COLS) == "Input"


def test_alias_task():
    assert _pick_col(["TASK", "extra"], _PROMPT_COLS) == "TASK"


# ---------------------------------------------------------------------------
# Ground truth aliases
# ---------------------------------------------------------------------------


def test_gt_groundtruth():
    assert _pick_col(["groundtruth", "x"], _GT_COLS) == "groundtruth"


def test_gt_ground_truth():
    assert _pick_col(["ground_truth", "x"], _GT_COLS) == "ground_truth"


def test_gt_answer():
    assert _pick_col(["Answer", "x"], _GT_COLS) == "Answer"


def test_gt_expected():
    assert _pick_col(["EXPECTED", "x"], _GT_COLS) == "EXPECTED"


def test_gt_reference():
    assert _pick_col(["Reference", "x"], _GT_COLS) == "Reference"


# ---------------------------------------------------------------------------
# ID aliases
# ---------------------------------------------------------------------------


def test_id_instance_id():
    assert _pick_col(["instance_id", "x"], _ID_COLS) == "instance_id"


def test_id_id():
    assert _pick_col(["ID", "x"], _ID_COLS) == "ID"


def test_id_item_id():
    assert _pick_col(["item_id", "x"], _ID_COLS) == "item_id"


def test_id_question_id():
    assert _pick_col(["Question_ID", "x"], _ID_COLS) == "Question_ID"


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------


def test_priority_prompt_over_question():
    """'prompt' should be preferred over 'question' since it comes first."""
    assert _pick_col(["question", "prompt"], _PROMPT_COLS) == "prompt"


def test_priority_groundtruth_over_answer():
    assert _pick_col(["answer", "groundtruth"], _GT_COLS) == "groundtruth"


# ---------------------------------------------------------------------------
# No match
# ---------------------------------------------------------------------------


def test_no_match_returns_none():
    assert _pick_col(["foo", "bar"], _PROMPT_COLS) is None


def test_empty_columns():
    assert _pick_col([], _PROMPT_COLS) is None
