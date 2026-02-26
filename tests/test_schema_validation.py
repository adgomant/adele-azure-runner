import pytest
from pydantic import ValidationError

from adele_runner.models import JudgeResult


def test_judge_result_schema_accepts_valid() -> None:
    result = JudgeResult(score=3, verdict="partial", reason="partially correct")
    assert result.score == 3


def test_judge_result_schema_rejects_invalid_score() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(score=8, verdict="correct", reason="invalid")
