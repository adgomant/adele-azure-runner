from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class NormalizedExample(BaseModel):
    instance_id: str
    prompt: str
    ground_truth: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceOutput(BaseModel):
    run_id: str
    instance_id: str
    model_id: str
    prompt: str
    response_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    raw_response: dict[str, Any] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    verdict: Literal["correct", "incorrect", "partial", "unknown"]
    reason: str


class JudgeOutput(BaseModel):
    run_id: str
    instance_id: str
    model_id: str
    judge_name: str
    judge_model_id: str
    prompt_text: str
    raw_output_text: str
    parsed: JudgeResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    dataset_name: str
    dataset_split: str
    dataset_revision: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    code_version: str | None = None
    config_snapshot: dict[str, Any]


class JudgeSummary(BaseModel):
    judge_name: str
    average_score: float
    pass_rate: float


class SummaryReport(BaseModel):
    total_examples: int
    judge_summaries: list[JudgeSummary]
    pairwise_agreement: dict[str, float]


class JudgeParseCandidate(BaseModel):
    score: int
    verdict: str
    reason: str

    @field_validator("verdict")
    @classmethod
    def normalize_verdict(cls, value: str) -> str:
        return value.strip().lower()
