"""Internal data schemas for ADeLe Runner."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InferenceOutput(BaseModel):
    """Single model inference result."""

    instance_id: str
    model_id: str
    prompt: str
    response: str | None
    status: Literal["success", "failed"] = "success"
    error_message: str | None = None
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: str = ""


class JudgeOutput(BaseModel):
    """Single judge evaluation result."""

    instance_id: str
    model_id: str
    judge_name: str
    score: int | None = Field(default=None, ge=1, le=5)
    verdict: Literal["correct", "incorrect", "partial", "unknown"] | None
    reason: str | None
    raw_output: str | None
    judge_prompt: str
    status: Literal["success", "parse_failed", "request_failed", "skipped"] = "success"
    error_message: str | None = None
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: str = ""


class DatasetItem(BaseModel):
    """Normalised dataset row."""

    instance_id: str
    prompt: str
    ground_truth: str
    metadata: dict = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Captures provenance for a complete run."""

    run_id: str
    dataset_name: str
    dataset_revision: str | None = None
    model_id: str
    params: dict = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    code_version: str = "0.1.0"
    total_instances: int = 0
    completed_instances: int = 0
