from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    run_id: str
    output_dir: str = "runs"


class DatasetConfig(BaseModel):
    name: str = "adele"
    hf_id: str = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
    split: str = "train"
    limit: int | None = None
    seed: int = 42


class FoundryConfig(BaseModel):
    endpoint: str
    api_key_env: str | None = None
    model: str
    api_version: str = "2024-05-01-preview"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0


class BatchConfig(BaseModel):
    enabled: bool = False
    azure_openai_endpoint: str | None = None
    api_key_env: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    completion_endpoint: str = "/chat/completions"
    completion_window: str = "24h"


class InferenceConfig(BaseModel):
    mode: Literal["auto", "foundry_async", "azure_openai_batch"] = "auto"
    foundry: FoundryConfig
    batch: BatchConfig = Field(default_factory=BatchConfig)


class ConcurrencyConfig(BaseModel):
    max_in_flight: int = 16
    request_timeout_s: int = 120
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0


class JudgeModelConfig(BaseModel):
    name: str
    provider: Literal["foundry"] = "foundry"
    model: str


class JudgingConfig(BaseModel):
    enabled: bool = True
    prompt_template: str = "v1"
    require_json: bool = True
    score_threshold: int = 4
    judges: list[JudgeModelConfig] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    inference: InferenceConfig
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    judging: JudgingConfig = Field(default_factory=JudgingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
