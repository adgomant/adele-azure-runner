"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AzureFoundryConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_AI_API_KEY"


class AzureBatchConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"


class AzureConfig(BaseModel):
    foundry: AzureFoundryConnection = Field(default_factory=AzureFoundryConnection)
    batch: AzureBatchConnection = Field(default_factory=AzureBatchConnection)


class RunConfig(BaseModel):
    run_id: str = "adele_run"
    output_dir: str = "runs"


class DatasetConfig(BaseModel):
    name: str = "adele"
    hf_id: str = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
    split: str = "train"
    limit: int | None = None


class InferenceConfig(BaseModel):
    mode: Literal["auto", "foundry", "batch"] = "auto"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0


class ConcurrencyConfig(BaseModel):
    max_in_flight: int = 16
    request_timeout_s: float = 120.0
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    max_poll_time_s: float = 3600.0
    batch_completion_window: str = "24h"


class JudgeConfig(BaseModel):
    name: str
    provider: Literal["foundry", "batch"] = "foundry"
    model: str


class JudgingConfig(BaseModel):
    enabled: bool = True
    prompt_template: str = "v1"
    judges: list[JudgeConfig] = Field(default_factory=list)

    @field_validator("judges", mode="before")
    @classmethod
    def _coerce_none(cls, v: Any) -> Any:
        return v if v is not None else []


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ModelPricing(BaseModel):
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


class PricingConfig(BaseModel):
    enabled: bool = False
    models: dict[str, ModelPricing] = Field(default_factory=dict)

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_none(cls, v: Any) -> Any:
        return v if v is not None else {}


_PLACEHOLDER_RE = re.compile(r"<[A-Z_-]+>")


class AppConfig(BaseModel):
    """Root config object."""

    azure: AzureConfig = Field(default_factory=AzureConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    judging: JudgingConfig = Field(default_factory=JudgingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @model_validator(mode="after")
    def resolve_api_keys(self) -> AppConfig:
        """Replace api_key_env placeholders with actual env var values (validation only)."""
        return self

    def get_foundry_api_key(self) -> str:
        key = os.environ.get(self.azure.foundry.api_key_env, "")
        if not key:
            raise ValueError(f"Environment variable '{self.azure.foundry.api_key_env}' is not set.")
        return key

    def get_batch_api_key(self) -> str:
        key = os.environ.get(self.azure.batch.api_key_env, "")
        if not key:
            raise ValueError(f"Environment variable '{self.azure.batch.api_key_env}' is not set.")
        return key

    def resolve_inference_mode(self) -> Literal["foundry", "batch"]:
        """Resolve the effective inference mode from config."""
        if self.inference.mode == "batch":
            return "batch"
        return "foundry"

    def validate_config(self, *, dry_run: bool = False) -> list[str]:
        """Validate config for common mistakes. Returns list of error messages."""
        errors: list[str] = []
        mode = self.resolve_inference_mode()

        if mode == "foundry":
            ep = self.azure.foundry.endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    f"Foundry endpoint is not configured: '{ep}'. "
                    "Set azure.foundry.endpoint in config.yaml."
                )
            if not self.inference.model:
                errors.append("Model is not set. Use --model or set inference.model.")
        else:
            ep = self.azure.batch.endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    f"Batch endpoint is not configured: '{ep}'. "
                    "Set azure.batch.endpoint in config.yaml."
                )
            if not self.inference.model:
                errors.append("Model is not set. Use --model or set inference.model.")

        if self.judging.enabled and not self.judging.judges:
            errors.append(
                "Judging is enabled but no judges configured. Use --judge or set judging.judges."
            )

        if self.judging.prompt_template not in ("v1", "v2"):
            errors.append(
                f"Unknown judge prompt template: '{self.judging.prompt_template}'. Use 'v1' or 'v2'."
            )

        return errors

    def run_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.run_id

    def outputs_path(self) -> Path:
        return self.run_dir() / "outputs.jsonl"

    def judge_outputs_path(self) -> Path:
        return self.run_dir() / "judge_outputs.jsonl"

    def manifest_path(self) -> Path:
        return self.run_dir() / "run_manifest.json"

    def merged_path(self) -> Path:
        return self.run_dir() / "merged_results.parquet"

    def metrics_path(self) -> Path:
        return self.run_dir() / "metrics.json"

    def ground_truths_cache_path(self) -> Path:
        return self.run_dir() / "ground_truths.json"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from a YAML file, with environment variable overrides."""
    raw: dict[str, Any] = {}

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)
