"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    run_id: str = "adele_run"
    output_dir: str = "runs"


class DatasetConfig(BaseModel):
    name: str = "adele"
    hf_id: str = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
    split: str = "train"
    limit: int | None = None


class FoundryInferenceConfig(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_AI_API_KEY"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0


class BatchInferenceConfig(BaseModel):
    enabled: bool = False
    azure_openai_endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    deployment: str = ""
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"
    completion_window: str = "24h"


class InferenceConfig(BaseModel):
    mode: Literal["auto", "foundry_async", "azure_openai_batch"] = "auto"
    foundry: FoundryInferenceConfig = Field(default_factory=FoundryInferenceConfig)
    batch: BatchInferenceConfig = Field(default_factory=BatchInferenceConfig)


class ConcurrencyConfig(BaseModel):
    max_in_flight: int = 16
    request_timeout_s: float = 120.0
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    max_poll_time_s: float = 3600.0


class JudgeConfig(BaseModel):
    name: str
    provider: Literal["foundry", "batch"] = "foundry"
    model: str


class JudgingConfig(BaseModel):
    enabled: bool = True
    prompt_template: str = "v1"
    require_json: bool = True
    judges: list[JudgeConfig] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ModelPricing(BaseModel):
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


class PricingConfig(BaseModel):
    enabled: bool = False
    models: dict[str, ModelPricing] = Field(default_factory=dict)


_PLACEHOLDER_RE = re.compile(r"<[A-Z_-]+>")


class AppConfig(BaseModel):
    """Root config object."""

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
        key = os.environ.get(self.inference.foundry.api_key_env, "")
        if not key:
            raise ValueError(
                f"Environment variable '{self.inference.foundry.api_key_env}' is not set."
            )
        return key

    def get_batch_api_key(self) -> str:
        key = os.environ.get(self.inference.batch.api_key_env, "")
        if not key:
            raise ValueError(
                f"Environment variable '{self.inference.batch.api_key_env}' is not set."
            )
        return key

    def resolve_inference_mode(self) -> Literal["foundry_async", "azure_openai_batch"]:
        """Resolve the effective inference mode from config."""
        if self.inference.mode == "azure_openai_batch":
            return "azure_openai_batch"
        if self.inference.mode == "auto" and self.inference.batch.enabled:
            return "azure_openai_batch"
        return "foundry_async"

    def validate_config(self, *, dry_run: bool = False) -> list[str]:
        """Validate config for common mistakes. Returns list of error messages."""
        errors: list[str] = []
        mode = self.resolve_inference_mode()

        if mode == "foundry_async":
            ep = self.inference.foundry.endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    f"Foundry endpoint is not configured: '{ep}'. "
                    "Set inference.foundry.endpoint in config.yaml."
                )
            if not self.inference.foundry.model:
                errors.append("Foundry model is not set. Use --model or set inference.foundry.model.")
        else:
            ep = self.inference.batch.azure_openai_endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    f"Batch endpoint is not configured: '{ep}'. "
                    "Set inference.batch.azure_openai_endpoint in config.yaml."
                )
            if not self.inference.batch.deployment:
                errors.append(
                    "Batch deployment is not set. Use --model or set inference.batch.deployment."
                )

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
