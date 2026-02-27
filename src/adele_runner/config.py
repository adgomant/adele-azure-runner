"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RateLimitsConfig(BaseModel):
    """Rate limits for an Azure Foundry deployment (from Azure portal)."""

    tokens_per_minute: int
    requests_per_minute: int


class AzureFoundryConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_AI_API_KEY"


class AzureBatchConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"
    max_requests_per_file: int = 50_000
    max_bytes_per_file: int = 100_000_000  # 100 MB


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
    rate_limits: RateLimitsConfig | None = None


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
    rate_limits: RateLimitsConfig | None = None
    max_tokens: int = 512


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


def compute_concurrency_from_rate_limits(
    rate_limits: RateLimitsConfig,
    max_tokens: int,
) -> ConcurrencyConfig:
    """Compute optimal concurrency parameters from Azure rate limits.

    Uses TPM, RPM, and ``max_tokens`` to derive safe values for
    ``max_in_flight``, ``request_timeout_s``, and backoff parameters.
    """
    tpm = rate_limits.tokens_per_minute
    rpm = rate_limits.requests_per_minute

    # Worst-case tokens per request (prompt + completion ≈ max_tokens)
    est_tokens_per_request = max(1, max_tokens)

    # Max RPM the token budget allows
    rpm_from_tpm = tpm / est_tokens_per_request

    # Effective RPM = tighter constraint
    effective_rpm = min(rpm, rpm_from_tpm)

    # Estimate avg request duration from max_tokens (~80 tok/s, capped 2–300s)
    avg_duration_s = max(2.0, min(300.0, max_tokens / 80))

    # max_in_flight ≈ effective_rpm * avg_duration / 60, 80% safety margin
    max_in_flight = max(1, int(effective_rpm * avg_duration_s / 60 * 0.8))

    # request_timeout_s: generation time + 10s overhead, capped 30–600s
    request_timeout_s = round(max(30.0, min(600.0, max_tokens / 50 + 10)), 1)

    # backoff: base scales with rate limit density
    backoff_base_s = round(max(0.5, min(10.0, 60.0 / max(1, effective_rpm))), 1)
    backoff_max_s = round(max(30.0, min(120.0, backoff_base_s * 10)), 1)

    return ConcurrencyConfig(
        max_in_flight=max_in_flight,
        request_timeout_s=request_timeout_s,
        backoff_base_s=backoff_base_s,
        backoff_max_s=backoff_max_s,
    )


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

    def apply_rate_limit_overrides(
        self,
        rate_limits: RateLimitsConfig | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Override concurrency settings with values computed from rate limits.

        When *rate_limits* is ``None`` the call is a no-op.
        """
        if rate_limits is None:
            return

        mt = max_tokens or self.inference.max_tokens
        computed = compute_concurrency_from_rate_limits(rate_limits, mt)

        # Preserve non-rate-limit params from user config
        computed.max_retries = self.concurrency.max_retries
        computed.max_poll_time_s = self.concurrency.max_poll_time_s
        computed.batch_completion_window = self.concurrency.batch_completion_window

        self.concurrency = computed

        logger.info(
            "Rate-limit auto-tuning: TPM=%d RPM=%d max_tokens=%d → "
            "max_in_flight=%d request_timeout=%.1fs backoff=%.1f–%.1fs",
            rate_limits.tokens_per_minute,
            rate_limits.requests_per_minute,
            mt,
            computed.max_in_flight,
            computed.request_timeout_s,
            computed.backoff_base_s,
            computed.backoff_max_s,
        )

    def get_most_restrictive_judge_rate_limits(self) -> RateLimitsConfig | None:
        """Return the most restrictive rate limits among foundry judges, or ``None``."""
        limits = [
            j.rate_limits
            for j in self.judging.judges
            if j.provider == "foundry" and j.rate_limits is not None
        ]
        if not limits:
            return None
        return min(limits, key=lambda rl: rl.tokens_per_minute)

    def get_max_judge_max_tokens(self) -> int:
        """Return the largest ``max_tokens`` among foundry judges (for auto-tuning)."""
        values = [j.max_tokens for j in self.judging.judges if j.provider == "foundry"]
        return max(values) if values else 512

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

        # Batch splitting limits must not exceed Azure hard maximums
        from adele_runner.utils.batch_split import (
            AZURE_BATCH_MAX_FILE_BYTES,
            AZURE_BATCH_MAX_REQUESTS,
        )

        batch = self.azure.batch
        if batch.max_requests_per_file < 1:
            errors.append("azure.batch.max_requests_per_file must be at least 1.")
        elif batch.max_requests_per_file > AZURE_BATCH_MAX_REQUESTS:
            errors.append(
                f"azure.batch.max_requests_per_file ({batch.max_requests_per_file}) "
                f"exceeds Azure maximum ({AZURE_BATCH_MAX_REQUESTS})."
            )
        if batch.max_bytes_per_file < 1:
            errors.append("azure.batch.max_bytes_per_file must be at least 1.")
        elif batch.max_bytes_per_file > AZURE_BATCH_MAX_FILE_BYTES:
            errors.append(
                f"azure.batch.max_bytes_per_file ({batch.max_bytes_per_file}) "
                f"exceeds Azure maximum ({AZURE_BATCH_MAX_FILE_BYTES})."
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
