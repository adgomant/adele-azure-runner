"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ProviderKind = Literal["azure_openai", "azure_ai_inference", "google_genai", "anthropic"]
ExecutionMode = Literal["request_response", "batch", "auto"]
GoogleBackend = Literal["gemini_api", "vertex_ai"]
TargetSupportedMode = Literal["request_response", "batch"]

_PLACEHOLDER_RE = re.compile(r"<[A-Z_-]+>")
_AZURE_BATCH_DEPLOYMENT_TYPES = {"global_batch", "data_zone_batch"}


class RateLimitsConfig(BaseModel):
    """Rate limits for a provider target."""

    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    concurrent_requests: int | None = None
    batch_requests_per_minute: int | None = None
    batch_queue_requests: int | None = None
    batch_enqueued_tokens: int | None = None

    @field_validator("*")
    @classmethod
    def _validate_positive_limits(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("rate limit values must be >= 1")
        return value


class AzureOpenAIConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"
    max_requests_per_file: int = 100_000
    max_bytes_per_file: int = 200_000_000


class AzureAIInferenceConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_AI_API_KEY"


class GoogleGenAIConnection(BaseModel):
    api_key_env: str = "GEMINI_API_KEY"
    backend: GoogleBackend = "gemini_api"
    project: str = ""
    location: str = ""


class AnthropicConnection(BaseModel):
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = ""


class ProvidersConfig(BaseModel):
    azure_openai: AzureOpenAIConnection = Field(default_factory=AzureOpenAIConnection)
    azure_ai_inference: AzureAIInferenceConnection = Field(default_factory=AzureAIInferenceConnection)
    google_genai: GoogleGenAIConnection = Field(default_factory=GoogleGenAIConnection)
    anthropic: AnthropicConnection = Field(default_factory=AnthropicConnection)


class RunConfig(BaseModel):
    run_id: str = "adele_run"
    output_dir: str = "runs"


class DatasetConfig(BaseModel):
    name: str = "adele"
    hf_id: str = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
    split: str = "train"
    limit: int | None = None


class TargetConfig(BaseModel):
    """Optional deployment/model capability hints."""

    supported_modes: list[TargetSupportedMode] | None = None
    batch_capable: bool | None = None
    deployment_type: str | None = None


class InferenceConfig(BaseModel):
    provider: ProviderKind = "azure_ai_inference"
    mode: ExecutionMode = "auto"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    rate_limits: RateLimitsConfig | None = None
    budget_usd: float | None = None


class ConcurrencyConfig(BaseModel):
    max_in_flight: int = 16
    request_timeout_s: float = 120.0
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    max_poll_time_s: float = 3600.0
    batch_completion_window: str = "24h"
    effective_rpm: int | None = None


class JudgeConfig(BaseModel):
    name: str
    provider: ProviderKind = "azure_ai_inference"
    mode: ExecutionMode = "request_response"
    model: str
    rate_limits: RateLimitsConfig | None = None
    max_tokens: int = 512
    budget_usd: float | None = None


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

    def get_model_pricing(self, key: str) -> ModelPricing | None:
        return self.models.get(key)


def compute_concurrency_from_rate_limits(
    rate_limits: RateLimitsConfig,
    max_tokens: int,
) -> ConcurrencyConfig:
    """Compute optimal concurrency parameters from rate limits."""
    est_tokens_per_request = max(1, max_tokens)
    est_input_tokens = est_tokens_per_request
    est_output_tokens = est_tokens_per_request

    rpm_candidates: list[float] = []
    if rate_limits.requests_per_minute is not None:
        rpm_candidates.append(float(rate_limits.requests_per_minute))
    if rate_limits.tokens_per_minute is not None:
        rpm_candidates.append(rate_limits.tokens_per_minute / est_tokens_per_request)
    if rate_limits.input_tokens_per_minute is not None:
        rpm_candidates.append(rate_limits.input_tokens_per_minute / est_input_tokens)
    if rate_limits.output_tokens_per_minute is not None:
        rpm_candidates.append(rate_limits.output_tokens_per_minute / est_output_tokens)
    if rate_limits.requests_per_day is not None:
        rpm_candidates.append(rate_limits.requests_per_day / 1440.0)
    if rate_limits.tokens_per_day is not None:
        rpm_candidates.append(rate_limits.tokens_per_day / (est_tokens_per_request * 1440.0))

    effective_rpm = min(rpm_candidates) if rpm_candidates else None
    avg_duration_s = max(2.0, min(300.0, max_tokens / 200))
    if effective_rpm is None:
        max_in_flight = rate_limits.concurrent_requests or 1
    else:
        max_in_flight = max(1, int(effective_rpm * avg_duration_s / 60 * 0.8))
    if rate_limits.concurrent_requests is not None:
        max_in_flight = min(max_in_flight, rate_limits.concurrent_requests)
    request_timeout_s = round(max(30.0, min(600.0, max_tokens / 50 + 10)), 1)
    derived_rpm = max(1.0, effective_rpm or 1.0)
    backoff_base_s = round(max(0.5, min(10.0, 60.0 / derived_rpm)), 1)
    backoff_max_s = round(max(30.0, min(120.0, backoff_base_s * 10)), 1)

    return ConcurrencyConfig(
        max_in_flight=max_in_flight,
        request_timeout_s=request_timeout_s,
        backoff_base_s=backoff_base_s,
        backoff_max_s=backoff_max_s,
        effective_rpm=max(1, int(effective_rpm)) if effective_rpm is not None else None,
    )


class AppConfig(BaseModel):
    """Root config object."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    targets: dict[str, TargetConfig] = Field(default_factory=dict)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    judging: JudgingConfig = Field(default_factory=JudgingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @field_validator("targets", mode="before")
    @classmethod
    def _coerce_targets_none(cls, v: Any) -> Any:
        return v if v is not None else {}

    def get_provider_api_key(self, provider: ProviderKind) -> str:
        env_name = {
            "azure_openai": self.providers.azure_openai.api_key_env,
            "azure_ai_inference": self.providers.azure_ai_inference.api_key_env,
            "google_genai": self.providers.google_genai.api_key_env,
            "anthropic": self.providers.anthropic.api_key_env,
        }[provider]
        key = os.environ.get(env_name, "")
        if not key:
            raise ValueError(f"Environment variable '{env_name}' is not set.")
        return key

    def get_target_config(self, model: str) -> TargetConfig | None:
        return self.targets.get(model)

    def get_most_restrictive_judge_rate_limits(self) -> RateLimitsConfig | None:
        limits = [
            judge.rate_limits
            for judge in self.judging.judges
            if judge.mode == "request_response" and judge.rate_limits is not None
        ]
        if not limits:
            return None
        return min(
            limits,
            key=lambda rate_limits: rate_limits.tokens_per_minute
            or rate_limits.input_tokens_per_minute
            or rate_limits.requests_per_minute
            or 0,
        )

    def get_max_judge_max_tokens(self) -> int:
        values = [judge.max_tokens for judge in self.judging.judges if judge.mode == "request_response"]
        return max(values) if values else 512

    def _validate_provider_connection(self, provider: ProviderKind, errors: list[str]) -> None:
        if provider == "azure_ai_inference":
            endpoint = self.providers.azure_ai_inference.endpoint
            if not endpoint or _PLACEHOLDER_RE.search(endpoint):
                errors.append(
                    "Azure AI Inference endpoint is not configured. "
                    "Set providers.azure_ai_inference.endpoint in config.yaml."
                )
            return

        if provider == "azure_openai":
            endpoint = self.providers.azure_openai.endpoint
            if not endpoint or _PLACEHOLDER_RE.search(endpoint):
                errors.append(
                    "Azure OpenAI endpoint is not configured. "
                    "Set providers.azure_openai.endpoint in config.yaml."
                )
            return

        if provider == "google_genai":
            google_cfg = self.providers.google_genai
            if google_cfg.backend == "vertex_ai":
                if not google_cfg.project:
                    errors.append(
                        "providers.google_genai.project is required when backend='vertex_ai'."
                    )
                if not google_cfg.location:
                    errors.append(
                        "providers.google_genai.location is required when backend='vertex_ai'."
                    )

    def _validate_target_mode_support(
        self,
        provider: ProviderKind,
        mode: ExecutionMode,
        model: str,
        errors: list[str],
        *,
        context: str,
    ) -> None:
        if mode == "auto":
            return

        if provider == "azure_ai_inference" and mode == "batch":
            errors.append(f"{context}: provider 'azure_ai_inference' does not support batch mode.")
            return

        if provider == "google_genai":
            if mode == "batch" and self.providers.google_genai.backend not in {"gemini_api", "vertex_ai"}:
                errors.append(
                    f"{context}: provider 'google_genai' does not support batch for backend "
                    f"'{self.providers.google_genai.backend}'."
                )
            return

        if provider == "azure_openai" and mode == "batch":
            target_cfg = self.get_target_config(model)
            if target_cfg is None:
                errors.append(
                    f"{context}: batch mode for provider 'azure_openai' requires targets.{model}.supported_modes, "
                    "targets.<name>.batch_capable=true, or a batch deployment_type hint."
                )
                return
            if target_cfg.supported_modes is not None and "batch" in target_cfg.supported_modes:
                return
            if target_cfg.batch_capable:
                return
            if (
                target_cfg.deployment_type
                and target_cfg.deployment_type.lower() in _AZURE_BATCH_DEPLOYMENT_TYPES
            ):
                return
            errors.append(
                f"{context}: target '{model}' is not marked as batch-capable for provider 'azure_openai'."
            )

    def validate_config(
        self,
        *,
        dry_run: bool = False,
        validate_inference: bool = True,
        validate_judging: bool = True,
    ) -> list[str]:
        """Validate config for common mistakes. Returns list of error messages."""
        _ = dry_run
        errors: list[str] = []

        if validate_inference:
            if not self.inference.model:
                errors.append("Model is not set. Use --model or set inference.model.")
            else:
                self._validate_provider_connection(self.inference.provider, errors)
                self._validate_target_mode_support(
                    self.inference.provider,
                    self.inference.mode,
                    self.inference.model,
                    errors,
                    context="Inference",
                )

        if validate_judging and self.judging.enabled and not self.judging.judges:
            errors.append(
                "Judging is enabled but no judges configured. Use --judge or set judging.judges."
            )

        if validate_judging and self.judging.prompt_template not in ("v1", "v2"):
            errors.append(
                f"Unknown judge prompt template: '{self.judging.prompt_template}'. Use 'v1' or 'v2'."
            )

        if validate_inference:
            self._validate_budget_fields(
                budget_usd=self.inference.budget_usd,
                pricing_key=self.inference.model,
                errors=errors,
                context="Inference",
            )

        if validate_judging:
            for judge in self.judging.judges:
                self._validate_provider_connection(judge.provider, errors)
                self._validate_target_mode_support(
                    judge.provider,
                    judge.mode,
                    judge.model,
                    errors,
                    context=f"Judge '{judge.name}'",
                )
                self._validate_budget_fields(
                    budget_usd=judge.budget_usd,
                    pricing_key=judge.name,
                    errors=errors,
                    context=f"Judge '{judge.name}'",
                )
                self._validate_rate_limit_fields(
                    judge.provider,
                    judge.mode,
                    judge.rate_limits,
                    errors,
                    context=f"Judge '{judge.name}'",
                )

        if validate_inference:
            self._validate_rate_limit_fields(
                self.inference.provider,
                self.inference.mode,
                self.inference.rate_limits,
                errors,
                context="Inference",
            )

        from adele_runner.utils.batch_split import (
            AZURE_BATCH_MAX_FILE_BYTES,
            AZURE_BATCH_MAX_REQUESTS,
        )

        azure_openai = self.providers.azure_openai
        if azure_openai.max_requests_per_file < 1:
            errors.append("providers.azure_openai.max_requests_per_file must be at least 1.")
        elif azure_openai.max_requests_per_file > AZURE_BATCH_MAX_REQUESTS:
            errors.append(
                f"providers.azure_openai.max_requests_per_file ({azure_openai.max_requests_per_file}) "
                f"exceeds Azure maximum ({AZURE_BATCH_MAX_REQUESTS})."
            )
        if azure_openai.max_bytes_per_file < 1:
            errors.append("providers.azure_openai.max_bytes_per_file must be at least 1.")
        elif azure_openai.max_bytes_per_file > AZURE_BATCH_MAX_FILE_BYTES:
            errors.append(
                f"providers.azure_openai.max_bytes_per_file ({azure_openai.max_bytes_per_file}) "
                f"exceeds Azure maximum ({AZURE_BATCH_MAX_FILE_BYTES})."
            )

        return errors

    def _validate_rate_limit_fields(
        self,
        provider: ProviderKind,
        mode: ExecutionMode,
        rate_limits: RateLimitsConfig | None,
        errors: list[str],
        *,
        context: str,
    ) -> None:
        if rate_limits is None:
            return

        if provider == "anthropic" and mode == "batch":
            if (
                rate_limits.batch_requests_per_minute is not None
                and rate_limits.batch_requests_per_minute > 50
            ):
                errors.append(
                    f"{context}: anthropic batch_requests_per_minute exceeds Anthropic limit (50)."
                )

    def run_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.run_id

    def _validate_budget_fields(
        self,
        *,
        budget_usd: float | None,
        pricing_key: str,
        errors: list[str],
        context: str,
    ) -> None:
        if budget_usd is None:
            return
        if budget_usd <= 0:
            errors.append(f"{context}: budget_usd must be greater than 0.")
        if not self.pricing.enabled:
            errors.append(f"{context}: budget_usd requires pricing.enabled=true.")
            return
        if not pricing_key:
            errors.append(f"{context}: budget_usd requires a pricing key.")
            return
        if pricing_key not in self.pricing.models:
            errors.append(
                f"{context}: budget_usd requires pricing.models.{pricing_key} to be configured."
            )

    def outputs_path(self) -> Path:
        return self.run_dir() / "outputs.jsonl"

    def judge_outputs_path(self) -> Path:
        return self.run_dir() / "judge_outputs.jsonl"

    def batch_jobs_path(self) -> Path:
        return self.run_dir() / "batch_jobs.jsonl"

    def manifest_path(self) -> Path:
        return self.run_dir() / "run_manifest.json"

    def merged_path(self) -> Path:
        return self.run_dir() / "merged_results.parquet"

    def metrics_path(self) -> Path:
        return self.run_dir() / "metrics.json"

    def ground_truths_cache_path(self) -> Path:
        return self.run_dir() / "ground_truths.json"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from a YAML file."""
    raw: dict[str, Any] = {}

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)
