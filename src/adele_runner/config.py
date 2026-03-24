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

ProviderKind = Literal["azure_openai", "azure_ai_inference", "google_genai", "anthropic"]
ExecutionMode = Literal["request_response", "batch", "auto"]
ResolvedExecutionKind = Literal["request_response", "batch"]
GoogleBackend = Literal["gemini_api", "vertex_ai"]
TargetSupportedMode = Literal["request_response", "batch"]

_PLACEHOLDER_RE = re.compile(r"<[A-Z_-]+>")
_LEGACY_PROVIDER_ALIAS = {
    "foundry": "azure_ai_inference",
    "google": "google_genai",
}
_LEGACY_INFERENCE_MODE_ALIAS = {
    "foundry": ("azure_ai_inference", "request_response"),
    "google": ("google_genai", "request_response"),
    "batch": ("azure_openai", "batch"),
}
_AZURE_BATCH_DEPLOYMENT_TYPES = {"global_batch", "data_zone_batch"}


class RateLimitsConfig(BaseModel):
    """Rate limits for a provider target."""

    tokens_per_minute: int
    requests_per_minute: int


class AzureOpenAIConnection(BaseModel):
    endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"
    max_requests_per_file: int = 50_000
    max_bytes_per_file: int = 100_000_000


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


class AzureFoundryConnection(BaseModel):
    """Legacy compatibility shape for older config files."""

    endpoint: str = ""
    api_key_env: str = "AZURE_AI_API_KEY"


class AzureBatchConnection(BaseModel):
    """Legacy compatibility shape for older config files."""

    endpoint: str = ""
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version: str = ""
    completion_endpoint: str = "/chat/completions"
    max_requests_per_file: int = 50_000
    max_bytes_per_file: int = 100_000_000


class AzureConfig(BaseModel):
    """Legacy compatibility shape for older config files."""

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

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        mode = data.get("mode")
        provider = data.get("provider")

        if provider in _LEGACY_PROVIDER_ALIAS:
            logger.warning("Deprecated provider alias '%s'; use '%s'.", provider, _LEGACY_PROVIDER_ALIAS[provider])
            data["provider"] = _LEGACY_PROVIDER_ALIAS[provider]

        if mode in _LEGACY_INFERENCE_MODE_ALIAS:
            mapped_provider, mapped_mode = _LEGACY_INFERENCE_MODE_ALIAS[mode]
            logger.warning(
                "Deprecated inference.mode '%s'; use provider='%s' and mode='%s'.",
                mode,
                mapped_provider,
                mapped_mode,
            )
            data.setdefault("provider", mapped_provider)
            data["mode"] = mapped_mode

        if provider is None and "provider" not in data:
            data["provider"] = "azure_ai_inference"

        return data


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

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        provider = data.get("provider")
        mode = data.get("mode")

        if provider == "batch" and mode is None:
            logger.warning(
                "Deprecated judge provider 'batch'; use provider='azure_openai' and mode='batch'."
            )
            data["provider"] = "azure_openai"
            data["mode"] = "batch"
        elif provider == "foundry" and mode is None:
            logger.warning(
                "Deprecated judge provider 'foundry'; use provider='azure_ai_inference' and mode='request_response'."
            )
            data["provider"] = "azure_ai_inference"
            data["mode"] = "request_response"
        elif provider in _LEGACY_PROVIDER_ALIAS:
            logger.warning("Deprecated judge provider alias '%s'.", provider)
            data["provider"] = _LEGACY_PROVIDER_ALIAS[provider]

        if mode in _LEGACY_INFERENCE_MODE_ALIAS:
            mapped_provider, mapped_mode = _LEGACY_INFERENCE_MODE_ALIAS[mode]
            data.setdefault("provider", mapped_provider)
            data["mode"] = mapped_mode

        data.setdefault("provider", "azure_ai_inference")
        data.setdefault("mode", "request_response")
        return data


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


def compute_concurrency_from_rate_limits(
    rate_limits: RateLimitsConfig,
    max_tokens: int,
) -> ConcurrencyConfig:
    """Compute optimal concurrency parameters from rate limits."""
    tpm = rate_limits.tokens_per_minute
    rpm = rate_limits.requests_per_minute

    est_tokens_per_request = max(1, max_tokens)
    rpm_from_tpm = tpm / est_tokens_per_request
    effective_rpm = min(rpm, rpm_from_tpm)
    avg_duration_s = max(2.0, min(300.0, max_tokens / 200))
    max_in_flight = max(1, int(effective_rpm * avg_duration_s / 60 * 0.8))
    request_timeout_s = round(max(30.0, min(600.0, max_tokens / 50 + 10)), 1)
    backoff_base_s = round(max(0.5, min(10.0, 60.0 / max(1, effective_rpm))), 1)
    backoff_max_s = round(max(30.0, min(120.0, backoff_base_s * 10)), 1)

    return ConcurrencyConfig(
        max_in_flight=max_in_flight,
        request_timeout_s=request_timeout_s,
        backoff_base_s=backoff_base_s,
        backoff_max_s=backoff_max_s,
        effective_rpm=max(1, int(effective_rpm)),
    )


class AppConfig(BaseModel):
    """Root config object."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    google: GoogleGenAIConnection = Field(default_factory=GoogleGenAIConnection)
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_root_fields(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        providers = dict(data.get("providers") or {})

        azure = data.get("azure") or {}
        foundry = azure.get("foundry") or {}
        batch = azure.get("batch") or {}
        google = data.get("google") or {}

        if foundry:
            providers.setdefault("azure_ai_inference", {})
            for key, value in foundry.items():
                providers["azure_ai_inference"].setdefault(key, value)

        if batch:
            providers.setdefault("azure_openai", {})
            for key, value in batch.items():
                providers["azure_openai"].setdefault(key, value)

        if google:
            providers.setdefault("google_genai", {})
            for key, value in google.items():
                providers["google_genai"].setdefault(key, value)

        data["providers"] = providers
        return data

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

    def get_foundry_api_key(self) -> str:
        return self.get_provider_api_key("azure_ai_inference")

    def get_batch_api_key(self) -> str:
        return self.get_provider_api_key("azure_openai")

    def get_google_api_key(self) -> str:
        return self.get_provider_api_key("google_genai")

    def get_anthropic_api_key(self) -> str:
        return self.get_provider_api_key("anthropic")

    def resolve_inference_mode(self) -> ResolvedExecutionKind:
        """Legacy helper retained for compatibility."""
        if self.inference.mode == "batch":
            return "batch"
        if self.inference.mode == "auto" and self.inference.provider == "azure_openai":
            target_cfg = self.get_target_config(self.inference.model)
            if target_cfg and target_cfg.supported_modes == ["batch"]:
                return "batch"
        return "request_response"

    def get_target_config(self, model: str) -> TargetConfig | None:
        return self.targets.get(model)

    def apply_rate_limit_overrides(
        self,
        rate_limits: RateLimitsConfig | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Legacy helper retained for compatibility with tests/dry runs."""
        if rate_limits is None:
            return

        mt = max_tokens or self.inference.max_tokens
        computed = compute_concurrency_from_rate_limits(rate_limits, mt)
        computed.max_retries = self.concurrency.max_retries
        computed.max_poll_time_s = self.concurrency.max_poll_time_s
        computed.batch_completion_window = self.concurrency.batch_completion_window
        self.concurrency = computed

    def get_most_restrictive_judge_rate_limits(self) -> RateLimitsConfig | None:
        limits = [
            j.rate_limits
            for j in self.judging.judges
            if j.mode == "request_response" and j.rate_limits is not None
        ]
        if not limits:
            return None
        return min(limits, key=lambda rl: rl.tokens_per_minute)

    def get_max_judge_max_tokens(self) -> int:
        values = [j.max_tokens for j in self.judging.judges if j.mode == "request_response"]
        return max(values) if values else 512

    def _validate_provider_connection(self, provider: ProviderKind, errors: list[str]) -> None:
        if provider == "azure_ai_inference":
            ep = self.providers.azure_ai_inference.endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    "Azure AI Inference endpoint is not configured. "
                    "Set providers.azure_ai_inference.endpoint in config.yaml."
                )
        elif provider == "azure_openai":
            ep = self.providers.azure_openai.endpoint
            if not ep or _PLACEHOLDER_RE.search(ep):
                errors.append(
                    "Azure OpenAI endpoint is not configured. "
                    "Set providers.azure_openai.endpoint in config.yaml."
                )
        elif provider == "google_genai":
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
            google_cfg = self.providers.google_genai
            if mode == "batch" and google_cfg.backend not in {"gemini_api", "vertex_ai"}:
                errors.append(
                    f"{context}: provider 'google_genai' does not support batch for backend '{google_cfg.backend}'."
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
            if target_cfg.deployment_type and target_cfg.deployment_type.lower() in _AZURE_BATCH_DEPLOYMENT_TYPES:
                return
            errors.append(
                f"{context}: target '{model}' is not marked as batch-capable for provider 'azure_openai'."
            )

    def validate_config(self, *, dry_run: bool = False) -> list[str]:
        """Validate config for common mistakes. Returns list of error messages."""
        errors: list[str] = []

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

        if self.judging.enabled and not self.judging.judges:
            errors.append(
                "Judging is enabled but no judges configured. Use --judge or set judging.judges."
            )

        if self.judging.prompt_template not in ("v1", "v2"):
            errors.append(
                f"Unknown judge prompt template: '{self.judging.prompt_template}'. Use 'v1' or 'v2'."
            )

        for judge in self.judging.judges:
            self._validate_provider_connection(judge.provider, errors)
            self._validate_target_mode_support(
                judge.provider,
                judge.mode,
                judge.model,
                errors,
                context=f"Judge '{judge.name}'",
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
