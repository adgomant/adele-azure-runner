"""Provider-specific rate-limit and batch-limit policy resolution."""

from __future__ import annotations

from dataclasses import dataclass

from adele_runner.config import AppConfig, RateLimitsConfig, compute_concurrency_from_rate_limits
from adele_runner.runtime.types import ConcurrencySummary, ProviderKind, ResolvedProviderTarget

AZURE_BATCH_MAX_REQUESTS = 100_000
AZURE_BATCH_MAX_FILE_BYTES = 200_000_000
ANTHROPIC_BATCH_MAX_REQUESTS = 100_000
ANTHROPIC_BATCH_MAX_BYTES = 256_000_000
ANTHROPIC_BATCH_MAX_QUEUE_REQUESTS = 300_000
ANTHROPIC_BATCH_RPM = 50
GOOGLE_GEMINI_BATCH_CONCURRENT_JOBS = 100
GOOGLE_GEMINI_BATCH_MAX_BYTES = 2_000_000_000

_GOOGLE_DAILY_RESET_TZ = "America/Los_Angeles"


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """Normalized request-response budgets used by the adaptive limiter."""

    provider_kind: ProviderKind
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    concurrent_requests: int | None = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    daily_reset_timezone: str | None = None
    enable_header_feedback: bool = False

    @property
    def estimated_total_tokens(self) -> int:
        return max(1, self.estimated_input_tokens + self.estimated_output_tokens)

    def has_limits(self) -> bool:
        return any(
            value is not None
            for value in (
                self.requests_per_minute,
                self.tokens_per_minute,
                self.input_tokens_per_minute,
                self.output_tokens_per_minute,
                self.requests_per_day,
                self.tokens_per_day,
                self.concurrent_requests,
            )
        )


@dataclass(frozen=True, slots=True)
class BatchBudget:
    """Normalized batch-mode hard caps and queue budgets."""

    provider_kind: ProviderKind
    max_requests_per_batch: int | None = None
    max_bytes_per_batch: int | None = None
    batch_requests_per_minute: int | None = None
    batch_queue_requests: int | None = None
    batch_enqueued_tokens: int | None = None
    active_jobs_limit: int | None = None


def _rate_limits_or_default(rate_limits: RateLimitsConfig | None) -> RateLimitsConfig:
    return rate_limits or RateLimitsConfig()


def _build_request_budget(
    provider: ProviderKind,
    rate_limits: RateLimitsConfig | None,
    *,
    max_tokens: int,
    daily_reset_timezone: str | None = None,
    enable_header_feedback: bool = False,
) -> RequestBudget:
    limits = _rate_limits_or_default(rate_limits)
    estimate = max(1, max_tokens)
    return RequestBudget(
        provider_kind=provider,
        requests_per_minute=limits.requests_per_minute,
        tokens_per_minute=limits.tokens_per_minute,
        input_tokens_per_minute=limits.input_tokens_per_minute,
        output_tokens_per_minute=limits.output_tokens_per_minute,
        requests_per_day=limits.requests_per_day,
        tokens_per_day=limits.tokens_per_day,
        concurrent_requests=limits.concurrent_requests,
        estimated_input_tokens=estimate,
        estimated_output_tokens=estimate,
        daily_reset_timezone=daily_reset_timezone,
        enable_header_feedback=enable_header_feedback,
    )


def resolve_request_budget(
    config: AppConfig,
    target: ResolvedProviderTarget,
    *,
    max_tokens: int,
) -> RequestBudget:
    if target.provider_kind == "google_genai":
        daily_reset_tz = (
            _GOOGLE_DAILY_RESET_TZ if config.providers.google_genai.backend == "gemini_api" else None
        )
        return _build_request_budget(
            "google_genai",
            target.rate_limits,
            max_tokens=max_tokens,
            daily_reset_timezone=daily_reset_tz,
            enable_header_feedback=False,
        )

    if target.provider_kind == "anthropic":
        return _build_request_budget(
            "anthropic",
            target.rate_limits,
            max_tokens=max_tokens,
            enable_header_feedback=True,
        )

    if target.provider_kind in {"azure_openai", "azure_ai_inference"}:
        return _build_request_budget(
            target.provider_kind,
            target.rate_limits,
            max_tokens=max_tokens,
            enable_header_feedback=True,
        )

    return _build_request_budget(target.provider_kind, target.rate_limits, max_tokens=max_tokens)


def resolve_batch_budget(
    config: AppConfig,
    target: ResolvedProviderTarget,
) -> BatchBudget:
    limits = _rate_limits_or_default(target.rate_limits)

    if target.provider_kind == "azure_openai":
        azure_cfg = config.providers.azure_openai
        return BatchBudget(
            provider_kind="azure_openai",
            max_requests_per_batch=min(azure_cfg.max_requests_per_file, AZURE_BATCH_MAX_REQUESTS),
            max_bytes_per_batch=min(azure_cfg.max_bytes_per_file, AZURE_BATCH_MAX_FILE_BYTES),
            batch_requests_per_minute=limits.batch_requests_per_minute,
            batch_queue_requests=limits.batch_queue_requests,
            batch_enqueued_tokens=limits.batch_enqueued_tokens,
        )

    if target.provider_kind == "anthropic":
        return BatchBudget(
            provider_kind="anthropic",
            max_requests_per_batch=ANTHROPIC_BATCH_MAX_REQUESTS,
            max_bytes_per_batch=ANTHROPIC_BATCH_MAX_BYTES,
            batch_requests_per_minute=min(
                limits.batch_requests_per_minute or ANTHROPIC_BATCH_RPM,
                ANTHROPIC_BATCH_RPM,
            ),
            batch_queue_requests=min(
                limits.batch_queue_requests or ANTHROPIC_BATCH_MAX_QUEUE_REQUESTS,
                ANTHROPIC_BATCH_MAX_QUEUE_REQUESTS,
            ),
            batch_enqueued_tokens=limits.batch_enqueued_tokens,
        )

    if target.provider_kind == "google_genai":
        active_jobs_limit = (
            GOOGLE_GEMINI_BATCH_CONCURRENT_JOBS
            if config.providers.google_genai.backend == "gemini_api"
            else None
        )
        max_bytes = (
            GOOGLE_GEMINI_BATCH_MAX_BYTES
            if config.providers.google_genai.backend == "gemini_api"
            else None
        )
        return BatchBudget(
            provider_kind="google_genai",
            max_requests_per_batch=None,
            max_bytes_per_batch=max_bytes,
            batch_requests_per_minute=limits.batch_requests_per_minute,
            batch_queue_requests=limits.batch_queue_requests,
            batch_enqueued_tokens=limits.batch_enqueued_tokens,
            active_jobs_limit=active_jobs_limit,
        )

    return BatchBudget(provider_kind=target.provider_kind)


def summarize_request_budget(
    budget: RequestBudget,
    *,
    max_tokens: int,
) -> ConcurrencySummary:
    limits = RateLimitsConfig(
        tokens_per_minute=budget.tokens_per_minute,
        requests_per_minute=budget.requests_per_minute,
        input_tokens_per_minute=budget.input_tokens_per_minute,
        output_tokens_per_minute=budget.output_tokens_per_minute,
        requests_per_day=budget.requests_per_day,
        tokens_per_day=budget.tokens_per_day,
        concurrent_requests=budget.concurrent_requests,
    )
    computed = compute_concurrency_from_rate_limits(limits, max_tokens=max_tokens)
    return ConcurrencySummary(
        effective_rpm=computed.effective_rpm,
        max_in_flight=computed.max_in_flight,
    )
