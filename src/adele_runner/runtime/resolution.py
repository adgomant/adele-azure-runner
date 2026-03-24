"""Resolve public config into internal targets and execution settings."""

from __future__ import annotations

from adele_runner.config import AppConfig, ConcurrencyConfig, compute_concurrency_from_rate_limits
from adele_runner.runtime.types import (
    ExecutionSettings,
    ResolvedInferenceTarget,
    ResolvedJudgeTarget,
)


def _settings_from_concurrency(cfg: ConcurrencyConfig) -> ExecutionSettings:
    return ExecutionSettings(
        max_in_flight=cfg.max_in_flight,
        request_timeout_s=cfg.request_timeout_s,
        max_retries=cfg.max_retries,
        backoff_base_s=cfg.backoff_base_s,
        backoff_max_s=cfg.backoff_max_s,
        max_poll_time_s=cfg.max_poll_time_s,
        batch_completion_window=cfg.batch_completion_window,
        effective_rpm=cfg.effective_rpm,
    )


def build_execution_settings(
    base: ConcurrencyConfig,
    *,
    rate_limits=None,  # type: ignore[no-untyped-def]
    max_tokens: int | None = None,
) -> ExecutionSettings:
    """Build per-stage execution settings without mutating the config."""
    if rate_limits is None:
        return _settings_from_concurrency(base)

    computed = compute_concurrency_from_rate_limits(rate_limits, max_tokens or 2048)
    computed.max_retries = base.max_retries
    computed.max_poll_time_s = base.max_poll_time_s
    computed.batch_completion_window = base.batch_completion_window
    return _settings_from_concurrency(computed)


def resolve_inference_target(config: AppConfig) -> ResolvedInferenceTarget:
    """Resolve the configured inference lane."""
    mode = config.resolve_inference_mode()
    if mode == "batch":
        return ResolvedInferenceTarget(
            adapter_kind="azure_openai",
            execution_kind="batch",
            model=config.inference.model,
            temperature=config.inference.temperature,
            max_tokens=config.inference.max_tokens,
            top_p=config.inference.top_p,
            rate_limits=None,
        )
    if mode == "google":
        return ResolvedInferenceTarget(
            adapter_kind="google_genai",
            execution_kind="request_response",
            model=config.inference.model,
            temperature=config.inference.temperature,
            max_tokens=config.inference.max_tokens,
            top_p=config.inference.top_p,
            rate_limits=config.inference.rate_limits,
        )
    return ResolvedInferenceTarget(
        adapter_kind="foundry",
        execution_kind="request_response",
        model=config.inference.model,
        temperature=config.inference.temperature,
        max_tokens=config.inference.max_tokens,
        top_p=config.inference.top_p,
        rate_limits=config.inference.rate_limits,
    )


def resolve_inference_execution_settings(config: AppConfig) -> ExecutionSettings:
    """Resolve inference execution settings."""
    target = resolve_inference_target(config)
    if target.execution_kind == "request_response":
        return build_execution_settings(
            config.concurrency,
            rate_limits=target.rate_limits,
            max_tokens=target.max_tokens,
        )
    return build_execution_settings(config.concurrency)


def resolve_judge_targets(config: AppConfig) -> list[ResolvedJudgeTarget]:
    """Resolve configured judges into explicit internal targets."""
    targets: list[ResolvedJudgeTarget] = []
    for judge in config.judging.judges:
        if judge.provider == "batch":
            targets.append(
                ResolvedJudgeTarget(
                    judge_name=judge.name,
                    adapter_kind="azure_openai",
                    execution_kind="batch",
                    model=judge.model,
                    prompt_template=config.judging.prompt_template,
                    max_tokens=judge.max_tokens,
                    rate_limits=None,
                )
            )
            continue
        targets.append(
            ResolvedJudgeTarget(
                judge_name=judge.name,
                adapter_kind="foundry",
                execution_kind="request_response",
                model=judge.model,
                prompt_template=config.judging.prompt_template,
                max_tokens=judge.max_tokens,
                rate_limits=judge.rate_limits,
            )
        )
    return targets


def resolve_judge_request_response_settings(config: AppConfig) -> ExecutionSettings:
    """Resolve shared request-response settings for the judge stage."""
    return build_execution_settings(
        config.concurrency,
        rate_limits=config.get_most_restrictive_judge_rate_limits(),
        max_tokens=config.get_max_judge_max_tokens(),
    )


def resolve_batch_execution_settings(config: AppConfig) -> ExecutionSettings:
    """Resolve settings for batch execution lanes."""
    return build_execution_settings(config.concurrency)

