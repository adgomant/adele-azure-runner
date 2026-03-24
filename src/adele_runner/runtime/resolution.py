"""Resolve public config into internal targets, bindings, and execution settings."""

from __future__ import annotations

from adele_runner.config import AppConfig, ConcurrencyConfig, compute_concurrency_from_rate_limits
from adele_runner.runtime.registry import bind_mode
from adele_runner.runtime.types import (
    ExecutionKind,
    ExecutionSettings,
    ResolvedInferenceTarget,
    ResolvedJudgeTarget,
    ResolvedModeBinding,
    ResolvedProviderTarget,
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
    if rate_limits is None:
        return _settings_from_concurrency(base)

    computed = compute_concurrency_from_rate_limits(rate_limits, max_tokens or 2048)
    computed.max_retries = base.max_retries
    computed.max_poll_time_s = base.max_poll_time_s
    computed.batch_completion_window = base.batch_completion_window
    return _settings_from_concurrency(computed)


def _provider_target_from_config(
    config: AppConfig,
    *,
    provider_kind,
    model: str,
    rate_limits=None,  # type: ignore[no-untyped-def]
) -> ResolvedProviderTarget:
    target_cfg = config.get_target_config(model)
    metadata: dict[str, object] = {}
    if target_cfg is not None:
        if target_cfg.supported_modes is not None:
            metadata["supported_modes"] = list(target_cfg.supported_modes)
        if target_cfg.batch_capable is not None:
            metadata["batch_capable"] = target_cfg.batch_capable
        if target_cfg.deployment_type is not None:
            metadata["deployment_type"] = target_cfg.deployment_type
    return ResolvedProviderTarget(
        provider_kind=provider_kind,
        model=model,
        rate_limits=rate_limits,
        metadata=metadata,
    )


def resolve_inference_target(config: AppConfig) -> ResolvedInferenceTarget:
    provider_target = _provider_target_from_config(
        config,
        provider_kind=config.inference.provider,
        model=config.inference.model,
        rate_limits=config.inference.rate_limits,
    )
    binding = bind_mode(config, provider_target, config.inference.mode)
    return ResolvedInferenceTarget(
        provider_target=provider_target,
        requested_mode=config.inference.mode,
        prompt_mode=binding.execution_kind,
        temperature=config.inference.temperature,
        max_tokens=config.inference.max_tokens,
        top_p=config.inference.top_p,
    )


def resolve_inference_binding(config: AppConfig, target: ResolvedInferenceTarget) -> ResolvedModeBinding:
    return bind_mode(config, target.provider_target, target.requested_mode)


def resolve_judge_targets(config: AppConfig) -> list[ResolvedJudgeTarget]:
    targets: list[ResolvedJudgeTarget] = []
    for judge in config.judging.judges:
        provider_target = _provider_target_from_config(
            config,
            provider_kind=judge.provider,
            model=judge.model,
            rate_limits=judge.rate_limits,
        )
        binding = bind_mode(config, provider_target, judge.mode)
        targets.append(
            ResolvedJudgeTarget(
                judge_name=judge.name,
                provider_target=provider_target,
                requested_mode=judge.mode,
                prompt_mode=binding.execution_kind,
                prompt_template=config.judging.prompt_template,
                max_tokens=judge.max_tokens,
            )
        )
    return targets


def resolve_judge_binding(config: AppConfig, target: ResolvedJudgeTarget) -> ResolvedModeBinding:
    return bind_mode(config, target.provider_target, target.requested_mode)


def resolve_execution_settings_for_target(
    config: AppConfig,
    execution_kind: ExecutionKind,
    *,
    rate_limits=None,  # type: ignore[no-untyped-def]
    max_tokens: int | None = None,
) -> ExecutionSettings:
    if execution_kind == "request_response":
        return build_execution_settings(
            config.concurrency,
            rate_limits=rate_limits,
            max_tokens=max_tokens,
        )
    return build_execution_settings(config.concurrency)


def resolve_inference_execution_settings(config: AppConfig) -> ExecutionSettings:
    target = resolve_inference_target(config)
    return resolve_execution_settings_for_target(
        config,
        target.prompt_mode,
        rate_limits=target.rate_limits,
        max_tokens=target.max_tokens,
    )


def resolve_judge_execution_settings(config: AppConfig, target: ResolvedJudgeTarget) -> ExecutionSettings:
    return resolve_execution_settings_for_target(
        config,
        target.prompt_mode,
        rate_limits=target.rate_limits,
        max_tokens=target.max_tokens,
    )
