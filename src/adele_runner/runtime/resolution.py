"""Resolve public config into internal targets, bindings, and execution settings."""

from __future__ import annotations

from adele_runner.config import AppConfig, ConcurrencyConfig, compute_concurrency_from_rate_limits
from adele_runner.runtime.provider_limits import (
    resolve_batch_budget,
    resolve_request_budget,
)
from adele_runner.runtime.registry import bind_mode
from adele_runner.runtime.types import (
    ExecutionKind,
    ExecutionSettings,
    ResolvedInferencePlan,
    ResolvedInferenceTarget,
    ResolvedJudgePlan,
    ResolvedJudgeTarget,
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
    config: AppConfig | None = None,
    target: ResolvedProviderTarget | None = None,
    execution_kind: ExecutionKind = "request_response",
    rate_limits=None,  # type: ignore[no-untyped-def]
    max_tokens: int | None = None,
) -> ExecutionSettings:
    if config is None or target is None:
        if execution_kind == "batch" or rate_limits is None:
            return _settings_from_concurrency(base)

        computed = compute_concurrency_from_rate_limits(rate_limits, max_tokens or 2048)
        computed.max_retries = base.max_retries
        computed.max_poll_time_s = base.max_poll_time_s
        computed.batch_completion_window = base.batch_completion_window
        return _settings_from_concurrency(computed)

    if execution_kind == "batch":
        settings = _settings_from_concurrency(base)
        return ExecutionSettings(
            max_in_flight=settings.max_in_flight,
            request_timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
            max_poll_time_s=settings.max_poll_time_s,
            batch_completion_window=settings.batch_completion_window,
            effective_rpm=settings.effective_rpm,
            batch_budget=resolve_batch_budget(config, target),
        )

    request_budget = resolve_request_budget(config, target, max_tokens=max_tokens or 2048)
    if not request_budget.has_limits():
        settings = _settings_from_concurrency(base)
        clamped_concurrency = (
            min(settings.max_in_flight, request_budget.concurrent_requests)
            if request_budget.concurrent_requests is not None
            else settings.max_in_flight
        )
        return ExecutionSettings(
            max_in_flight=clamped_concurrency,
            request_timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
            max_poll_time_s=settings.max_poll_time_s,
            batch_completion_window=settings.batch_completion_window,
            effective_rpm=settings.effective_rpm,
            request_budget=request_budget,
        )

    computed = compute_concurrency_from_rate_limits(rate_limits, max_tokens or 2048)
    computed.max_retries = base.max_retries
    computed.max_poll_time_s = base.max_poll_time_s
    computed.batch_completion_window = base.batch_completion_window
    settings = _settings_from_concurrency(computed)
    return ExecutionSettings(
        max_in_flight=settings.max_in_flight,
        request_timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        backoff_base_s=settings.backoff_base_s,
        backoff_max_s=settings.backoff_max_s,
        max_poll_time_s=settings.max_poll_time_s,
        batch_completion_window=settings.batch_completion_window,
        effective_rpm=settings.effective_rpm,
        request_budget=request_budget,
    )


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


def resolve_execution_settings_for_target(
    config: AppConfig,
    target: ResolvedProviderTarget,
    execution_kind: ExecutionKind,
    *,
    rate_limits=None,  # type: ignore[no-untyped-def]
    max_tokens: int | None = None,
) -> ExecutionSettings:
    return build_execution_settings(
        config.concurrency,
        config=config,
        target=target,
        execution_kind=execution_kind,
        rate_limits=rate_limits,
        max_tokens=max_tokens,
    )


def resolve_inference_plan(config: AppConfig) -> ResolvedInferencePlan:
    target = resolve_inference_target(config)
    binding = bind_mode(config, target.provider_target, target.requested_mode)
    settings = resolve_execution_settings_for_target(
        config,
        target.provider_target,
        binding.execution_kind,
        rate_limits=target.rate_limits,
        max_tokens=target.max_tokens,
    )
    return ResolvedInferencePlan(target=target, binding=binding, settings=settings)


def resolve_judge_plans(config: AppConfig) -> list[ResolvedJudgePlan]:
    plans: list[ResolvedJudgePlan] = []
    for target in resolve_judge_targets(config):
        binding = bind_mode(config, target.provider_target, target.requested_mode)
        settings = resolve_execution_settings_for_target(
            config,
            target.provider_target,
            binding.execution_kind,
            rate_limits=target.rate_limits,
            max_tokens=target.max_tokens,
        )
        plans.append(ResolvedJudgePlan(target=target, binding=binding, settings=settings))
    return plans
