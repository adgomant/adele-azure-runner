"""Tests for internal target, binding, and execution resolution."""

from __future__ import annotations

import pytest

from adele_runner.config import AppConfig
from adele_runner.runtime.resolution import (
    resolve_inference_plan,
    resolve_inference_target,
    resolve_judge_plans,
    resolve_judge_targets,
)


def test_resolve_inference_target_google_request_response():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "provider": "google_genai",
                "mode": "request_response",
                "model": "gemini-2.5-flash",
                "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
            }
        }
    )

    target = resolve_inference_target(cfg)

    assert target.provider_kind == "google_genai"
    assert target.prompt_mode == "request_response"
    assert target.model == "gemini-2.5-flash"
    assert target.rate_limits is not None


def test_resolve_inference_plan_azure_openai_batch():
    cfg = AppConfig.model_validate(
        {
            "targets": {"gpt-4o": {"batch_capable": True}},
            "inference": {
                "provider": "azure_openai",
                "mode": "batch",
                "model": "gpt-4o",
            },
        }
    )

    plan = resolve_inference_plan(cfg)

    assert plan.binding.provider_target.provider_kind == "azure_openai"
    assert plan.binding.execution_kind == "batch"


def test_resolve_judge_targets_preserves_provider_mode_pairs():
    cfg = AppConfig.model_validate(
        {
            "targets": {"gpt-4o-mini": {"batch_capable": True}},
            "judging": {
                "prompt_template": "v2",
                "judges": [
                    {
                        "name": "j1",
                        "provider": "azure_ai_inference",
                        "mode": "request_response",
                        "model": "gpt-4o",
                    },
                    {
                        "name": "j2",
                        "provider": "azure_openai",
                        "mode": "batch",
                        "model": "gpt-4o-mini",
                    },
                ],
            },
        }
    )

    targets = resolve_judge_targets(cfg)

    assert [target.provider_kind for target in targets] == ["azure_ai_inference", "azure_openai"]
    assert [target.prompt_mode for target in targets] == ["request_response", "batch"]
    assert all(target.prompt_template == "v2" for target in targets)


def test_inference_execution_settings_auto_tune_without_mutating_config():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "provider": "azure_ai_inference",
                "mode": "request_response",
                "model": "gpt-4o",
                "max_tokens": 2048,
                "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
            },
            "concurrency": {"max_in_flight": 999},
        }
    )

    plan = resolve_inference_plan(cfg)

    assert plan.settings.max_in_flight != 999
    assert cfg.concurrency.max_in_flight == 999


def test_judge_plan_uses_target_specific_limits():
    cfg = AppConfig.model_validate(
        {
            "judging": {
                "judges": [
                    {
                        "name": "j1",
                        "provider": "azure_ai_inference",
                        "mode": "request_response",
                        "model": "gpt-4o",
                        "rate_limits": {"tokens_per_minute": 30_000, "requests_per_minute": 100},
                        "max_tokens": 1024,
                    }
                ]
            },
            "concurrency": {"max_in_flight": 999},
        }
    )

    plan = resolve_judge_plans(cfg)[0]

    assert plan.settings.max_in_flight != 999
    assert plan.settings.request_timeout_s >= 30.0


def test_azure_ai_inference_batch_binding_rejected():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "provider": "azure_ai_inference",
                "mode": "batch",
                "model": "gpt-4o",
            }
        }
    )

    with pytest.raises(ValueError, match="does not support batch mode"):
        resolve_inference_plan(cfg)


def test_google_batch_binding_resolves_for_gemini_api():
    cfg = AppConfig.model_validate(
        {
            "providers": {"google_genai": {"backend": "gemini_api"}},
            "inference": {
                "provider": "google_genai",
                "mode": "batch",
                "model": "gemini-2.5-flash",
            },
        }
    )

    plan = resolve_inference_plan(cfg)

    assert plan.binding.execution_kind == "batch"


def test_judge_plan_uses_target_provider_mode():
    cfg = AppConfig.model_validate(
        {
            "targets": {"claude-batch": {"supported_modes": ["batch"]}},
            "judging": {
                "judges": [
                    {
                        "name": "j1",
                        "provider": "anthropic",
                        "mode": "batch",
                        "model": "claude-batch",
                    }
                ]
            },
        }
    )

    plan = resolve_judge_plans(cfg)[0]

    assert plan.binding.provider_target.provider_kind == "anthropic"
    assert plan.binding.execution_kind == "batch"
