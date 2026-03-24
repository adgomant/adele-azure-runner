"""Tests for internal target and execution resolution."""

from __future__ import annotations

from adele_runner.config import AppConfig
from adele_runner.runtime.resolution import (
    resolve_inference_execution_settings,
    resolve_inference_target,
    resolve_judge_request_response_settings,
    resolve_judge_targets,
)


def test_resolve_inference_target_google():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "mode": "google",
                "model": "gemini-2.5-flash",
                "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
            }
        }
    )

    target = resolve_inference_target(cfg)

    assert target.adapter_kind == "google_genai"
    assert target.execution_kind == "request_response"
    assert target.model == "gemini-2.5-flash"
    assert target.rate_limits is not None


def test_resolve_inference_target_batch():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "mode": "batch",
                "model": "gpt-4o",
            }
        }
    )

    target = resolve_inference_target(cfg)

    assert target.adapter_kind == "azure_openai"
    assert target.execution_kind == "batch"


def test_resolve_judge_targets_maps_providers():
    cfg = AppConfig.model_validate(
        {
            "judging": {
                "prompt_template": "v2",
                "judges": [
                    {"name": "j1", "provider": "foundry", "model": "gpt-4o"},
                    {"name": "j2", "provider": "batch", "model": "gpt-4o-mini"},
                ],
            }
        }
    )

    targets = resolve_judge_targets(cfg)

    assert [target.adapter_kind for target in targets] == ["foundry", "azure_openai"]
    assert [target.execution_kind for target in targets] == ["request_response", "batch"]
    assert all(target.prompt_template == "v2" for target in targets)


def test_inference_execution_settings_auto_tune_without_mutating_config():
    cfg = AppConfig.model_validate(
        {
            "inference": {
                "mode": "foundry",
                "model": "gpt-4o",
                "max_tokens": 2048,
                "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
            },
            "concurrency": {"max_in_flight": 999},
        }
    )

    settings = resolve_inference_execution_settings(cfg)

    assert settings.max_in_flight != 999
    assert cfg.concurrency.max_in_flight == 999


def test_judge_request_response_settings_use_most_restrictive_limits():
    cfg = AppConfig.model_validate(
        {
            "judging": {
                "judges": [
                    {
                        "name": "j1",
                        "provider": "foundry",
                        "model": "gpt-4o",
                        "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
                        "max_tokens": 256,
                    },
                    {
                        "name": "j2",
                        "provider": "foundry",
                        "model": "claude",
                        "rate_limits": {"tokens_per_minute": 30_000, "requests_per_minute": 100},
                        "max_tokens": 1024,
                    },
                ]
            },
            "concurrency": {"max_in_flight": 999},
        }
    )

    settings = resolve_judge_request_response_settings(cfg)

    assert settings.max_in_flight != 999
    assert settings.request_timeout_s >= 30.0

