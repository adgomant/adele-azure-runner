"""Tests for config validation."""

from __future__ import annotations

from adele_runner.config import AppConfig


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


def test_missing_model():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.endpoint.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": ""},
    )
    errors = cfg.validate_config()
    assert any("Model is not set" in error for error in errors)


def test_missing_azure_ai_inference_endpoint():
    cfg = _cfg(
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("Azure AI Inference endpoint" in error for error in errors)


def test_google_vertex_requires_project_and_location():
    cfg = _cfg(
        providers={"google_genai": {"backend": "vertex_ai"}},
        inference={"provider": "google_genai", "mode": "batch", "model": "gemini-2.5-flash"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("project" in error for error in errors)
    assert any("location" in error for error in errors)


def test_azure_openai_batch_requires_target_hint():
    cfg = _cfg(
        providers={"azure_openai": {"endpoint": "https://real.openai.azure.com"}},
        inference={"provider": "azure_openai", "mode": "batch", "model": "gpt-4o"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("requires targets.gpt-4o.supported_modes" in error for error in errors)


def test_azure_openai_batch_allowed_with_batch_capable_target():
    cfg = _cfg(
        providers={"azure_openai": {"endpoint": "https://real.openai.azure.com"}},
        targets={"gpt-4o": {"batch_capable": True}},
        inference={"provider": "azure_openai", "mode": "batch", "model": "gpt-4o"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert not any("batch-capable" in error for error in errors)


def test_azure_ai_inference_batch_rejected():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
        inference={"provider": "azure_ai_inference", "mode": "batch", "model": "gpt-4o"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("does not support batch mode" in error for error in errors)


def test_judging_enabled_no_judges():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={"enabled": True, "judges": []},
    )
    errors = cfg.validate_config()
    assert any("no judges configured" in error.lower() for error in errors)


def test_valid_prompt_templates():
    for template in ("v1", "v2"):
        cfg = _cfg(
            providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
            inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
            judging={"prompt_template": template, "enabled": False},
        )
        errors = cfg.validate_config()
        assert not any("template" in error.lower() for error in errors)


def test_batch_limits_default_values():
    cfg = _cfg()
    assert cfg.providers.azure_openai.max_requests_per_file == 100_000
    assert cfg.providers.azure_openai.max_bytes_per_file == 200_000_000


def test_anthropic_batch_limits_reject_excess_queue_budget():
    cfg = _cfg(
        inference={
            "provider": "anthropic",
            "mode": "batch",
            "model": "claude-sonnet-4-5",
            "rate_limits": {"batch_queue_requests": 100_001},
        },
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("batch_queue_requests" in error for error in errors)


def test_budget_requires_pricing_enabled():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
        inference={
            "provider": "azure_ai_inference",
            "mode": "request_response",
            "model": "gpt-4o",
            "budget_usd": 5.0,
        },
        judging={"enabled": False},
        pricing={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("pricing.enabled=true" in error for error in errors)


def test_budget_requires_matching_pricing_entry():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
        inference={
            "provider": "azure_ai_inference",
            "mode": "request_response",
            "model": "gpt-4o",
            "budget_usd": 5.0,
        },
        judging={"enabled": False},
        pricing={"enabled": True, "models": {"other-model": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}},
    )
    errors = cfg.validate_config()
    assert any("pricing.models.gpt-4o" in error for error in errors)


def test_judge_budget_requires_pricing_key_by_judge_name():
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={
            "enabled": True,
            "judges": [
                {
                    "name": "judge-a",
                    "provider": "azure_ai_inference",
                    "mode": "request_response",
                    "model": "gpt-4o",
                    "budget_usd": 1.0,
                }
            ],
        },
        pricing={"enabled": True, "models": {"gpt-4o": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}},
    )
    errors = cfg.validate_config()
    assert any("pricing.models.judge-a" in error for error in errors)


def test_judge_only_validation_skips_inference_provider_requirements():
    cfg = _cfg(
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={
            "enabled": True,
            "judges": [
                {
                    "name": "claude-batch",
                    "provider": "anthropic",
                    "mode": "batch",
                    "model": "claude-sonnet-4-5",
                }
            ],
        },
    )
    errors = cfg.validate_config(validate_inference=False, validate_judging=True)
    assert not any("Azure AI Inference endpoint" in error for error in errors)
