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


def test_legacy_root_fields_normalize_into_providers():
    cfg = _cfg(
        azure={
            "foundry": {"endpoint": "https://foundry.example"},
            "batch": {"endpoint": "https://openai.example", "api_version": "2024-10-21"},
        },
        google={"api_key_env": "GEMINI_API_KEY"},
        inference={"mode": "foundry", "model": "gpt-4o"},
        judging={"enabled": False},
    )
    assert cfg.providers.azure_ai_inference.endpoint == "https://foundry.example"
    assert cfg.providers.azure_openai.endpoint == "https://openai.example"
    assert cfg.providers.google_genai.api_key_env == "GEMINI_API_KEY"


def test_batch_limits_default_values():
    cfg = _cfg()
    assert cfg.providers.azure_openai.max_requests_per_file == 50_000
    assert cfg.providers.azure_openai.max_bytes_per_file == 100_000_000

