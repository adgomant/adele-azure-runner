"""Tests for provider-specific request and batch budget resolution."""

from __future__ import annotations

from adele_runner.config import AppConfig
from adele_runner.runtime.resolution import resolve_inference_plan


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


def test_azure_openai_request_budget_resolves_with_header_feedback():
    cfg = _cfg(
        providers={"azure_openai": {"endpoint": "https://real.openai.azure.com"}},
        inference={
            "provider": "azure_openai",
            "mode": "request_response",
            "model": "gpt-4o",
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
        judging={"enabled": False},
    )

    plan = resolve_inference_plan(cfg)
    assert plan.settings.request_budget is not None
    assert plan.settings.request_budget.provider_kind == "azure_openai"
    assert plan.settings.request_budget.enable_header_feedback is True


def test_anthropic_request_budget_supports_input_and_output_tokens():
    cfg = _cfg(
        inference={
            "provider": "anthropic",
            "mode": "request_response",
            "model": "claude-sonnet-4-5",
            "rate_limits": {
                "requests_per_minute": 50,
                "input_tokens_per_minute": 30_000,
                "output_tokens_per_minute": 8_000,
            },
        },
        judging={"enabled": False},
    )

    plan = resolve_inference_plan(cfg)
    assert plan.settings.request_budget is not None
    assert plan.settings.request_budget.input_tokens_per_minute == 30_000
    assert plan.settings.request_budget.output_tokens_per_minute == 8_000


def test_google_gemini_api_budget_uses_pacific_day_reset():
    cfg = _cfg(
        providers={"google_genai": {"backend": "gemini_api"}},
        inference={
            "provider": "google_genai",
            "mode": "request_response",
            "model": "gemini-2.5-flash",
            "rate_limits": {"requests_per_day": 1_000},
        },
        judging={"enabled": False},
    )

    plan = resolve_inference_plan(cfg)
    assert plan.settings.request_budget is not None
    assert plan.settings.request_budget.daily_reset_timezone == "America/Los_Angeles"


def test_google_vertex_budget_stays_reactive_only_without_limits():
    cfg = _cfg(
        providers={
            "google_genai": {"backend": "vertex_ai", "project": "p", "location": "us-central1"}
        },
        inference={"provider": "google_genai", "mode": "request_response", "model": "gemini-2.5-flash"},
        judging={"enabled": False},
    )

    plan = resolve_inference_plan(cfg)
    assert plan.settings.request_budget is not None
    assert plan.settings.request_budget.has_limits() is False
    assert plan.settings.request_budget.daily_reset_timezone is None


def test_azure_openai_batch_budget_uses_updated_hard_caps():
    cfg = _cfg(
        providers={"azure_openai": {"endpoint": "https://real.openai.azure.com"}},
        targets={"gpt-4o": {"batch_capable": True}},
        inference={"provider": "azure_openai", "mode": "batch", "model": "gpt-4o"},
        judging={"enabled": False},
    )

    plan = resolve_inference_plan(cfg)
    assert plan.settings.batch_budget is not None
    assert plan.settings.batch_budget.max_requests_per_batch == 100_000
    assert plan.settings.batch_budget.max_bytes_per_batch == 200_000_000
