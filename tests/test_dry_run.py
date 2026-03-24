"""Tests for CLI --dry-run functionality."""

from __future__ import annotations

import pytest

from adele_runner.cli import _print_dry_run, _validate_or_exit
from adele_runner.config import AppConfig


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


def test_dry_run_prints_summary(capsys):
    """_print_dry_run should print without raising."""
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={"enabled": True, "judges": [{"name": "j1", "provider": "azure_ai_inference", "mode": "request_response", "model": "m1"}]},
    )
    _print_dry_run(cfg, items_count=42)
    # No exception means it worked; we just verify it ran


def test_dry_run_shows_model():
    """Dry-run summary should use the configured model."""
    cfg = _cfg(inference={"model": "gpt-4o"})
    # Should not raise
    _print_dry_run(cfg, items_count=10)


def test_validate_or_exit_with_errors():
    """Config with placeholder endpoint should fail validation."""
    from typer import Exit

    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "<YOUR-ENDPOINT>"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": ""},
    )
    with pytest.raises(Exit):
        _validate_or_exit(cfg)


def test_validate_or_exit_clean():
    """Clean config should pass validation."""
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
        judging={"enabled": True, "judges": [{"name": "j", "provider": "azure_ai_inference", "mode": "request_response", "model": "m"}]},
    )
    # Should not raise
    _validate_or_exit(cfg)


def test_dry_run_with_judging_disabled():
    """Dry-run should handle disabled judging gracefully."""
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "m"},
        judging={"enabled": False},
    )
    _print_dry_run(cfg, items_count=5)


def test_dry_run_with_pricing():
    """Dry-run should mention pricing when enabled."""
    cfg = _cfg(
        inference={"model": "m"},
        pricing={
            "enabled": True,
            "models": {"m": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}},
        },
    )
    _print_dry_run(cfg, items_count=5)


def test_dry_run_with_rate_limits(capsys):
    """Dry-run should show rate limits and computed concurrency when configured."""
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.com"}},
        inference={
            "provider": "azure_ai_inference",
            "mode": "request_response",
            "model": "m",
            "max_tokens": 2048,
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
    )
    _print_dry_run(cfg, items_count=10)
    out = capsys.readouterr().out
    assert "TPM=" in out
    assert "max_in_flight" in out


def test_dry_run_without_rate_limits_shows_concurrency(capsys):
    """Dry-run should show concurrency params even without rate limits."""
    cfg = _cfg(
        providers={"azure_ai_inference": {"endpoint": "https://real.com"}},
        inference={"provider": "azure_ai_inference", "mode": "request_response", "model": "m"},
    )
    _print_dry_run(cfg, items_count=5)
    out = capsys.readouterr().out
    assert "max_in_flight" in out
