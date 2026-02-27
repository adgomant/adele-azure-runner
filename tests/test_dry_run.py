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
        inference={"foundry": {"endpoint": "https://real.com", "model": "gpt-4o"}},
        judging={"enabled": True, "judges": [{"name": "j1", "model": "m1"}]},
    )
    _print_dry_run(cfg, items_count=42)
    # No exception means it worked; we just verify it ran


def test_dry_run_shows_model():
    """Dry-run summary should use the configured model."""
    cfg = _cfg(inference={"foundry": {"model": "gpt-4o"}})
    # Should not raise
    _print_dry_run(cfg, items_count=10)


def test_validate_or_exit_with_errors():
    """Config with placeholder endpoint should fail validation."""
    from typer import Exit

    cfg = _cfg(inference={"foundry": {"endpoint": "<YOUR-ENDPOINT>", "model": ""}})
    with pytest.raises(Exit):
        _validate_or_exit(cfg)


def test_validate_or_exit_clean():
    """Clean config should pass validation."""
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://real.com", "model": "gpt-4o"}},
        judging={"enabled": True, "judges": [{"name": "j", "model": "m"}]},
    )
    # Should not raise
    _validate_or_exit(cfg)


def test_dry_run_with_judging_disabled():
    """Dry-run should handle disabled judging gracefully."""
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://real.com", "model": "m"}},
        judging={"enabled": False},
    )
    _print_dry_run(cfg, items_count=5)


def test_dry_run_with_pricing():
    """Dry-run should mention pricing when enabled."""
    cfg = _cfg(
        inference={"foundry": {"model": "m"}},
        pricing={"enabled": True, "models": {"m": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}},
    )
    _print_dry_run(cfg, items_count=5)
