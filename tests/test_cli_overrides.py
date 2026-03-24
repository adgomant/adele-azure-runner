"""Tests for CLI override logic."""

from __future__ import annotations

import pytest
import typer

from adele_runner.cli import apply_cli_overrides
from adele_runner.config import AppConfig


def _base_config(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


def test_run_id_overrides_config():
    cfg = _base_config(run={"run_id": "from-config"})
    apply_cli_overrides(cfg, run_id="from-cli")
    assert cfg.run.run_id == "from-cli"


def test_inference_provider_and_mode_override():
    cfg = _base_config()
    apply_cli_overrides(
        cfg,
        provider="azure_openai",
        mode="request_response",
        model="gpt-4o",
    )
    assert cfg.inference.provider == "azure_openai"
    assert cfg.inference.mode == "request_response"
    assert cfg.inference.model == "gpt-4o"


def test_legacy_mode_alias_normalizes_to_provider_and_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="foundry")
    assert cfg.inference.provider == "azure_ai_inference"
    assert cfg.inference.mode == "request_response"


def test_invalid_provider_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, provider="invalid-provider")


def test_invalid_mode_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, mode="invalid")


def test_legacy_mode_conflicts_with_explicit_provider():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, provider="anthropic", mode="foundry")


def test_tpm_rpm_overrides_inference_rate_limits():
    cfg = _base_config()
    apply_cli_overrides(cfg, tpm=80_000, rpm=300)
    assert cfg.inference.rate_limits is not None
    assert cfg.inference.rate_limits.tokens_per_minute == 80_000
    assert cfg.inference.rate_limits.requests_per_minute == 300


def test_tpm_without_rpm_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="--tpm and --rpm must both"):
        apply_cli_overrides(cfg, tpm=80_000)


def test_single_judge_new_format():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:anthropic:request_response"])
    judge = cfg.judging.judges[0]
    assert judge.name == "gpt-4o"
    assert judge.provider == "anthropic"
    assert judge.mode == "request_response"


def test_single_judge_legacy_batch_format():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:batch"])
    judge = cfg.judging.judges[0]
    assert judge.provider == "azure_openai"
    assert judge.mode == "batch"


def test_judge_with_rate_limits_new_format():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:azure_ai_inference:request_response:80000:300:1024"])
    judge = cfg.judging.judges[0]
    assert judge.provider == "azure_ai_inference"
    assert judge.mode == "request_response"
    assert judge.max_tokens == 1024
    assert judge.rate_limits is not None
    assert judge.rate_limits.tokens_per_minute == 80_000


def test_judge_batch_with_rate_limits_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="not supported for batch mode"):
        apply_cli_overrides(cfg, judges=["gpt-4o:azure_openai:batch:80000:300"])


def test_judges_replace_config():
    cfg = _base_config(
        judging={
            "judges": [
                {
                    "name": "old",
                    "provider": "azure_ai_inference",
                    "mode": "request_response",
                    "model": "old-model",
                }
            ]
        }
    )
    apply_cli_overrides(cfg, judges=["new-model:anthropic:request_response"])
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "new-model"


def test_judge_legacy_foundry_with_rate_limits():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:foundry:80000:300"])
    judge = cfg.judging.judges[0]
    assert judge.provider == "azure_ai_inference"
    assert judge.mode == "request_response"
    assert judge.rate_limits is not None

