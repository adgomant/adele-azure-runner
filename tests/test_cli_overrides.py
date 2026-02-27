"""Tests for CLI override logic (--model, --mode, --judge, --judge-template, --tpm, --rpm)."""

from __future__ import annotations

import pytest
import typer

from adele_runner.cli import apply_cli_overrides
from adele_runner.config import AppConfig


def _base_config(**overrides) -> AppConfig:
    """Return a minimal AppConfig, optionally with nested overrides."""
    return AppConfig.model_validate(overrides)


# ---------------------------------------------------------------------------
# --mode overrides
# ---------------------------------------------------------------------------


def test_no_overrides_leaves_config_unchanged():
    cfg = _base_config()
    original_mode = cfg.inference.mode
    original_model = cfg.inference.model
    apply_cli_overrides(cfg)
    assert cfg.inference.mode == original_mode
    assert cfg.inference.model == original_model


def test_mode_batch_sets_internal_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="batch")
    assert cfg.inference.mode == "batch"


def test_mode_foundry_sets_internal_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="foundry")
    assert cfg.inference.mode == "foundry"


def test_mode_auto_sets_internal_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="auto")
    assert cfg.inference.mode == "auto"


def test_invalid_mode_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, mode="invalid")


# ---------------------------------------------------------------------------
# --model overrides (single model only)
# ---------------------------------------------------------------------------


def test_model_with_foundry_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="foundry", model="my-foundry-model")
    assert cfg.inference.model == "my-foundry-model"


def test_model_with_batch_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="batch", model="my-batch-deployment")
    assert cfg.inference.model == "my-batch-deployment"


def test_model_without_mode_uses_resolved_default():
    cfg = _base_config()
    # Default mode is "auto" → resolves to foundry
    apply_cli_overrides(cfg, model="some-model")
    assert cfg.inference.model == "some-model"


def test_model_with_batch_mode_set_in_config():
    cfg = _base_config(inference={"mode": "batch"})
    apply_cli_overrides(cfg, model="some-deployment")
    assert cfg.inference.model == "some-deployment"


def test_model_none_preserves_config():
    cfg = _base_config(inference={"model": "from-config"})
    apply_cli_overrides(cfg)
    assert cfg.inference.model == "from-config"


# ---------------------------------------------------------------------------
# --tpm / --rpm overrides
# ---------------------------------------------------------------------------


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


def test_rpm_without_tpm_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="--tpm and --rpm must both"):
        apply_cli_overrides(cfg, rpm=300)


def test_tpm_rpm_not_set_preserves_config():
    cfg = _base_config(
        inference={
            "model": "m",
            "rate_limits": {"tokens_per_minute": 50_000, "requests_per_minute": 200},
        },
    )
    apply_cli_overrides(cfg)
    assert cfg.inference.rate_limits is not None
    assert cfg.inference.rate_limits.tokens_per_minute == 50_000


def test_tpm_rpm_overrides_existing_config():
    cfg = _base_config(
        inference={
            "model": "m",
            "rate_limits": {"tokens_per_minute": 50_000, "requests_per_minute": 200},
        },
    )
    apply_cli_overrides(cfg, tpm=80_000, rpm=300)
    assert cfg.inference.rate_limits.tokens_per_minute == 80_000
    assert cfg.inference.rate_limits.requests_per_minute == 300


# ---------------------------------------------------------------------------
# --judge overrides
# ---------------------------------------------------------------------------


def test_no_judges_leaves_config_unchanged():
    cfg = _base_config(judging={"judges": [{"name": "orig", "model": "m"}]})
    apply_cli_overrides(cfg)
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "orig"


def test_single_judge_foundry():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o"])
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "gpt-4o"
    assert cfg.judging.judges[0].model == "gpt-4o"
    assert cfg.judging.judges[0].provider == "foundry"
    assert cfg.judging.judges[0].rate_limits is None
    assert cfg.judging.enabled is True


def test_single_judge_batch():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:batch"])
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "gpt-4o"
    assert cfg.judging.judges[0].model == "gpt-4o"
    assert cfg.judging.judges[0].provider == "batch"
    assert cfg.judging.judges[0].rate_limits is None


def test_multiple_judges_mixed():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:batch", "claude-3-opus"])
    assert len(cfg.judging.judges) == 2
    assert cfg.judging.judges[0].provider == "batch"
    assert cfg.judging.judges[1].provider == "foundry"


def test_judges_replace_config():
    cfg = _base_config(judging={"judges": [{"name": "old", "model": "old-model"}]})
    apply_cli_overrides(cfg, judges=["new-model"])
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "new-model"


def test_invalid_judge_provider_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, judges=["gpt-4o:invalid"])


# ---------------------------------------------------------------------------
# --judge with rate limits (MODEL:PROVIDER:TPM:RPM)
# ---------------------------------------------------------------------------


def test_judge_foundry_with_rate_limits():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:foundry:80000:300"])
    j = cfg.judging.judges[0]
    assert j.name == "gpt-4o"
    assert j.provider == "foundry"
    assert j.rate_limits is not None
    assert j.rate_limits.tokens_per_minute == 80_000
    assert j.rate_limits.requests_per_minute == 300


def test_judge_batch_with_rate_limits_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="not supported for batch"):
        apply_cli_overrides(cfg, judges=["gpt-4o:batch:80000:300"])


def test_judge_non_numeric_tpm_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="must be integers"):
        apply_cli_overrides(cfg, judges=["gpt-4o:foundry:abc:300"])


def test_judge_non_numeric_rpm_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="must be integers"):
        apply_cli_overrides(cfg, judges=["gpt-4o:foundry:80000:xyz"])


def test_judge_three_parts_raises():
    """Three colon-separated parts is ambiguous — reject it."""
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="Invalid judge format"):
        apply_cli_overrides(cfg, judges=["gpt-4o:foundry:80000"])


def test_judge_five_parts_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter, match="Invalid judge format"):
        apply_cli_overrides(cfg, judges=["a:b:c:d:e"])


def test_multiple_judges_with_mixed_rate_limits():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:foundry:80000:300", "claude:foundry"])
    assert cfg.judging.judges[0].rate_limits is not None
    assert cfg.judging.judges[0].rate_limits.tokens_per_minute == 80_000
    assert cfg.judging.judges[1].rate_limits is None


# ---------------------------------------------------------------------------
# --judge-template overrides
# ---------------------------------------------------------------------------


def test_judge_template_override_v2():
    cfg = _base_config()
    apply_cli_overrides(cfg, judge_template="v2")
    assert cfg.judging.prompt_template == "v2"


def test_judge_template_override_v1():
    cfg = _base_config(judging={"prompt_template": "v2"})
    apply_cli_overrides(cfg, judge_template="v1")
    assert cfg.judging.prompt_template == "v1"


def test_judge_template_not_set_preserves_config():
    cfg = _base_config(judging={"prompt_template": "v2"})
    apply_cli_overrides(cfg)
    assert cfg.judging.prompt_template == "v2"


def test_combined_judge_and_template():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o"], judge_template="v2")
    assert cfg.judging.judges[0].name == "gpt-4o"
    assert cfg.judging.prompt_template == "v2"
    assert cfg.judging.enabled is True
