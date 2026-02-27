"""Tests for CLI override logic (--model, --mode, --judge, --judge-template)."""

from __future__ import annotations

import pytest
import typer

from adele_runner.cli import apply_cli_overrides
from adele_runner.config import AppConfig


def _base_config(**overrides) -> AppConfig:
    """Return a minimal AppConfig, optionally with nested overrides."""
    return AppConfig.model_validate(overrides)


def test_no_overrides_leaves_config_unchanged():
    cfg = _base_config()
    original_mode = cfg.inference.mode
    original_model = cfg.inference.foundry.model
    apply_cli_overrides(cfg)
    assert cfg.inference.mode == original_mode
    assert cfg.inference.foundry.model == original_model


def test_mode_batch_sets_internal_mode_and_enabled():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="batch")
    assert cfg.inference.mode == "azure_openai_batch"
    assert cfg.inference.batch.enabled is True


def test_mode_foundry_sets_internal_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="foundry")
    assert cfg.inference.mode == "foundry_async"


def test_mode_auto_sets_internal_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="auto")
    assert cfg.inference.mode == "auto"


def test_invalid_mode_raises():
    cfg = _base_config()
    with pytest.raises(typer.BadParameter):
        apply_cli_overrides(cfg, mode="invalid")


def test_model_with_foundry_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="foundry", models=["my-foundry-model"])
    assert cfg.inference.foundry.model == "my-foundry-model"


def test_model_with_batch_mode():
    cfg = _base_config()
    apply_cli_overrides(cfg, mode="batch", models=["my-batch-deployment"])
    assert cfg.inference.batch.deployment == "my-batch-deployment"


def test_model_without_mode_uses_resolved_default():
    cfg = _base_config()
    # Default mode is "auto" with batch.enabled=False → resolves to foundry_async
    apply_cli_overrides(cfg, models=["some-model"])
    assert cfg.inference.foundry.model == "some-model"


def test_model_without_mode_batch_enabled():
    cfg = _base_config(inference={"mode": "auto", "batch": {"enabled": True}})
    apply_cli_overrides(cfg, models=["some-deployment"])
    assert cfg.inference.batch.deployment == "some-deployment"


def test_multi_model_sets_first():
    """When multiple models given, apply_cli_overrides sets the first one."""
    cfg = _base_config()
    apply_cli_overrides(cfg, models=["model-a", "model-b", "model-c"])
    assert cfg.inference.foundry.model == "model-a"


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
    assert cfg.judging.enabled is True


def test_single_judge_batch():
    cfg = _base_config()
    apply_cli_overrides(cfg, judges=["gpt-4o:batch"])
    assert len(cfg.judging.judges) == 1
    assert cfg.judging.judges[0].name == "gpt-4o"
    assert cfg.judging.judges[0].model == "gpt-4o"
    assert cfg.judging.judges[0].provider == "batch"


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
