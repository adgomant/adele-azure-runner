"""Tests for config validation."""

from __future__ import annotations

from adele_runner.config import AppConfig


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


# ---------------------------------------------------------------------------
# Foundry endpoint
# ---------------------------------------------------------------------------


def test_empty_foundry_endpoint():
    cfg = _cfg(inference={"foundry": {"endpoint": "", "model": "gpt-4o"}})
    errors = cfg.validate_config()
    assert any("endpoint" in e.lower() for e in errors)


def test_placeholder_foundry_endpoint():
    cfg = _cfg(inference={"foundry": {"endpoint": "https://<YOUR-RESOURCE>.services.ai.azure.com", "model": "m"}})
    errors = cfg.validate_config()
    assert any("endpoint" in e.lower() for e in errors)


def test_valid_foundry_endpoint_no_error():
    cfg = _cfg(inference={"foundry": {"endpoint": "https://myresource.services.ai.azure.com", "model": "m"}})
    errors = cfg.validate_config()
    endpoint_errors = [e for e in errors if "endpoint" in e.lower()]
    assert len(endpoint_errors) == 0


# ---------------------------------------------------------------------------
# Missing model
# ---------------------------------------------------------------------------


def test_missing_foundry_model():
    cfg = _cfg(inference={"foundry": {"endpoint": "https://real.endpoint.com", "model": ""}})
    errors = cfg.validate_config()
    assert any("model" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Batch mode validation
# ---------------------------------------------------------------------------


def test_empty_batch_endpoint():
    cfg = _cfg(inference={"mode": "azure_openai_batch", "batch": {"enabled": True, "azure_openai_endpoint": ""}})
    errors = cfg.validate_config()
    assert any("batch" in e.lower() and "endpoint" in e.lower() for e in errors)


def test_placeholder_batch_endpoint():
    cfg = _cfg(inference={
        "mode": "azure_openai_batch",
        "batch": {
            "enabled": True,
            "azure_openai_endpoint": "https://<YOUR-AOAI-RESOURCE>.openai.azure.com",
        },
    })
    errors = cfg.validate_config()
    assert any("batch" in e.lower() and "endpoint" in e.lower() for e in errors)


def test_missing_batch_deployment():
    cfg = _cfg(inference={
        "mode": "azure_openai_batch",
        "batch": {
            "enabled": True,
            "azure_openai_endpoint": "https://real.openai.azure.com",
            "deployment": "",
        },
    })
    errors = cfg.validate_config()
    assert any("deployment" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Judging validation
# ---------------------------------------------------------------------------


def test_judging_enabled_no_judges():
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://real.com", "model": "m"}},
        judging={"enabled": True, "judges": []},
    )
    errors = cfg.validate_config()
    assert any("judge" in e.lower() for e in errors)


def test_judging_disabled_no_judges_ok():
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://real.com", "model": "m"}},
        judging={"enabled": False, "judges": []},
    )
    errors = cfg.validate_config()
    judge_errors = [e for e in errors if "judge" in e.lower()]
    assert len(judge_errors) == 0


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def test_invalid_prompt_template():
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://real.com", "model": "m"}},
        judging={"prompt_template": "v99"},
    )
    errors = cfg.validate_config()
    assert any("template" in e.lower() for e in errors)


def test_valid_prompt_templates():
    for template in ("v1", "v2"):
        cfg = _cfg(
            inference={"foundry": {"endpoint": "https://real.com", "model": "m"}},
            judging={"prompt_template": template, "enabled": False},
        )
        errors = cfg.validate_config()
        template_errors = [e for e in errors if "template" in e.lower()]
        assert len(template_errors) == 0, f"Unexpected error for template={template}: {template_errors}"


# ---------------------------------------------------------------------------
# Clean config
# ---------------------------------------------------------------------------


def test_fully_valid_config():
    cfg = _cfg(
        inference={"foundry": {"endpoint": "https://myresource.azure.com", "model": "gpt-4o"}},
        judging={"enabled": True, "judges": [{"name": "j", "model": "m"}]},
    )
    errors = cfg.validate_config()
    assert errors == []
