"""Tests for config validation."""

from __future__ import annotations

from adele_runner.config import AppConfig


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


# ---------------------------------------------------------------------------
# Foundry endpoint
# ---------------------------------------------------------------------------


def test_empty_foundry_endpoint():
    cfg = _cfg(
        azure={"foundry": {"endpoint": ""}},
        inference={"model": "gpt-4o"},
    )
    errors = cfg.validate_config()
    assert any("endpoint" in e.lower() for e in errors)


def test_placeholder_foundry_endpoint():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://<YOUR-RESOURCE>.services.ai.azure.com"}},
        inference={"model": "m"},
    )
    errors = cfg.validate_config()
    assert any("endpoint" in e.lower() for e in errors)


def test_valid_foundry_endpoint_no_error():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://myresource.services.ai.azure.com"}},
        inference={"model": "m"},
    )
    errors = cfg.validate_config()
    endpoint_errors = [e for e in errors if "endpoint" in e.lower()]
    assert len(endpoint_errors) == 0


# ---------------------------------------------------------------------------
# Missing model
# ---------------------------------------------------------------------------


def test_missing_model():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://real.endpoint.com"}},
        inference={"model": ""},
    )
    errors = cfg.validate_config()
    assert any("model" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Batch mode validation
# ---------------------------------------------------------------------------


def test_empty_batch_endpoint():
    cfg = _cfg(
        azure={"batch": {"endpoint": ""}},
        inference={"mode": "batch"},
    )
    errors = cfg.validate_config()
    assert any("batch" in e.lower() and "endpoint" in e.lower() for e in errors)


def test_placeholder_batch_endpoint():
    cfg = _cfg(
        azure={"batch": {"endpoint": "https://<YOUR-AOAI-RESOURCE>.openai.azure.com"}},
        inference={"mode": "batch"},
    )
    errors = cfg.validate_config()
    assert any("batch" in e.lower() and "endpoint" in e.lower() for e in errors)


def test_missing_batch_model():
    cfg = _cfg(
        azure={"batch": {"endpoint": "https://real.openai.azure.com"}},
        inference={"mode": "batch", "model": ""},
    )
    errors = cfg.validate_config()
    assert any("model" in e.lower() for e in errors)


def test_google_mode_does_not_require_azure_endpoint():
    cfg = _cfg(
        inference={"mode": "google", "model": "gemini-2.5-flash"},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    endpoint_errors = [e for e in errors if "endpoint" in e.lower()]
    assert len(endpoint_errors) == 0


def test_google_mode_requires_model():
    cfg = _cfg(
        inference={"mode": "google", "model": ""},
        judging={"enabled": False},
    )
    errors = cfg.validate_config()
    assert any("model" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Judging validation
# ---------------------------------------------------------------------------


def test_judging_enabled_no_judges():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://real.com"}},
        inference={"model": "m"},
        judging={"enabled": True, "judges": []},
    )
    errors = cfg.validate_config()
    assert any("judge" in e.lower() for e in errors)


def test_judging_disabled_no_judges_ok():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://real.com"}},
        inference={"model": "m"},
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
        azure={"foundry": {"endpoint": "https://real.com"}},
        inference={"model": "m"},
        judging={"prompt_template": "v99"},
    )
    errors = cfg.validate_config()
    assert any("template" in e.lower() for e in errors)


def test_valid_prompt_templates():
    for template in ("v1", "v2"):
        cfg = _cfg(
            azure={"foundry": {"endpoint": "https://real.com"}},
            inference={"model": "m"},
            judging={"prompt_template": template, "enabled": False},
        )
        errors = cfg.validate_config()
        template_errors = [e for e in errors if "template" in e.lower()]
        assert len(template_errors) == 0, (
            f"Unexpected error for template={template}: {template_errors}"
        )


# ---------------------------------------------------------------------------
# Clean config
# ---------------------------------------------------------------------------


def test_fully_valid_config():
    cfg = _cfg(
        azure={"foundry": {"endpoint": "https://myresource.azure.com"}},
        inference={"model": "gpt-4o"},
        judging={"enabled": True, "judges": [{"name": "j", "model": "m"}]},
    )
    errors = cfg.validate_config()
    assert errors == []


# ---------------------------------------------------------------------------
# Rate limits config
# ---------------------------------------------------------------------------


def test_rate_limits_parsed_from_config():
    cfg = _cfg(
        inference={
            "model": "m",
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
    )
    rl = cfg.inference.rate_limits
    assert rl is not None
    assert rl.tokens_per_minute == 80_000
    assert rl.requests_per_minute == 300


def test_rate_limits_none_by_default():
    cfg = _cfg()
    assert cfg.inference.rate_limits is None


def test_rate_limits_overrides_concurrency():
    cfg = _cfg(
        inference={
            "model": "m",
            "max_tokens": 2048,
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
        concurrency={"max_in_flight": 999},
    )
    cfg.apply_rate_limit_overrides(cfg.inference.rate_limits, cfg.inference.max_tokens)
    # Should override the explicit max_in_flight=999
    assert cfg.concurrency.max_in_flight != 999
    assert cfg.concurrency.max_in_flight >= 1


# ---------------------------------------------------------------------------
# Per-judge rate limits
# ---------------------------------------------------------------------------


def test_judge_rate_limits_parsed():
    cfg = _cfg(
        judging={
            "judges": [
                {
                    "name": "j1",
                    "model": "m1",
                    "rate_limits": {"tokens_per_minute": 50_000, "requests_per_minute": 200},
                }
            ]
        },
    )
    assert cfg.judging.judges[0].rate_limits is not None
    assert cfg.judging.judges[0].rate_limits.tokens_per_minute == 50_000


def test_judge_rate_limits_none_by_default():
    cfg = _cfg(
        judging={"judges": [{"name": "j1", "model": "m1"}]},
    )
    assert cfg.judging.judges[0].rate_limits is None


def test_most_restrictive_judge_rate_limits():
    cfg = _cfg(
        judging={
            "judges": [
                {
                    "name": "j1",
                    "provider": "foundry",
                    "model": "m1",
                    "rate_limits": {"tokens_per_minute": 100_000, "requests_per_minute": 500},
                },
                {
                    "name": "j2",
                    "provider": "foundry",
                    "model": "m2",
                    "rate_limits": {"tokens_per_minute": 30_000, "requests_per_minute": 100},
                },
            ]
        },
    )
    rl = cfg.get_most_restrictive_judge_rate_limits()
    assert rl is not None
    assert rl.tokens_per_minute == 30_000


def test_most_restrictive_judge_rate_limits_ignores_batch():
    cfg = _cfg(
        judging={
            "judges": [
                {
                    "name": "j1",
                    "provider": "batch",
                    "model": "m1",
                    "rate_limits": {"tokens_per_minute": 10_000, "requests_per_minute": 50},
                },
                {
                    "name": "j2",
                    "provider": "foundry",
                    "model": "m2",
                    "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
                },
            ]
        },
    )
    rl = cfg.get_most_restrictive_judge_rate_limits()
    assert rl is not None
    # Batch judge should be ignored
    assert rl.tokens_per_minute == 80_000


def test_most_restrictive_judge_rate_limits_none_when_no_foundry():
    cfg = _cfg(
        judging={
            "judges": [
                {"name": "j1", "provider": "batch", "model": "m1"},
            ]
        },
    )
    assert cfg.get_most_restrictive_judge_rate_limits() is None


# ---------------------------------------------------------------------------
# Batch splitting limits
# ---------------------------------------------------------------------------


def test_batch_max_requests_exceeds_azure_limit():
    cfg = _cfg(azure={"batch": {"max_requests_per_file": 200_000}})
    errors = cfg.validate_config()
    assert any("max_requests_per_file" in e and "exceeds" in e for e in errors)


def test_batch_max_bytes_exceeds_azure_limit():
    cfg = _cfg(azure={"batch": {"max_bytes_per_file": 300_000_000}})
    errors = cfg.validate_config()
    assert any("max_bytes_per_file" in e and "exceeds" in e for e in errors)


def test_batch_max_requests_below_one():
    cfg = _cfg(azure={"batch": {"max_requests_per_file": 0}})
    errors = cfg.validate_config()
    assert any("max_requests_per_file" in e and "at least 1" in e for e in errors)


def test_batch_max_bytes_below_one():
    cfg = _cfg(azure={"batch": {"max_bytes_per_file": 0}})
    errors = cfg.validate_config()
    assert any("max_bytes_per_file" in e and "at least 1" in e for e in errors)


def test_batch_limits_at_azure_max_ok():
    cfg = _cfg(
        azure={
            "foundry": {"endpoint": "https://real.com"},
            "batch": {"max_requests_per_file": 100_000, "max_bytes_per_file": 200_000_000},
        },
        inference={"model": "m"},
    )
    errors = cfg.validate_config()
    batch_errors = [e for e in errors if "max_requests_per_file" in e or "max_bytes_per_file" in e]
    assert len(batch_errors) == 0


def test_batch_limits_default_values():
    cfg = _cfg()
    assert cfg.azure.batch.max_requests_per_file == 50_000
    assert cfg.azure.batch.max_bytes_per_file == 100_000_000
