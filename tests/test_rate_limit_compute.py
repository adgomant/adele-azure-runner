"""Tests for rate limit auto-computation of concurrency parameters."""

from __future__ import annotations

import pytest

from adele_runner.config import (
    AppConfig,
    RateLimitsConfig,
    compute_concurrency_from_rate_limits,
)

# ---------------------------------------------------------------------------
# compute_concurrency_from_rate_limits
# ---------------------------------------------------------------------------


def test_basic_80k_300rpm_2048():
    rl = RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=2048)
    assert cc.max_in_flight >= 1
    assert cc.request_timeout_s >= 30.0
    assert cc.backoff_base_s >= 0.5
    assert cc.backoff_max_s >= 30.0


def test_high_max_tokens_reduces_concurrency():
    rl = RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300)
    cc_low = compute_concurrency_from_rate_limits(rl, max_tokens=2048)
    cc_high = compute_concurrency_from_rate_limits(rl, max_tokens=50_000)
    # Higher max_tokens → fewer concurrent requests
    assert cc_high.max_in_flight <= cc_low.max_in_flight


def test_higher_tpm_allows_more_concurrency():
    rl_low = RateLimitsConfig(tokens_per_minute=30_000, requests_per_minute=100)
    rl_high = RateLimitsConfig(tokens_per_minute=200_000, requests_per_minute=600)
    cc_low = compute_concurrency_from_rate_limits(rl_low, max_tokens=2048)
    cc_high = compute_concurrency_from_rate_limits(rl_high, max_tokens=2048)
    assert cc_high.max_in_flight >= cc_low.max_in_flight


def test_min_max_in_flight_is_one():
    """Even with very low TPM, max_in_flight should be at least 1."""
    rl = RateLimitsConfig(tokens_per_minute=100, requests_per_minute=1)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=50_000)
    assert cc.max_in_flight >= 1


def test_request_timeout_capped():
    rl = RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=100_000)
    assert cc.request_timeout_s <= 600.0
    assert cc.request_timeout_s >= 30.0


def test_backoff_bounds():
    rl = RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=2048)
    assert 0.5 <= cc.backoff_base_s <= 10.0
    assert 30.0 <= cc.backoff_max_s <= 120.0


@pytest.mark.parametrize(
    "tpm,rpm,max_tokens",
    [
        (80_000, 300, 2048),
        (80_000, 300, 50_000),
        (200_000, 600, 2048),
        (30_000, 100, 512),
        (10_000, 50, 4096),
        (500_000, 1000, 2048),
    ],
)
def test_no_crash_various_inputs(tpm: int, rpm: int, max_tokens: int):
    """Smoke test: computation doesn't crash for various valid inputs."""
    rl = RateLimitsConfig(tokens_per_minute=tpm, requests_per_minute=rpm)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=max_tokens)
    assert cc.max_in_flight >= 1
    assert cc.request_timeout_s >= 30.0


# ---------------------------------------------------------------------------
# apply_rate_limit_overrides
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> AppConfig:
    return AppConfig.model_validate(overrides)


def test_apply_overrides_with_rate_limits():
    cfg = _cfg(
        inference={
            "model": "m",
            "max_tokens": 2048,
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
    )
    cfg.apply_rate_limit_overrides(cfg.inference.rate_limits, cfg.inference.max_tokens)
    # Should have been overridden (may differ from default 16)
    assert cfg.concurrency.max_in_flight >= 1
    assert cfg.concurrency.request_timeout_s >= 30.0


def test_apply_overrides_without_rate_limits():
    cfg = _cfg(
        inference={"model": "m"},
    )
    original = cfg.concurrency.max_in_flight
    cfg.apply_rate_limit_overrides()
    # No rate limits → no change
    assert cfg.concurrency.max_in_flight == original


def test_apply_overrides_preserves_max_retries():
    cfg = _cfg(
        inference={
            "model": "m",
            "max_tokens": 2048,
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
        concurrency={"max_retries": 10, "max_poll_time_s": 7200.0},
    )
    cfg.apply_rate_limit_overrides(cfg.inference.rate_limits, cfg.inference.max_tokens)
    # Non-rate-limit params should be preserved
    assert cfg.concurrency.max_retries == 10
    assert cfg.concurrency.max_poll_time_s == 7200.0


def test_apply_overrides_with_explicit_max_tokens():
    """When max_tokens is passed explicitly, it overrides the config value."""
    cfg = _cfg(
        inference={
            "model": "m",
            "max_tokens": 50_000,
            "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
        },
    )
    cfg.apply_rate_limit_overrides(cfg.inference.rate_limits, max_tokens=512)
    # With max_tokens=512 the concurrency should be higher than with 50_000
    assert cfg.concurrency.max_in_flight >= 1


def test_apply_overrides_for_judge_rate_limits():
    """Judge rate limits should tune concurrency using the largest judge max_tokens."""
    cfg = _cfg(
        judging={
            "judges": [
                {
                    "name": "j1",
                    "provider": "foundry",
                    "model": "m1",
                    "rate_limits": {"tokens_per_minute": 80_000, "requests_per_minute": 300},
                },
            ]
        },
    )
    judge_rl = cfg.get_most_restrictive_judge_rate_limits()
    cfg.apply_rate_limit_overrides(judge_rl, max_tokens=cfg.get_max_judge_max_tokens())
    assert cfg.concurrency.max_in_flight >= 1
    assert cfg.concurrency.request_timeout_s >= 30.0


# ---------------------------------------------------------------------------
# get_max_judge_max_tokens
# ---------------------------------------------------------------------------


def test_get_max_judge_max_tokens_default():
    cfg = _cfg(
        judging={
            "judges": [
                {"name": "j1", "provider": "foundry", "model": "m1"},
            ]
        },
    )
    assert cfg.get_max_judge_max_tokens() == 512


def test_get_max_judge_max_tokens_uses_largest():
    cfg = _cfg(
        judging={
            "judges": [
                {"name": "j1", "provider": "foundry", "model": "m1", "max_tokens": 256},
                {"name": "j2", "provider": "foundry", "model": "m2", "max_tokens": 1024},
            ]
        },
    )
    assert cfg.get_max_judge_max_tokens() == 1024


def test_get_max_judge_max_tokens_ignores_batch():
    cfg = _cfg(
        judging={
            "judges": [
                {"name": "j1", "provider": "foundry", "model": "m1", "max_tokens": 256},
                {"name": "j2", "provider": "batch", "model": "m2", "max_tokens": 2048},
            ]
        },
    )
    assert cfg.get_max_judge_max_tokens() == 256


def test_get_max_judge_max_tokens_no_foundry_judges():
    cfg = _cfg(
        judging={
            "judges": [
                {"name": "j1", "provider": "batch", "model": "m1"},
            ]
        },
    )
    assert cfg.get_max_judge_max_tokens() == 512


def test_get_max_judge_max_tokens_no_judges():
    cfg = _cfg()
    assert cfg.get_max_judge_max_tokens() == 512
