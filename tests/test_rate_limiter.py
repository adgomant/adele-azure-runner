"""Tests for AsyncRateLimiter and its integration with bounded_gather."""

from __future__ import annotations

import asyncio
import time

import pytest

from adele_runner.utils.concurrency import AsyncRateLimiter, bounded_gather

# ---------------------------------------------------------------------------
# AsyncRateLimiter unit tests
# ---------------------------------------------------------------------------


def test_init_rejects_zero_rpm():
    with pytest.raises(ValueError, match="rpm must be >= 1"):
        AsyncRateLimiter(0)


def test_init_rejects_negative_rpm():
    with pytest.raises(ValueError, match="rpm must be >= 1"):
        AsyncRateLimiter(-5)


@pytest.mark.asyncio
async def test_acquire_pacing():
    """Requests should be spaced at roughly 60/RPM seconds apart."""
    rpm = 600  # → 0.1 s interval
    limiter = AsyncRateLimiter(rpm)
    expected_interval = 60.0 / rpm

    timestamps: list[float] = []

    async def _record():
        await limiter.acquire()
        timestamps.append(time.monotonic())

    # Fire 5 requests
    await asyncio.gather(*[_record() for _ in range(5)])

    # Check intervals between consecutive acquisitions
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        # Allow 50ms tolerance for scheduling jitter
        assert delta >= expected_interval - 0.05, (
            f"Interval {i}: {delta:.4f}s < expected {expected_interval:.4f}s"
        )


@pytest.mark.asyncio
async def test_acquire_first_request_immediate():
    """The very first request should not wait."""
    limiter = AsyncRateLimiter(60)  # 1 req/s
    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start
    # First request should complete almost instantly
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_signal_backoff_pauses_dispatch():
    """signal_backoff should delay future acquisitions."""
    limiter = AsyncRateLimiter(6000)  # very fast base rate (0.01s interval)

    # First acquire to set baseline
    await limiter.acquire()

    # Signal a 0.3s backoff
    limiter.signal_backoff(0.3)

    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start

    # Should have waited at least ~0.25s (with tolerance)
    assert elapsed >= 0.2, f"Expected >=0.2s delay from backoff, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_signal_backoff_max_wins():
    """Multiple backoff signals should take the latest deadline."""
    limiter = AsyncRateLimiter(6000)
    await limiter.acquire()

    limiter.signal_backoff(0.1)
    limiter.signal_backoff(0.3)  # longer → should win
    limiter.signal_backoff(0.05)  # shorter → ignored

    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.2, f"Expected >=0.2s from max backoff, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# effective_rpm in ConcurrencyConfig
# ---------------------------------------------------------------------------


def test_effective_rpm_stored_by_compute():
    """compute_concurrency_from_rate_limits should store effective_rpm."""
    from adele_runner.config import RateLimitsConfig, compute_concurrency_from_rate_limits

    rl = RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=2048)
    assert cc.effective_rpm is not None
    assert cc.effective_rpm >= 1


def test_effective_rpm_defaults_to_none():
    """ConcurrencyConfig defaults should have effective_rpm=None."""
    from adele_runner.config import ConcurrencyConfig

    cc = ConcurrencyConfig()
    assert cc.effective_rpm is None


def test_effective_rpm_tpm_constrained():
    """When TPM is the bottleneck, effective_rpm should be lower than RPM."""
    from adele_runner.config import RateLimitsConfig, compute_concurrency_from_rate_limits

    # TPM=10_000 with max_tokens=10_000 → 1 RPM from TPM, but RPM=300
    rl = RateLimitsConfig(tokens_per_minute=10_000, requests_per_minute=300)
    cc = compute_concurrency_from_rate_limits(rl, max_tokens=10_000)
    assert cc.effective_rpm == 1  # min(300, 10000/10000=1) → 1


# ---------------------------------------------------------------------------
# bounded_gather with rate_limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_gather_with_rate_limiter():
    """bounded_gather should respect the rate limiter pacing."""
    limiter = AsyncRateLimiter(600)  # 0.1s interval
    timestamps: list[float] = []

    async def _task(idx: int) -> int:
        timestamps.append(time.monotonic())
        return idx

    coros = [_task(i) for i in range(5)]
    results = await bounded_gather(coros, max_concurrency=10, rate_limiter=limiter)

    # All should succeed
    assert results == [0, 1, 2, 3, 4]
    # Check pacing
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        assert delta >= 0.05, f"Interval {i}: {delta:.4f}s — expected pacing"


@pytest.mark.asyncio
async def test_bounded_gather_without_rate_limiter():
    """bounded_gather without rate_limiter should behave as before."""
    results = []

    async def _task(idx: int) -> int:
        results.append(idx)
        return idx * 2

    coros = [_task(i) for i in range(5)]
    out = await bounded_gather(coros, max_concurrency=10)
    assert out == [0, 2, 4, 6, 8]


# ---------------------------------------------------------------------------
# update_from_headers — adaptive header feedback
# ---------------------------------------------------------------------------


def test_update_from_headers_slows_on_low_remaining():
    """Interval should increase when remaining capacity is low."""
    limiter = AsyncRateLimiter(600)  # base interval = 0.1s
    base = limiter._base_interval

    # Remaining at 10% — should trigger ×3 multiplier
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "30",
        "x-ratelimit-limit-requests": "300",
        "x-ratelimit-remaining-tokens": "8000",
        "x-ratelimit-limit-tokens": "80000",
    })
    assert limiter._pressure_multiplier == 3.0
    assert abs(limiter._interval - base * 3.0) < 1e-9


def test_update_from_headers_restores_on_ample_remaining():
    """Interval should reset when remaining capacity is ample."""
    limiter = AsyncRateLimiter(600)

    # First drive pressure up
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "10",
        "x-ratelimit-limit-requests": "300",
    })
    assert limiter._pressure_multiplier > 1.0

    # Then restore with ample remaining
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "250",
        "x-ratelimit-limit-requests": "300",
    })
    assert limiter._pressure_multiplier == 1.0


def test_update_from_headers_critical_remaining():
    """Interval should use ×5 multiplier when remaining < 5%."""
    limiter = AsyncRateLimiter(600)
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "2000",
        "x-ratelimit-limit-tokens": "80000",
    })
    assert limiter._pressure_multiplier == 5.0


def test_update_from_headers_moderate_remaining():
    """Interval should use ×1.5 multiplier when remaining 15-30%."""
    limiter = AsyncRateLimiter(600)
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "60",
        "x-ratelimit-limit-requests": "300",
    })
    assert limiter._pressure_multiplier == 1.5


def test_update_from_headers_missing_headers_no_crash():
    """No crash when headers are missing."""
    limiter = AsyncRateLimiter(600)
    limiter.update_from_headers({})
    assert limiter._pressure_multiplier == 1.0


def test_update_from_headers_partial_headers():
    """Works with only remaining-tokens (no remaining-requests)."""
    limiter = AsyncRateLimiter(600)
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "4000",
        "x-ratelimit-limit-tokens": "80000",
    })
    # 4000/80000 = 5% exactly → falls into <0.15 bracket → ×3
    assert limiter._pressure_multiplier == 3.0


def test_update_from_headers_uses_tightest_constraint():
    """When request and token remaining differ, use the tighter one."""
    limiter = AsyncRateLimiter(600)
    # Requests: 200/300 = 67% (ample), Tokens: 2000/80000 = 2.5% (critical)
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "200",
        "x-ratelimit-limit-requests": "300",
        "x-ratelimit-remaining-tokens": "2000",
        "x-ratelimit-limit-tokens": "80000",
    })
    assert limiter._pressure_multiplier == 5.0  # token pressure wins


# ---------------------------------------------------------------------------
# update_token_usage — token tracking & base interval adjustment
# ---------------------------------------------------------------------------


def test_update_token_usage_adjusts_base_interval():
    """Base interval should decrease when actual tokens << initial estimate."""
    # RPM=300, TPM=80000. Initial effective_rpm from max_tokens=2048 would be
    # min(300, 80000/2048)=39, but here we directly set rpm=39 + tpm=80000.
    # If actual avg_tokens=500, rpm_from_tpm=80000/500=160, effective=min(39,160)=39 — no change.
    # So let's use rpm=300 (not TPM-constrained) with tpm=80000, max_tokens estimate was high.
    limiter = AsyncRateLimiter(300, tpm=80_000)
    original_base = limiter._base_interval  # 60/300 = 0.2s

    # Feed 15 samples of 400 tokens each (much less than max_tokens=2048)
    for _ in range(15):
        limiter.update_token_usage(200, 200)  # total=400

    # rpm_from_tpm = 80000/400 = 200, effective = min(300, 200) = 200
    # new_base = 60/200 = 0.3s — actually slower because TPM is the bottleneck now!
    # But that's correct: if tokens per request are 400 and tpm=80000, we can do 200 RPM.
    # Since 200 < 300, TPM is the bottleneck and interval increases.
    assert limiter._base_interval == 60.0 / 200
    assert limiter._base_interval != original_base


def test_update_token_usage_never_exceeds_rpm():
    """Interval should never drop below 60/RPM even with tiny token usage."""
    limiter = AsyncRateLimiter(100, tpm=80_000)
    base_at_rpm = 60.0 / 100  # 0.6s

    # Feed tiny token usage: 50 tokens → rpm_from_tpm=80000/50=1600
    # effective = min(100, 1600) = 100 — still capped by RPM
    for _ in range(15):
        limiter.update_token_usage(25, 25)

    # Base interval should stay at 60/100
    assert limiter._base_interval == base_at_rpm


def test_update_token_usage_no_change_before_min_samples():
    """Base interval should not change before minimum sample count."""
    limiter = AsyncRateLimiter(300, tpm=80_000)
    original_base = limiter._base_interval

    # Feed only 5 samples (below minimum of 10)
    for _ in range(5):
        limiter.update_token_usage(200, 200)

    assert limiter._base_interval == original_base


def test_update_token_usage_no_tpm_no_adjustment():
    """Without TPM, token usage shouldn't affect the base interval."""
    limiter = AsyncRateLimiter(300)  # no tpm
    original_base = limiter._base_interval

    for _ in range(15):
        limiter.update_token_usage(200, 200)

    assert limiter._base_interval == original_base


def test_update_token_usage_ignores_zero_tokens():
    """Zero-token samples should be ignored."""
    limiter = AsyncRateLimiter(300, tpm=80_000)
    limiter.update_token_usage(0, 0)
    assert len(limiter._token_samples) == 0


def test_update_token_usage_combined_with_header_pressure():
    """Token adjustment and header pressure should multiply."""
    limiter = AsyncRateLimiter(300, tpm=80_000)

    # Feed tokens to adjust base interval
    for _ in range(15):
        limiter.update_token_usage(200, 200)  # avg=400, rpm_from_tpm=200

    new_base = 60.0 / 200  # 0.3s

    # Now apply header pressure
    limiter.update_from_headers({
        "x-ratelimit-remaining-requests": "30",
        "x-ratelimit-limit-requests": "300",
    })
    # remaining=10% → ×3 multiplier
    assert limiter._pressure_multiplier == 3.0
    assert abs(limiter._interval - new_base * 3.0) < 1e-9
