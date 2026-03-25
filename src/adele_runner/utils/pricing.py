"""Shared pricing helpers for metrics and runtime budget tracking."""

from __future__ import annotations

from math import ceil

from adele_runner.config import ModelPricing
from adele_runner.runtime.types import ChatRequest


def estimate_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    model_pricing: ModelPricing,
) -> float:
    """Return USD cost for a token usage record."""
    prompt = max(0, prompt_tokens or 0)
    completion = max(0, completion_tokens or 0)
    return (
        prompt / 1000.0 * model_pricing.prompt_per_1k
        + completion / 1000.0 * model_pricing.completion_per_1k
    )


def estimate_prompt_tokens_from_request(request: ChatRequest) -> int:
    """Conservatively estimate prompt tokens from message text without a tokenizer."""
    total_chars = sum(len(message.content) for message in request.messages)
    return max(1, ceil(total_chars / 3))


def estimate_batch_request_cost_usd(
    request: ChatRequest,
    model_pricing: ModelPricing,
) -> float:
    """Conservative upper-bound batch cost estimate for budget admission."""
    prompt_tokens = estimate_prompt_tokens_from_request(request)
    completion_tokens = max(0, request.max_tokens or 0)
    return estimate_cost_usd(prompt_tokens, completion_tokens, model_pricing)
