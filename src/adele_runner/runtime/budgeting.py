"""Runtime budget tracking and enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from adele_runner.config import ModelPricing
from adele_runner.runtime.types import ChatRequest, ChatResponse
from adele_runner.utils.pricing import estimate_batch_request_cost_usd, estimate_cost_usd


@dataclass(frozen=True, slots=True)
class BudgetExceededError(RuntimeError):
    """Raised when a runtime spend budget is exhausted."""

    lane_name: str
    pricing_key: str
    spent_usd: float
    budget_usd: float
    estimated: bool = False

    def __str__(self) -> str:
        kind = "estimated" if self.estimated else "actual"
        return (
            f"Budget exceeded for {self.lane_name} ({self.pricing_key}): "
            f"spent=${self.spent_usd:.6f} budget=${self.budget_usd:.6f} [{kind}]"
        )


class BudgetTracker:
    """Tracks actual and estimated spend for a single execution lane."""

    def __init__(
        self,
        *,
        lane_name: str,
        pricing_key: str,
        budget_usd: float,
        model_pricing: ModelPricing,
    ) -> None:
        self.lane_name = lane_name
        self.pricing_key = pricing_key
        self.budget_usd = budget_usd
        self.model_pricing = model_pricing
        self.spent_usd = 0.0

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd

    def can_submit_batch_chunk(self, requests: list[ChatRequest]) -> None:
        estimated_cost = sum(
            estimate_batch_request_cost_usd(request, self.model_pricing) for request in requests
        )
        if self.spent_usd + estimated_cost > self.budget_usd:
            raise BudgetExceededError(
                lane_name=self.lane_name,
                pricing_key=self.pricing_key,
                spent_usd=self.spent_usd + estimated_cost,
                budget_usd=self.budget_usd,
                estimated=True,
            )

    def record_actual_usage(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        self.spent_usd += estimate_cost_usd(prompt_tokens, completion_tokens, self.model_pricing)
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                lane_name=self.lane_name,
                pricing_key=self.pricing_key,
                spent_usd=self.spent_usd,
                budget_usd=self.budget_usd,
                estimated=False,
            )

    def record_response(self, response: ChatResponse) -> None:
        self.record_actual_usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
