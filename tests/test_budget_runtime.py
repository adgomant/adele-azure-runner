"""Tests for runtime budget enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from adele_runner import cli
from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner
from adele_runner.runtime.budgeting import BudgetExceededError, BudgetTracker
from adele_runner.runtime.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ExecutionSettings,
    ResolvedInferencePlan,
    ResolvedInferenceTarget,
    ResolvedModeBinding,
    ResolvedProviderTarget,
)
from adele_runner.schemas import DatasetItem


@pytest.mark.asyncio
async def test_inference_budget_stop_writes_output_and_finalizes_manifest(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "budget-run", "output_dir": str(tmp_path)},
            "providers": {"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
            "inference": {
                "provider": "azure_ai_inference",
                "mode": "request_response",
                "model": "gpt-4o",
                "budget_usd": 5.0,
            },
            "pricing": {
                "enabled": True,
                "models": {"gpt-4o": {"prompt_per_1k": 2.5, "completion_per_1k": 10.0}},
            },
            "judging": {"enabled": False},
        }
    )
    item = DatasetItem(instance_id="i1", prompt="Hello", ground_truth="Hi")
    response1 = ChatResponse(
        request_id="i1",
        content="first",
        prompt_tokens=1_000,
        completion_tokens=500,
    )
    response2 = ChatResponse(
        request_id="i2",
        content="second",
        prompt_tokens=100,
        completion_tokens=50,
    )

    def _fake_create_adapter(*args, **kwargs):  # noqa: ANN002, ANN003
        return object()

    async def _fake_execute(self, **kwargs):  # noqa: ANN001
        kwargs["on_result"](response1)
        kwargs["on_result"](response2)
        return [response1, response2]

    mocker.patch.object(
        inference_runner,
        "resolve_inference_plan",
        return_value=ResolvedInferencePlan(
            target=ResolvedInferenceTarget(
                provider_target=ResolvedProviderTarget(
                    provider_kind="azure_ai_inference",
                    model="gpt-4o",
                ),
                requested_mode="request_response",
                prompt_mode="request_response",
                temperature=0.0,
                max_tokens=2048,
                top_p=1.0,
            ),
            binding=ResolvedModeBinding(
                provider_target=ResolvedProviderTarget(
                    provider_kind="azure_ai_inference",
                    model="gpt-4o",
                ),
                execution_kind="request_response",
                create_adapter=_fake_create_adapter,
            ),
            settings=ExecutionSettings(
                max_in_flight=2,
                request_timeout_s=30.0,
                max_retries=1,
                backoff_base_s=0.5,
                backoff_max_s=5.0,
                max_poll_time_s=60.0,
                batch_completion_window="24h",
            ),
        ),
    )
    mocker.patch.object(
        inference_runner.RequestResponseExecutor,
        "execute",
        autospec=True,
        side_effect=_fake_execute,
    )

    with pytest.raises(BudgetExceededError):
        await inference_runner.run_inference(cfg, [item])

    outputs = (tmp_path / "budget-run" / "outputs.jsonl").read_text().strip().splitlines()
    assert len(outputs) == 1
    manifest = json.loads((tmp_path / "budget-run" / "run_manifest.json").read_text())
    assert manifest["completed_instances"] == 1
    assert manifest["end_time"] is not None


def test_cli_exits_with_budget_code_on_budget_exhaustion(mocker, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  azure_ai_inference:
    endpoint: "https://real.ai.azure.com"
inference:
  provider: "azure_ai_inference"
  mode: "request_response"
  model: "gpt-4o"
  budget_usd: 5.0
judging:
  enabled: false
pricing:
  enabled: true
  models:
    gpt-4o:
      prompt_per_1k: 2.5
      completion_per_1k: 10.0
""".strip(),
        encoding="utf-8",
    )

    mocker.patch("adele_runner.datasets.adele.load_adele", return_value=[DatasetItem(instance_id="i1", prompt="x", ground_truth="y")])
    mocker.patch(
        "adele_runner.pipeline.inference_runner.run_inference",
        side_effect=BudgetExceededError(
            lane_name="inference:gpt-4o",
            pricing_key="gpt-4o",
            spent_usd=5.1,
            budget_usd=5.0,
            estimated=False,
        ),
    )

    with pytest.raises(typer.Exit) as exc:
        cli.run_inference(
            config=config_path,
            provider=None,
            model=None,
            mode=None,
            tpm=None,
            rpm=None,
            force_run=False,
            dry_run=False,
        )
    assert exc.value.exit_code == 3


def test_budget_tracker_rejects_batch_chunk_before_submit():
    pricing = AppConfig.model_validate(
        {"pricing": {"enabled": True, "models": {"gpt-4o": {"prompt_per_1k": 2.5, "completion_per_1k": 10.0}}}}
    ).pricing.models["gpt-4o"]
    tracker = BudgetTracker(
        lane_name="inference:gpt-4o",
        pricing_key="gpt-4o",
        budget_usd=1.0,
        model_pricing=pricing,
    )
    request = ChatRequest(
        request_id="r1",
        model="gpt-4o",
        messages=(ChatMessage(role="user", content="x" * 3_000),),
        max_tokens=1_000,
    )
    with pytest.raises(BudgetExceededError):
        tracker.can_submit_batch_chunk([request])
