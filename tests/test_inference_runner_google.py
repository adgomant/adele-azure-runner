"""Tests for inference runner provider dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner
from adele_runner.runtime.types import (
    ChatResponse,
    ExecutionSettings,
    ResolvedInferencePlan,
    ResolvedInferenceTarget,
    ResolvedModeBinding,
    ResolvedProviderTarget,
)
from adele_runner.schemas import DatasetItem


@pytest.mark.asyncio
async def test_inference_runner_uses_google_request_response_lane(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "google-run", "output_dir": str(tmp_path)},
            "inference": {
                "provider": "google_genai",
                "mode": "request_response",
                "model": "gemini-2.5-flash",
            },
            "judging": {"enabled": False},
        }
    )
    item = DatasetItem(instance_id="i1", prompt="Hello", ground_truth="Hi")
    response = ChatResponse(
        request_id="i1",
        content="Hi",
        prompt_tokens=2,
        completion_tokens=1,
    )

    def _fake_create_adapter(*args, **kwargs):  # noqa: ANN002, ANN003
        return object()

    async def _fake_execute(self, **kwargs):  # noqa: ANN001
        kwargs["on_result"](response)
        return [response]

    mocker.patch.object(
        inference_runner,
        "resolve_inference_plan",
        return_value=ResolvedInferencePlan(
            target=ResolvedInferenceTarget(
                provider_target=ResolvedProviderTarget(
                    provider_kind="google_genai",
                    model="gemini-2.5-flash",
                ),
                requested_mode="request_response",
                prompt_mode="request_response",
                temperature=0.0,
                max_tokens=2048,
                top_p=1.0,
            ),
            binding=ResolvedModeBinding(
                provider_target=ResolvedProviderTarget(
                    provider_kind="google_genai",
                    model="gemini-2.5-flash",
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
    execute = mocker.patch.object(
        inference_runner.RequestResponseExecutor,
        "execute",
        autospec=True,
        side_effect=_fake_execute,
    )
    batch_execute = mocker.patch.object(
        inference_runner.BatchExecutor,
        "execute",
        autospec=True,
    )

    completed = await inference_runner.run_inference(cfg, [item])

    assert len(completed) == 1
    assert completed[0].model_id == "gemini-2.5-flash"
    assert completed[0].response == "Hi"
    assert execute.call_count == 1
    batch_execute.assert_not_called()
