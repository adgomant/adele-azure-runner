"""Tests for inference runner provider dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner
from adele_runner.runtime.types import ChatResponse
from adele_runner.schemas import DatasetItem


@pytest.mark.asyncio
async def test_inference_runner_uses_google_request_response_lane(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "google-run", "output_dir": str(tmp_path)},
            "inference": {"mode": "google", "model": "gemini-2.5-flash"},
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

    async def _fake_execute(self, **kwargs):  # noqa: ANN001
        assert kwargs["adapter_kind"] == "google_genai"
        kwargs["on_result"](response)
        return [response]

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

