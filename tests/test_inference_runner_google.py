"""Tests for inference runner provider dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner
from adele_runner.schemas import DatasetItem, InferenceOutput


@pytest.mark.asyncio
async def test_async_inference_uses_google_adapter(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "google-run", "output_dir": str(tmp_path)},
            "inference": {"mode": "google", "model": "gemini-2.5-flash"},
            "judging": {"enabled": False},
        }
    )
    item = DatasetItem(instance_id="i1", prompt="Hello", ground_truth="Hi")
    result = InferenceOutput(
        instance_id="i1",
        model_id="gemini-2.5-flash",
        prompt="Hello",
        response="Hi",
        run_id="google-run",
    )

    adapter = mocker.Mock()
    adapter.infer = mocker.AsyncMock(return_value=result)

    google_cls = mocker.patch.object(inference_runner, "GoogleGenAIAdapter", return_value=adapter)
    foundry_cls = mocker.patch.object(inference_runner, "FoundryAdapter")

    completed = await inference_runner._run_async_inference(cfg, [item], tmp_path / "outputs.jsonl")

    assert completed == [result]
    google_cls.assert_called_once_with(cfg, rate_limiter=None)
    foundry_cls.assert_not_called()
