from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner, judge_runner
from adele_runner.runtime.executors import RequestResponseExecutor
from adele_runner.schemas import DatasetItem, InferenceOutput, JudgeOutput
from adele_runner.utils.io import read_jsonl


@pytest.mark.asyncio
async def test_inference_request_exception_persists_failed_row(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "test", "output_dir": str(tmp_path)},
            "providers": {"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
            "inference": {"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
            "judging": {"enabled": False},
        }
    )
    item = DatasetItem(instance_id="i1", prompt="Q", ground_truth="A")

    async def _fail_execute(self, **kwargs):  # noqa: ANN001
        exc = RuntimeError("boom")
        exc.request_id = "i1"  # type: ignore[attr-defined]
        kwargs["on_result"](exc)
        return []

    mocker.patch.object(RequestResponseExecutor, "execute", _fail_execute)

    await inference_runner.run_inference(cfg, [item])
    outputs = read_jsonl(cfg.outputs_path(), InferenceOutput)
    assert len(outputs) == 1
    assert outputs[0].status == "failed"
    assert outputs[0].response is None


@pytest.mark.asyncio
async def test_failed_inference_generates_skipped_judge_row(tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "test", "output_dir": str(tmp_path)},
            "providers": {"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
            "judging": {
                "enabled": True,
                "judges": [
                    {"name": "judge-a", "provider": "azure_ai_inference", "mode": "request_response", "model": "m1"}
                ],
            },
        }
    )
    inference_output = InferenceOutput(
        instance_id="i1",
        model_id="model-a",
        prompt="Q",
        response=None,
        status="failed",
        error_message="boom",
    )

    outputs = await judge_runner.run_judge(cfg, [inference_output], {"i1": "A"})
    assert len(outputs) == 1
    assert outputs[0].status == "skipped"
    judge_outputs = read_jsonl(cfg.judge_outputs_path(), JudgeOutput)
    assert judge_outputs[0].score is None
