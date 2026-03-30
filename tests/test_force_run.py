from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import inference_runner, judge_runner
from adele_runner.runtime.executors import RequestResponseExecutor
from adele_runner.schemas import DatasetItem, InferenceOutput, JudgeOutput
from adele_runner.utils.io import append_jsonl


@pytest.mark.asyncio
async def test_inference_force_run_bypasses_success_dedup(mocker, tmp_path: Path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "test", "output_dir": str(tmp_path)},
            "providers": {"azure_ai_inference": {"endpoint": "https://real.ai.azure.com"}},
            "inference": {"provider": "azure_ai_inference", "mode": "request_response", "model": "gpt-4o"},
            "judging": {"enabled": False},
        }
    )
    append_jsonl(
        cfg.outputs_path(),
        InferenceOutput(instance_id="i1", model_id="gpt-4o", prompt="Q", response="old", status="success"),
    )
    item = DatasetItem(instance_id="i1", prompt="Q", ground_truth="A")

    async def _fake_execute(self, **kwargs):  # noqa: ANN001
        kwargs["on_result"](
            inference_runner.ChatResponse(
                request_id="i1",
                content="new",
                prompt_tokens=1,
                completion_tokens=1,
            )
        )
        return []

    execute = mocker.patch.object(
        RequestResponseExecutor,
        "execute",
        autospec=True,
        side_effect=_fake_execute,
    )

    skipped = await inference_runner.run_inference(cfg, [item], force_run=False)
    forced = await inference_runner.run_inference(cfg, [item], force_run=True)

    assert skipped == []
    assert len(forced) == 1
    assert forced[0].response == "new"
    assert execute.call_count == 1


@pytest.mark.asyncio
async def test_judge_force_run_bypasses_success_dedup(mocker, tmp_path: Path):
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
    inference_output = InferenceOutput(instance_id="i1", model_id="model-a", prompt="Q", response="A")
    append_jsonl(
        cfg.judge_outputs_path(),
        JudgeOutput(
            instance_id="i1",
            model_id="model-a",
            judge_name="judge-a",
            score=5,
            verdict="correct",
            reason="ok",
            raw_output="5",
            judge_prompt="prompt",
        ),
    )

    async def _fake_execute(self, **kwargs):  # noqa: ANN001
        kwargs["on_result"](
            judge_runner.ChatResponse(
                request_id="judge-a::i1::model-a",
                content="4",
                prompt_tokens=1,
                completion_tokens=1,
            )
        )
        return []

    execute = mocker.patch.object(
        RequestResponseExecutor,
        "execute",
        autospec=True,
        side_effect=_fake_execute,
    )

    skipped = await judge_runner.run_judge(cfg, [inference_output], {"i1": "A"}, force_run=False)
    forced = await judge_runner.run_judge(cfg, [inference_output], {"i1": "A"}, force_run=True)

    assert skipped == []
    assert len(forced) == 1
    assert forced[0].score == 4
    assert execute.call_count == 1
