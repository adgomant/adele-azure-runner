from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from adele_runner.config import AppConfig
from adele_runner.pipeline import judge_runner
from adele_runner.runtime.batch_jobs import BatchJobRecord, append_batch_job_record, latest_batch_job_records
from adele_runner.runtime.types import (
    ChatResponse,
    ExecutionSettings,
    ResolvedJudgePlan,
    ResolvedJudgeTarget,
    ResolvedModeBinding,
    ResolvedProviderTarget,
)
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.io import read_jsonl


class _FakeResumableBatchAdapter:
    poll_interval_s = 0.0

    def __init__(self) -> None:
        self.submit_calls = 0

    def split_requests(self, requests, settings):  # noqa: ANN001
        return [requests]

    def submit_chunk(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.submit_calls += 1
        raise AssertionError("submit_chunk should not be called when recovering an existing batch")

    def refresh_submission(self, submission, settings):  # noqa: ANN001
        return submission.model_copy(
            update={
                "last_known_status": "completed",
                "is_terminal": True,
                "is_successful": True,
            }
        )

    def fetch_results(self, submission, requests, settings):  # noqa: ANN001
        return [
            ChatResponse(
                request_id=requests[0].request_id,
                content="5",
                prompt_tokens=10,
                completion_tokens=1,
            )
        ]


@pytest.mark.asyncio
async def test_run_judge_recovers_persisted_batch_without_resubmitting(tmp_path: Path):
    adapter = _FakeResumableBatchAdapter()
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "resume-run", "output_dir": str(tmp_path)},
            "judging": {
                "enabled": True,
                "prompt_template": "v2",
                "judges": [
                    {
                        "name": "judge-a",
                        "provider": "anthropic",
                        "mode": "batch",
                        "model": "claude-sonnet-4-5",
                    }
                ],
            },
        }
    )
    inference_output = InferenceOutput(
        instance_id="i1",
        model_id="model-a",
        prompt="Question?",
        response="Answer",
        run_id="resume-run",
    )
    batch_request_id = "judge-a::i1::model-a"
    append_batch_job_record(
        cfg.batch_jobs_path(),
        BatchJobRecord(
            run_id="resume-run",
            stage="judging",
            provider="anthropic",
            judge_name="judge-a",
            chunk_id="judging-anthropic-judge-a-abc123",
            remote_batch_id="msgbatch_123",
            request_ids=[batch_request_id],
            request_count=1,
            last_known_status="in_progress",
        ),
    )

    with patch.object(
        judge_runner,
        "resolve_judge_plans",
        return_value=[
            ResolvedJudgePlan(
                target=ResolvedJudgeTarget(
                    judge_name="judge-a",
                    provider_target=ResolvedProviderTarget(
                        provider_kind="anthropic",
                        model="claude-sonnet-4-5",
                    ),
                    requested_mode="batch",
                    prompt_mode="batch",
                    prompt_template="v2",
                    max_tokens=16,
                ),
                binding=ResolvedModeBinding(
                    provider_target=ResolvedProviderTarget(
                        provider_kind="anthropic",
                        model="claude-sonnet-4-5",
                    ),
                    execution_kind="batch",
                    create_adapter=lambda *args, **kwargs: adapter,  # noqa: ARG005
                ),
                settings=ExecutionSettings(
                    max_in_flight=1,
                    request_timeout_s=30.0,
                    max_retries=1,
                    backoff_base_s=0.0,
                    backoff_max_s=0.0,
                    max_poll_time_s=60.0,
                    batch_completion_window="24h",
                ),
            )
        ],
    ):
        outputs = await judge_runner.run_judge(cfg, [inference_output], {"i1": "Ground truth"})

    assert adapter.submit_calls == 0
    assert len(outputs) == 1
    assert outputs[0].score == 5

    judge_outputs = read_jsonl(cfg.judge_outputs_path(), JudgeOutput)
    assert len(judge_outputs) == 1
    latest_jobs = latest_batch_job_records(
        cfg.batch_jobs_path(),
        run_id="resume-run",
        stage="judging",
        provider="anthropic",
        judge_name="judge-a",
    )
    assert latest_jobs[0].results_downloaded_at is not None
