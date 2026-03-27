from __future__ import annotations

import pandas as pd

from adele_runner.config import AppConfig
from adele_runner.pipeline.merge import merge_results
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.io import append_jsonl


def test_merge_sets_verification_none_when_any_judge_is_missing_or_none(tmp_path):
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "test", "output_dir": str(tmp_path)},
            "judging": {
                "enabled": True,
                "judges": [
                    {"name": "j1", "provider": "azure_ai_inference", "mode": "request_response", "model": "m1"},
                    {"name": "j2", "provider": "azure_ai_inference", "mode": "request_response", "model": "m2"},
                ],
            },
        }
    )
    append_jsonl(
        cfg.outputs_path(),
        InferenceOutput(instance_id="i1", model_id="model-a", prompt="Q", response="A"),
    )
    append_jsonl(
        cfg.judge_outputs_path(),
        JudgeOutput(
            instance_id="i1",
            model_id="model-a",
            judge_name="j1",
            score=4,
            verdict="correct",
            reason="ok",
            raw_output="{}",
            judge_prompt="",
        ),
    )
    append_jsonl(
        cfg.judge_outputs_path(),
        JudgeOutput(
            instance_id="i1",
            model_id="model-a",
            judge_name="j2",
            score=None,
            verdict=None,
            reason=None,
            raw_output="oops",
            judge_prompt="",
            status="parse_failed",
            error_message="bad json",
        ),
    )

    merged_path = merge_results(cfg)
    df = pd.read_parquet(merged_path)
    assert df.loc[0, "judge_j2_score"] is None or pd.isna(df.loc[0, "judge_j2_score"])
    assert df.loc[0, "avg_judge_score"] is None or pd.isna(df.loc[0, "avg_judge_score"])
    assert df.loc[0, "verification_score"] is None or pd.isna(df.loc[0, "verification_score"])
