"""Merge inference + judge outputs into a single Parquet artefact."""

from __future__ import annotations

import logging
from pathlib import Path

from adele_runner.config import AppConfig
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.io import latest_jsonl_by_key, write_parquet

logger = logging.getLogger(__name__)


def merge_results(config: AppConfig) -> Path:
    """Merge outputs.jsonl + judge_outputs.jsonl → merged_results.parquet."""
    outputs_path = config.outputs_path()
    judge_path = config.judge_outputs_path()
    merged_path = config.merged_path()

    inf_records = list(latest_jsonl_by_key(outputs_path, InferenceOutput, "instance_id", "model_id").values())
    judge_records = list(
        latest_jsonl_by_key(judge_path, JudgeOutput, "instance_id", "model_id", "judge_name").values()
    )
    expected_judges = [judge.name for judge in config.judging.judges] if config.judging.enabled else []

    logger.info(
        "Merging %d inference outputs + %d judge outputs.",
        len(inf_records),
        len(judge_records),
    )

    # Build judge lookup: (instance_id, model_id) → {judge_name: JudgeOutput}
    judge_map: dict[tuple[str, str], dict[str, JudgeOutput]] = {}
    for jr in judge_records:
        key = (jr.instance_id, jr.model_id)
        judge_map.setdefault(key, {})[jr.judge_name] = jr

    rows = []
    for inf in inf_records:
        key = (inf.instance_id, inf.model_id)
        row: dict = {
            "instance_id": inf.instance_id,
            "model_id": inf.model_id,
            "prompt": inf.prompt,
            "response": inf.response,
            "tokens_prompt": inf.tokens_prompt,
            "tokens_completion": inf.tokens_completion,
            "latency_s": inf.latency_s,
            "finish_reason": inf.finish_reason,
            "inference_timestamp": str(inf.timestamp),
            "run_id": inf.run_id,
        }
        judges = judge_map.get(key, {})
        judge_scores: list[int] = []
        for judge_name in expected_judges:
            jr = judges.get(judge_name)
            row[f"judge_{judge_name}_score"] = jr.score if jr else None
            row[f"judge_{judge_name}_verdict"] = jr.verdict if jr else None
            row[f"judge_{judge_name}_reason"] = jr.reason if jr else None
            row[f"judge_{judge_name}_status"] = jr.status if jr else None
            row[f"judge_{judge_name}_error_message"] = jr.error_message if jr else None
            if jr is not None and jr.score is not None:
                judge_scores.append(jr.score)
        for judge_name, jr in judges.items():
            if judge_name in expected_judges:
                continue
            row[f"judge_{judge_name}_score"] = jr.score
            row[f"judge_{judge_name}_verdict"] = jr.verdict
            row[f"judge_{judge_name}_reason"] = jr.reason
            row[f"judge_{judge_name}_status"] = jr.status
            row[f"judge_{judge_name}_error_message"] = jr.error_message

        has_all_expected_scores = bool(expected_judges) and all(
            judges.get(judge_name) is not None and judges[judge_name].score is not None for judge_name in expected_judges
        )
        if has_all_expected_scores:
            avg = sum(judge_scores) / len(judge_scores)
            row["avg_judge_score"] = round(avg, 3)
            row["verification_score"] = 1 if avg >= 3 else 0
        else:
            row["avg_judge_score"] = None
            row["verification_score"] = None

        rows.append(row)

    write_parquet(merged_path, rows)
    logger.info("Merged %d rows → %s", len(rows), merged_path)
    return merged_path
