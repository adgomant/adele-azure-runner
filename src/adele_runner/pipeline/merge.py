"""Merge inference + judge outputs into a single Parquet artefact."""

from __future__ import annotations

import logging
from pathlib import Path

from adele_runner.config import AppConfig
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.io import iter_jsonl, write_parquet

logger = logging.getLogger(__name__)


def merge_results(config: AppConfig) -> Path:
    """Merge outputs.jsonl + judge_outputs.jsonl → merged_results.parquet."""
    outputs_path = config.outputs_path()
    judge_path = config.judge_outputs_path()
    merged_path = config.merged_path()

    inf_records = list(iter_jsonl(outputs_path, InferenceOutput))
    judge_records = list(iter_jsonl(judge_path, JudgeOutput))

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
        for judge_name, jr in judges.items():
            row[f"judge_{judge_name}_score"] = jr.score
            row[f"judge_{judge_name}_verdict"] = jr.verdict
            row[f"judge_{judge_name}_reason"] = jr.reason
            judge_scores.append(jr.score)

        if judge_scores:
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
