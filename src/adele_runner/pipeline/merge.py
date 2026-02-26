from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.utils.io import ensure_run_dir, stream_jsonl, write_parquet


def merge_results(config: AppConfig) -> Path:
    run_dir = ensure_run_dir(config.run.output_dir, config.run.run_id)
    outputs_path = run_dir / "outputs.jsonl"
    judges_path = run_dir / "judge_outputs.jsonl"
    merged_path = run_dir / "merged_results.parquet"

    outputs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in stream_jsonl(outputs_path):
        key = (str(row["instance_id"]), str(row["model_id"]))
        outputs_by_key[key] = row

    judge_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stream_jsonl(judges_path):
        key = (str(row["instance_id"]), str(row["model_id"]))
        judge_rows[key].append(row)

    merged: list[dict[str, Any]] = []
    for key, out in outputs_by_key.items():
        base = {
            "run_id": out.get("run_id"),
            "instance_id": out.get("instance_id"),
            "model_id": out.get("model_id"),
            "prompt": out.get("prompt"),
            "ground_truth": out.get("metadata", {}).get("ground_truth"),
            "model_response": out.get("response_text"),
        }
        judges = judge_rows.get(key, [])
        if not judges:
            merged.append(base)
            continue
        for judge in judges:
            merged.append(
                {
                    **base,
                    "judge_name": judge.get("judge_name"),
                    "judge_model_id": judge.get("judge_model_id"),
                    "judge_score": judge.get("parsed", {}).get("score"),
                    "judge_verdict": judge.get("parsed", {}).get("verdict"),
                    "judge_reason": judge.get("parsed", {}).get("reason"),
                }
            )

    write_parquet(merged_path, merged)
    return merged_path
