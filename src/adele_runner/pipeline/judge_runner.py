from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from adele_runner.adapters.foundry_inference import FoundryChatAdapter
from adele_runner.config import AppConfig, JudgeModelConfig
from adele_runner.models import JudgeOutput, JudgeResult
from adele_runner.utils.io import append_jsonl, ensure_run_dir, load_dedup_keys, stream_jsonl


def build_judge_prompt(prompt: str, ground_truth: str | None, response_text: str) -> str:
    return (
        "You are a strict benchmark judge. Return JSON only with keys: "
        "score (1..5), verdict (correct|incorrect|partial|unknown), reason.\n\n"
        f"PROMPT:\n{prompt}\n\nGROUND_TRUTH:\n{ground_truth or ''}\n\nMODEL_RESPONSE:\n{response_text}"
    )


def parse_judge_output(raw_text: str) -> JudgeResult:
    candidate = raw_text.strip()
    try:
        data = json.loads(candidate)
        return JudgeResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        repaired = match.group(0)
        repaired = repaired.replace("\n", " ")
        data = json.loads(repaired)
        return JudgeResult.model_validate(data)

    raise ValueError("Unable to parse judge output as valid JSON.")


def run_judging(config: AppConfig) -> Path:
    run_dir = ensure_run_dir(config.run.output_dir, config.run.run_id)
    outputs_path = run_dir / "outputs.jsonl"
    judge_path = run_dir / "judge_outputs.jsonl"
    dedup = load_dedup_keys(judge_path, ("instance_id", "model_id", "judge_name"))

    base_adapter = FoundryChatAdapter(config.inference.foundry)
    adapters: dict[str, FoundryChatAdapter] = {}

    for judge in config.judging.judges:
        adapters[judge.name] = _build_judge_adapter(config, judge, base_adapter)

    for item in stream_jsonl(outputs_path):
        for judge in config.judging.judges:
            key = (str(item["instance_id"]), str(item["model_id"]), judge.name)
            if key in dedup:
                continue

            prompt_text = build_judge_prompt(
                prompt=str(item.get("prompt", "")),
                ground_truth=str(item.get("metadata", {}).get("ground_truth", "")),
                response_text=str(item.get("response_text", "")),
            )
            raw = adapters[judge.name].complete(prompt_text, system_prompt="Respond with JSON only.")
            parsed = parse_judge_output(raw["text"])
            out = JudgeOutput(
                run_id=config.run.run_id,
                instance_id=str(item["instance_id"]),
                model_id=str(item["model_id"]),
                judge_name=judge.name,
                judge_model_id=judge.model,
                prompt_text=prompt_text,
                raw_output_text=raw["text"],
                parsed=parsed,
            )
            append_jsonl(judge_path, out.model_dump(mode="json"))
            dedup.add(key)
    return judge_path


def _build_judge_adapter(config: AppConfig, judge: JudgeModelConfig, base: FoundryChatAdapter) -> FoundryChatAdapter:
    if judge.model == config.inference.foundry.model:
        return base

    judge_cfg = config.inference.foundry.model_copy(update={"model": judge.model})
    return FoundryChatAdapter(judge_cfg)
