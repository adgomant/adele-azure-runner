"""Judging-stage prompt building, parsing, and output mapping."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from adele_runner.runtime.types import ChatMessage, ChatRequest, ChatResponse, ResolvedJudgeTarget
from adele_runner.schemas import InferenceOutput, JudgeOutput

logger = logging.getLogger(__name__)

_JUDGE_PROMPT_V2 = """\
TASK:
{question}

GROUND TRUTH:
{ground_truth}

Model Response:
{response}

You are an evaluation system that compares a MODEL RESPONSE to a GROUND TRUTH of a given TASK \
(as shown above), and rates the response on a scale of 1 to 5, where 5 means surely correct \
and 1 means surely incorrect. Your response must be concise, responding with exactly one \
integer score between 1 and 5 and nothing else that may be redundant.

score:
"""


_JUDGE_PROMPT_V1 = """\
You are an expert evaluator. Given the following question, reference answer, and model response, \
evaluate the model response.

Question:
{question}

Reference Answer:
{ground_truth}

Model Response:
{response}

Instructions:
- Score the response on a scale of 1-5 (1=completely wrong, 5=perfectly correct).
- Provide a verdict: "correct", "incorrect", "partial", or "unknown".
- Provide a brief reason (1-2 sentences).

Respond in JSON only, with exactly this structure:
{{"score": <int 1-5>, "verdict": "<correct|incorrect|partial|unknown>", "reason": "<string>"}}
"""

_JSON_LIKE_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_VALID_VERDICTS = {"correct", "incorrect", "partial", "unknown"}


def build_judge_prompt(
    item_prompt: str,
    ground_truth: str,
    response: str,
    template: str = "v1",
) -> str:
    """Build the judge prompt for a single inference output."""
    if template == "v1":
        return _JUDGE_PROMPT_V1.format(
            question=item_prompt,
            ground_truth=ground_truth,
            response=response,
        )
    if template == "v2":
        return _JUDGE_PROMPT_V2.format(
            question=item_prompt,
            ground_truth=ground_truth,
            response=response,
        )
    raise ValueError(f"Unknown judge prompt template: {template}")


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse judge JSON output with repair fallback."""
    try:
        obj = json.loads(raw.strip())
        return _validate_judge_obj(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    match = _JSON_LIKE_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group())
            return _validate_judge_obj(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    score_match = re.search(r'"score"\s*:\s*(\d)', raw)
    verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)

    score = int(score_match.group(1)) if score_match else 1
    verdict = verdict_match.group(1) if verdict_match else "unknown"
    if verdict not in _VALID_VERDICTS:
        verdict = "unknown"
    reason = reason_match.group(1) if reason_match else "Could not parse reason."

    logger.warning("Used regex fallback for judge output: score=%d verdict=%s", score, verdict)
    return {"score": max(1, min(5, score)), "verdict": verdict, "reason": reason}


def _validate_judge_obj(obj: dict[str, Any]) -> dict[str, Any]:
    score = int(obj["score"])
    if not (1 <= score <= 5):
        raise ValueError(f"Score out of range: {score}")
    verdict = str(obj.get("verdict", "unknown"))
    if verdict not in _VALID_VERDICTS:
        verdict = "unknown"
    reason = str(obj.get("reason", ""))
    return {"score": score, "verdict": verdict, "reason": reason}


def parse_judge_v2(raw: str) -> dict[str, Any]:
    """Parse v2 judge output: expects a bare integer score 1-5."""
    stripped = raw.strip()
    try:
        score = int(stripped)
        score = max(1, min(5, score))
        return {
            "score": score,
            "verdict": "unknown",
            "reason": "Scored via v2 bare-integer prompt",
        }
    except ValueError:
        pass

    match = re.search(r"\b([1-5])\b", raw)
    if match:
        score = int(match.group(1))
        return {
            "score": score,
            "verdict": "unknown",
            "reason": "Scored via v2 bare-integer prompt",
        }

    logger.warning("Could not parse v2 judge output, defaulting to score=1: %r", raw[:200])
    return {
        "score": 1,
        "verdict": "unknown",
        "reason": "Scored via v2 bare-integer prompt",
    }


def build_judge_request(
    inference_output: InferenceOutput,
    ground_truth: str,
    target: ResolvedJudgeTarget,
) -> tuple[ChatRequest, str]:
    """Build a normalized judge request and return it with its rendered prompt."""
    judge_prompt = build_judge_prompt(
        item_prompt=inference_output.prompt,
        ground_truth=ground_truth,
        response=inference_output.response,
        template=target.prompt_template,
    )
    request = ChatRequest(
        request_id=f"{target.judge_name}::{inference_output.instance_id}::{inference_output.model_id}",
        model=target.model,
        messages=(ChatMessage(role="user", content=judge_prompt),),
        temperature=0.0,
        max_tokens=target.max_tokens,
        top_p=1.0,
        metadata={
            "instance_id": inference_output.instance_id,
            "model_id": inference_output.model_id,
            "judge_name": target.judge_name,
        },
    )
    return request, judge_prompt


def build_judge_output(
    inference_output: InferenceOutput,
    target: ResolvedJudgeTarget,
    judge_prompt: str,
    response: ChatResponse,
    run_id: str,
) -> JudgeOutput:
    """Convert a normalized judge response into the public judge schema."""
    batch_result_type = response.metadata.get("anthropic_batch_result_type")
    if batch_result_type is not None and batch_result_type != "succeeded":
        return JudgeOutput(
            instance_id=inference_output.instance_id,
            model_id=inference_output.model_id,
            judge_name=target.judge_name,
            score=1,
            verdict="unknown",
            reason=f"Judge request failed in anthropic batch with result_type={batch_result_type}",
            raw_output=str(response.raw_output or response.content),
            judge_prompt=judge_prompt,
            tokens_prompt=response.prompt_tokens,
            tokens_completion=response.completion_tokens,
            run_id=run_id,
        )

    parsed = (
        parse_judge_v2(response.content)
        if target.prompt_template == "v2"
        else parse_judge_json(response.content)
    )
    return JudgeOutput(
        instance_id=inference_output.instance_id,
        model_id=inference_output.model_id,
        judge_name=target.judge_name,
        score=parsed["score"],
        verdict=parsed["verdict"],
        reason=parsed["reason"],
        raw_output=str(response.raw_output or response.content),
        judge_prompt=judge_prompt,
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        run_id=run_id,
    )
