"""Tests for judging stage request building and output mapping."""

from __future__ import annotations

from adele_runner.runtime.types import ChatResponse, ResolvedJudgeTarget, ResolvedProviderTarget
from adele_runner.schemas import InferenceOutput
from adele_runner.stages.judging import build_judge_output, build_judge_request


def test_build_judge_request_v1():
    inference_output = InferenceOutput(
        instance_id="i1",
        model_id="model-a",
        prompt="Question?",
        response="Answer",
    )
    target = ResolvedJudgeTarget(
        judge_name="judge-a",
        provider_target=ResolvedProviderTarget(provider_kind="azure_ai_inference", model="gpt-4o"),
        requested_mode="request_response",
        prompt_mode="request_response",
        prompt_template="v1",
        max_tokens=512,
    )

    request, prompt = build_judge_request(inference_output, "Ground truth", target)

    assert request.model == "gpt-4o"
    assert request.request_id == "judge-a::i1::model-a"
    assert "Ground truth" in prompt
    assert request.messages[0].content == prompt


def test_build_judge_output_v2():
    inference_output = InferenceOutput(
        instance_id="i1",
        model_id="model-a",
        prompt="Question?",
        response="Answer",
    )
    target = ResolvedJudgeTarget(
        judge_name="judge-a",
        provider_target=ResolvedProviderTarget(provider_kind="azure_ai_inference", model="gpt-4o"),
        requested_mode="request_response",
        prompt_mode="request_response",
        prompt_template="v2",
        max_tokens=512,
    )
    response = ChatResponse(
        request_id="judge-a::i1::model-a",
        content="4",
        prompt_tokens=100,
        completion_tokens=1,
    )

    output = build_judge_output(inference_output, target, "Rendered prompt", response, "run-1")

    assert output.score == 4
    assert output.judge_name == "judge-a"
    assert output.raw_output == "4"
    assert output.judge_prompt == "Rendered prompt"
    assert output.status == "success"


def test_build_judge_output_v2_parse_failure():
    inference_output = InferenceOutput(
        instance_id="i1",
        model_id="model-a",
        prompt="Question?",
        response="Answer",
    )
    target = ResolvedJudgeTarget(
        judge_name="judge-a",
        provider_target=ResolvedProviderTarget(provider_kind="azure_ai_inference", model="gpt-4o"),
        requested_mode="request_response",
        prompt_mode="request_response",
        prompt_template="v2",
        max_tokens=512,
    )
    response = ChatResponse(
        request_id="judge-a::i1::model-a",
        content="not a score",
        prompt_tokens=100,
        completion_tokens=1,
    )

    output = build_judge_output(inference_output, target, "Rendered prompt", response, "run-1")

    assert output.score is None
    assert output.status == "parse_failed"
