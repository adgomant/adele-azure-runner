"""Tests for inference stage request building and response mapping."""

from __future__ import annotations

from adele_runner.runtime.types import ChatResponse, ResolvedInferenceTarget, ResolvedProviderTarget
from adele_runner.schemas import DatasetItem
from adele_runner.stages.inference import build_inference_output, build_inference_request


def test_build_inference_request():
    item = DatasetItem(instance_id="i1", prompt="Hello", ground_truth="Hi")
    target = ResolvedInferenceTarget(
        provider_target=ResolvedProviderTarget(provider_kind="azure_ai_inference", model="gpt-4o"),
        requested_mode="request_response",
        prompt_mode="request_response",
        temperature=0.2,
        max_tokens=256,
        top_p=0.9,
    )

    request = build_inference_request(item, target)

    assert request.request_id == "i1"
    assert request.model == "gpt-4o"
    assert request.messages[0].content == "Hello"
    assert request.max_tokens == 256


def test_build_inference_output():
    item = DatasetItem(instance_id="i1", prompt="Hello", ground_truth="Hi")
    target = ResolvedInferenceTarget(
        provider_target=ResolvedProviderTarget(provider_kind="google_genai", model="gemini-2.5-flash"),
        requested_mode="request_response",
        prompt_mode="request_response",
        temperature=0.0,
        max_tokens=256,
        top_p=1.0,
    )
    response = ChatResponse(
        request_id="i1",
        content="Hello back",
        prompt_tokens=10,
        completion_tokens=4,
        latency_s=1.25,
        finish_reason="stop",
    )

    output = build_inference_output(item, target, response, "run-1")

    assert output.instance_id == "i1"
    assert output.model_id == "gemini-2.5-flash"
    assert output.response == "Hello back"
    assert output.tokens_prompt == 10
    assert output.run_id == "run-1"
