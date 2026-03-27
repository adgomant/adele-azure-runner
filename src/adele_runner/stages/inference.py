"""Inference-stage request building and output mapping."""

from __future__ import annotations

from adele_runner.runtime.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ResolvedInferenceTarget,
)
from adele_runner.schemas import DatasetItem, InferenceOutput


def build_inference_request(item: DatasetItem, target: ResolvedInferenceTarget) -> ChatRequest:
    """Build a normalized inference request for a dataset item."""
    return ChatRequest(
        request_id=item.instance_id,
        model=target.model,
        messages=(ChatMessage(role="user", content=item.prompt),),
        temperature=target.temperature,
        max_tokens=target.max_tokens,
        top_p=target.top_p,
        metadata={"instance_id": item.instance_id},
    )


def build_inference_output(
    item: DatasetItem,
    target: ResolvedInferenceTarget,
    response: ChatResponse,
    run_id: str,
) -> InferenceOutput:
    """Convert a normalized response into the public inference schema."""
    return InferenceOutput(
        instance_id=item.instance_id,
        model_id=target.model,
        prompt=item.prompt,
        response=response.content,
        status="success",
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        latency_s=response.latency_s,
        finish_reason=response.finish_reason,
        run_id=run_id,
    )


def build_failed_inference_output(
    item: DatasetItem,
    target: ResolvedInferenceTarget,
    error_message: str,
    run_id: str,
) -> InferenceOutput:
    """Persist a failed inference attempt as an explicit output row."""
    return InferenceOutput(
        instance_id=item.instance_id,
        model_id=target.model,
        prompt=item.prompt,
        response=None,
        status="failed",
        error_message=error_message,
        run_id=run_id,
    )
