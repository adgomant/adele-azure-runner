"""Tests for shared execution lanes."""

from __future__ import annotations

from pathlib import Path

import pytest

from adele_runner.config import RateLimitsConfig
from adele_runner.runtime.executors import (
    BatchExecutor,
    RequestResponseExecutor,
    create_rate_limiter,
)
from adele_runner.runtime.types import ChatMessage, ChatRequest, ChatResponse, ExecutionSettings


class _FakeRequestResponseAdapter:
    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse:
        return ChatResponse(
            request_id=request.request_id,
            content=f"response:{request.request_id}",
            prompt_tokens=1,
            completion_tokens=1,
        )


class _FakeBatchAdapter:
    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        return [
            ChatResponse(
                request_id=request.request_id,
                content=f"batch:{request.request_id}",
            )
            for request in requests
        ]


def _settings() -> ExecutionSettings:
    return ExecutionSettings(
        max_in_flight=2,
        request_timeout_s=30.0,
        max_retries=1,
        backoff_base_s=0.5,
        backoff_max_s=5.0,
        max_poll_time_s=60.0,
        batch_completion_window="24h",
        effective_rpm=None,
    )


def test_create_rate_limiter_returns_none_without_effective_rpm():
    assert create_rate_limiter(_settings(), None) is None


def test_create_rate_limiter_uses_settings_and_limits():
    settings = ExecutionSettings(
        max_in_flight=2,
        request_timeout_s=30.0,
        max_retries=1,
        backoff_base_s=0.5,
        backoff_max_s=5.0,
        max_poll_time_s=60.0,
        batch_completion_window="24h",
        effective_rpm=100,
    )
    limiter = create_rate_limiter(
        settings,
        RateLimitsConfig(tokens_per_minute=80_000, requests_per_minute=300),
    )
    assert limiter is not None


@pytest.mark.asyncio
async def test_request_response_executor_calls_callback():
    executor = RequestResponseExecutor()
    seen: list[str] = []
    requests = [
        ChatRequest(
            request_id="r1",
            model="m",
            messages=(ChatMessage(role="user", content="hello"),),
        )
    ]

    results = await executor.execute(
        adapter=_FakeRequestResponseAdapter(),
        requests=requests,
        settings=_settings(),
        on_result=lambda result: seen.append(result.request_id),  # type: ignore[union-attr]
    )

    assert [result.request_id for result in results] == ["r1"]
    assert seen == ["r1"]


@pytest.mark.asyncio
async def test_batch_executor_calls_callback(tmp_path: Path):
    executor = BatchExecutor()
    seen: list[str] = []
    requests = [
        ChatRequest(
            request_id="r1",
            model="m",
            messages=(ChatMessage(role="user", content="hello"),),
        )
    ]

    results = await executor.execute(
        adapter=_FakeBatchAdapter(),
        requests=requests,
        run_dir=tmp_path,
        settings=_settings(),
        on_result=lambda result: seen.append(result.request_id),  # type: ignore[union-attr]
    )

    assert [result.content for result in results] == ["batch:r1"]
    assert seen == ["r1"]
