from __future__ import annotations

from types import SimpleNamespace

from adele_runner.adapters.providers.anthropic.batch import (
    _CUSTOM_ID_ALLOWED_RE,
    AnthropicBatchAdapter,
    _CUSTOM_ID_MAX_LEN,
)
from adele_runner.runtime.types import ChatMessage, ChatRequest, ExecutionSettings


def test_batch_custom_id_is_shortened_to_provider_limit():
    request_id = "judge-a::" + ("segment-" * 12)
    custom_id = AnthropicBatchAdapter._batch_custom_id(request_id)

    assert len(custom_id) == _CUSTOM_ID_MAX_LEN
    assert custom_id != request_id
    assert _CUSTOM_ID_ALLOWED_RE.search(custom_id) is None


def test_extract_results_restores_original_request_id():
    request = ChatRequest(
        request_id="judge-a::" + ("segment-" * 12),
        model="claude-sonnet-4-5",
        messages=(ChatMessage(role="user", content="hello"),),
    )
    custom_id = AnthropicBatchAdapter._batch_custom_id(request.request_id)
    result = SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            message=SimpleNamespace(
                content=[SimpleNamespace(text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                stop_reason="end_turn",
            )
        ),
    )

    adapter = AnthropicBatchAdapter.__new__(AnthropicBatchAdapter)
    adapter._client = SimpleNamespace(
        messages=SimpleNamespace(
            batches=SimpleNamespace(results=lambda _batch_id: [result]),
        )
    )
    settings = ExecutionSettings(
        max_in_flight=1,
        request_timeout_s=30.0,
        max_retries=1,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        max_poll_time_s=60.0,
        batch_completion_window="24h",
    )

    outputs = adapter._extract_results("batch_123", [request], settings)

    assert outputs[0].request_id == request.request_id
    assert outputs[0].content == "ok"


def test_batch_request_omits_top_p_when_temperature_is_present():
    request = ChatRequest(
        request_id="r1",
        model="claude-sonnet-4-5",
        messages=(ChatMessage(role="user", content="hello"),),
        temperature=0.0,
        top_p=1.0,
    )

    adapter = AnthropicBatchAdapter.__new__(AnthropicBatchAdapter)
    payload = adapter._to_batch_request(request)

    assert payload["params"]["temperature"] == 0.0
    assert "top_p" not in payload["params"]


def test_call_with_retry_retries_transient_failure():
    adapter = AnthropicBatchAdapter.__new__(AnthropicBatchAdapter)
    seen = {"calls": 0}

    def flaky():
        seen["calls"] += 1
        if seen["calls"] == 1:
            raise RuntimeError("transient connection problem")
        return "ok"

    settings = ExecutionSettings(
        max_in_flight=1,
        request_timeout_s=30.0,
        max_retries=1,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        max_poll_time_s=60.0,
        batch_completion_window="24h",
    )

    result = adapter._call_with_retry(settings, flaky)

    assert result == "ok"
    assert seen["calls"] == 2
