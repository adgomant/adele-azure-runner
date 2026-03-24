"""Anthropic message batches adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.types import (
    AdapterCapabilities,
    ChatRequest,
    ChatResponse,
    ExecutionSettings,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 10


class AnthropicBatchAdapter:
    """Use Anthropic Message Batches as a normalized batch transport."""

    capabilities = AdapterCapabilities(request_response=False, batch=True)

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("anthropic package is required for anthropic mode. Run: uv sync") from exc

        kwargs: dict[str, Any] = {"api_key": self._cfg.get_provider_api_key("anthropic")}
        if self._cfg.providers.anthropic.base_url:
            kwargs["base_url"] = self._cfg.providers.anthropic.base_url
        return anthropic.Anthropic(**kwargs)

    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        _ = run_dir
        batch = self._client.messages.batches.create(
            requests=[
                {
                    "custom_id": request.request_id,
                    "params": {
                        "model": request.model,
                        "system": "\n\n".join(
                            message.content for message in request.messages if message.role == "system"
                        )
                        or None,
                        "messages": [
                            {
                                "role": "assistant" if message.role == "assistant" else "user",
                                "content": message.content,
                            }
                            for message in request.messages
                            if message.role != "system"
                        ],
                        "max_tokens": request.max_tokens or 1024,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                    },
                }
                for request in requests
            ]
        )

        poll_start = time.monotonic()
        while getattr(batch, "processing_status", None) not in {"ended", "completed"}:
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(f"Anthropic batch polling exceeded {settings.max_poll_time_s}s.")
            time.sleep(_POLL_INTERVAL_S)
            batch = self._client.messages.batches.retrieve(batch.id)

        outputs: list[ChatResponse] = []
        request_map = {request.request_id: request for request in requests}
        for result in self._client.messages.batches.results(batch.id):
            custom_id = getattr(result, "custom_id", None)
            result_body = getattr(result, "result", None)
            message = getattr(result_body, "message", None)
            content_blocks = getattr(message, "content", None) or []
            text = "".join(getattr(block, "text", "") for block in content_blocks)
            usage = getattr(message, "usage", None)
            request = request_map.get(custom_id)
            outputs.append(
                ChatResponse(
                    request_id=str(custom_id),
                    content=text,
                    prompt_tokens=getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "output_tokens", None),
                    finish_reason=getattr(message, "stop_reason", None),
                    raw_output=text,
                    metadata=dict(request.metadata) if request else {},
                )
            )
        return outputs
