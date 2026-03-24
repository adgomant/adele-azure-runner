"""Anthropic request-response adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.types import AdapterCapabilities, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class AnthropicRequestResponseAdapter:
    """Use the Anthropic Messages API as a request-response transport."""

    capabilities = AdapterCapabilities(request_response=True, batch=False)

    def __init__(
        self,
        config: AppConfig,
        rate_limiter: object | None = None,
        configured_rate_limits: object | None = None,  # noqa: ARG002
    ) -> None:
        self._cfg = config
        self._client = self._build_client()
        self._rate_limiter = rate_limiter

    def _build_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("anthropic package is required for anthropic mode. Run: uv sync") from exc

        kwargs: dict[str, Any] = {"api_key": self._cfg.get_provider_api_key("anthropic")}
        if self._cfg.providers.anthropic.base_url:
            kwargs["base_url"] = self._cfg.providers.anthropic.base_url
        return anthropic.Anthropic(**kwargs)

    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse:
        system_parts = [message.content for message in request.messages if message.role == "system"]
        message_payload = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": message.content,
            }
            for message in request.messages
            if message.role != "system"
        ]

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.messages.create,
                    model=request.model,
                    system="\n\n".join(system_parts) or None,
                    messages=message_payload,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Anthropic request timed out for request_id=%s after %.1fs",
                request.request_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            logger.error("Anthropic request failed for request_id=%s: %s", request.request_id, exc)
            raise

        latency = time.monotonic() - t0
        content_blocks = getattr(response, "content", None) or []
        text = "".join(getattr(block, "text", "") for block in content_blocks)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if self._rate_limiter is not None:
            self._rate_limiter.update_token_usage(  # type: ignore[attr-defined]
                input_tokens or 0,
                output_tokens or 0,
            )

        return ChatResponse(
            request_id=request.request_id,
            content=text,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            finish_reason=getattr(response, "stop_reason", None),
            latency_s=round(latency, 3),
            raw_output=text,
            metadata=dict(request.metadata),
        )
