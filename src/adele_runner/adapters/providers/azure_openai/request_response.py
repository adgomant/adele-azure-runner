"""Azure OpenAI request-response adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.types import AdapterCapabilities, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class AzureOpenAIRequestResponseAdapter:
    """Use the OpenAI-compatible Azure client for request-response chat completions."""

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
            from openai import AzureOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("openai package is required for azure_openai mode. Run: uv sync") from exc

        return AzureOpenAI(
            azure_endpoint=self._cfg.providers.azure_openai.endpoint,
            api_key=self._cfg.get_provider_api_key("azure_openai"),
            api_version=self._cfg.providers.azure_openai.api_version,
        )

    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse:
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=request.model,
                    messages=[
                        {"role": message.role, "content": message.content}
                        for message in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Azure OpenAI request timed out for request_id=%s after %.1fs",
                request.request_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            logger.error("Azure OpenAI request failed for request_id=%s: %s", request.request_id, exc)
            raise

        latency = time.monotonic() - t0
        choice = response.choices[0]
        usage = response.usage
        if self._rate_limiter is not None and usage is not None:
            self._rate_limiter.update_token_usage(  # type: ignore[attr-defined]
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )

        return ChatResponse(
            request_id=request.request_id,
            content=(choice.message.content or "") if choice.message else "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            latency_s=round(latency, 3),
            raw_output=(choice.message.content or "") if choice.message else "",
            metadata=dict(request.metadata),
        )
