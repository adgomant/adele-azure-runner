"""Azure AI Inference request-response adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig, RateLimitsConfig
from adele_runner.runtime.types import AdapterCapabilities, ChatMessage, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class AzureAIInferenceRequestResponseAdapter:
    """Wrap ChatCompletionsClient as a normalized request-response transport."""

    capabilities = AdapterCapabilities(request_response=True, batch=False)

    def __init__(
        self,
        config: AppConfig,
        rate_limiter: object | None = None,
        configured_rate_limits: RateLimitsConfig | None = None,
    ) -> None:
        self._cfg = config
        self._client = self._build_client()
        self._rate_limit_warned = False
        self._rate_limiter = rate_limiter
        self._configured_rate_limits = configured_rate_limits

    def _build_client(self) -> Any:
        try:
            from azure.ai.inference import ChatCompletionsClient  # type: ignore[import]
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("azure-ai-inference is required. Run: uv sync") from exc

        api_key = self._cfg.get_foundry_api_key()
        return ChatCompletionsClient(
            endpoint=self._cfg.providers.azure_ai_inference.endpoint,
            credential=AzureKeyCredential(api_key),
        )

    def _on_response(self, response: Any) -> None:
        try:
            headers = dict(response.http_response.headers)
        except AttributeError:
            return
        self._check_rate_limit_headers(headers)
        if self._rate_limiter is not None:
            self._rate_limiter.update_from_headers(headers)  # type: ignore[attr-defined]

    def _check_rate_limit_headers(self, headers: dict[str, str]) -> None:
        if self._rate_limit_warned:
            return
        rl = self._configured_rate_limits
        if rl is None:
            return

        actual_tpm_str = headers.get("x-ratelimit-limit-tokens")
        actual_rpm_str = headers.get("x-ratelimit-limit-requests")
        if actual_tpm_str is None and actual_rpm_str is None:
            return
        self._rate_limit_warned = True

        if actual_tpm_str:
            try:
                actual_tpm = int(actual_tpm_str)
                if abs(actual_tpm - rl.tokens_per_minute) / max(1, rl.tokens_per_minute) > 0.2:
                    logger.warning(
                        "Actual TPM from Azure AI Inference (%d) differs from config (%d) by >20%%",
                        actual_tpm,
                        rl.tokens_per_minute,
                    )
            except (ValueError, TypeError):
                pass

        if actual_rpm_str:
            try:
                actual_rpm = int(actual_rpm_str)
                if abs(actual_rpm - rl.requests_per_minute) / max(1, rl.requests_per_minute) > 0.2:
                    logger.warning(
                        "Actual RPM from Azure AI Inference (%d) differs from config (%d) by >20%%",
                        actual_rpm,
                        rl.requests_per_minute,
                    )
            except (ValueError, TypeError):
                pass

    def _to_sdk_message(self, message: ChatMessage) -> Any:
        from azure.ai.inference.models import (  # type: ignore[import]
            AssistantMessage,
            SystemMessage,
            UserMessage,
        )

        if message.role == "system":
            return SystemMessage(content=message.content)
        if message.role == "assistant":
            return AssistantMessage(content=message.content)
        return UserMessage(content=message.content)

    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse:
        messages = [self._to_sdk_message(message) for message in request.messages]
        t0 = time.monotonic()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.complete,
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    raw_response_hook=self._on_response,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Azure AI Inference request timed out for request_id=%s after %.1fs",
                request.request_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            logger.error(
                "Azure AI Inference request failed for request_id=%s: %s",
                request.request_id,
                exc,
            )
            raise

        latency = time.monotonic() - t0
        choice = response.choices[0]
        usage = response.usage

        if self._rate_limiter is not None and usage:
            self._rate_limiter.update_token_usage(  # type: ignore[attr-defined]
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
            )

        return ChatResponse(
            request_id=request.request_id,
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_s=round(latency, 3),
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            raw_output=choice.message.content or "",
            metadata=dict(request.metadata),
        )

