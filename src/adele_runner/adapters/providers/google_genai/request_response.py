"""Google Gemini request-response adapter using the google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.types import AdapterCapabilities, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class GoogleGenAIRequestResponseAdapter:
    """Wrap `google.genai.Client` as a normalized request-response transport."""

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
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("google-genai is required. Run: uv sync") from exc

        google_cfg = self._cfg.providers.google_genai
        if google_cfg.backend == "vertex_ai":
            return genai.Client(
                vertexai=True,
                project=google_cfg.project,
                location=google_cfg.location,
            )
        return genai.Client(api_key=self._cfg.get_google_api_key())

    def _flatten_messages(self, request: ChatRequest) -> str:
        parts: list[str] = []
        for message in request.messages:
            if message.role == "user":
                parts.append(message.content)
            else:
                parts.append(f"{message.role.upper()}:\n{message.content}")
        return "\n\n".join(parts)

    async def send(self, request: ChatRequest, *, timeout_s: float) -> ChatResponse:
        from google.genai import types  # type: ignore[import]

        generation_config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
        )

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=request.model,
                    contents=self._flatten_messages(request),
                    config=generation_config,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Google request timed out for request_id=%s after %.1fs",
                request.request_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            logger.error("Google request failed for request_id=%s: %s", request.request_id, exc)
            raise

        latency = time.monotonic() - t0
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        finish_reason = None

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            finish_reason = (
                getattr(finish, "name", None) or getattr(finish, "value", None) or str(finish)
                if finish is not None
                else None
            )

        if self._rate_limiter is not None:
            self._rate_limiter.update_token_usage(  # type: ignore[attr-defined]
                prompt_tokens or 0,
                completion_tokens or 0,
            )

        return ChatResponse(
            request_id=request.request_id,
            content=getattr(response, "text", "") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=round(latency, 3),
            finish_reason=finish_reason,
            raw_output=getattr(response, "text", "") or "",
            metadata=dict(request.metadata),
        )

