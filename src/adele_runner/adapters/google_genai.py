"""Google Gemini inference adapter using the google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.schemas import DatasetItem, InferenceOutput

logger = logging.getLogger(__name__)


class GoogleGenAIAdapter:
    """Wraps ``google.genai.Client`` for Gemini inference."""

    def __init__(self, config: AppConfig, rate_limiter: object | None = None) -> None:
        self._cfg = config
        self._client = self._build_client()
        self._rate_limiter = rate_limiter

    def _build_client(self) -> Any:
        try:
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("google-genai is required. Run: uv sync") from exc

        return genai.Client(api_key=self._cfg.get_google_api_key())

    async def infer(self, item: DatasetItem) -> InferenceOutput:
        from google.genai import types  # type: ignore[import]

        inf = self._cfg.inference
        timeout_s = self._cfg.concurrency.request_timeout_s
        generation_config = types.GenerateContentConfig(
            temperature=inf.temperature,
            top_p=inf.top_p,
            max_output_tokens=inf.max_tokens,
        )

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=inf.model,
                    contents=item.prompt,
                    config=generation_config,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Inference timed out for instance_id=%s after %.1fs",
                item.instance_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            logger.error("Inference failed for instance_id=%s: %s", item.instance_id, exc)
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

        return InferenceOutput(
            instance_id=item.instance_id,
            model_id=inf.model,
            prompt=item.prompt,
            response=getattr(response, "text", "") or "",
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_s=round(latency, 3),
            finish_reason=finish_reason,
            run_id=self._cfg.run.run_id,
        )
