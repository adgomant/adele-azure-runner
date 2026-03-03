"""Azure AI Foundry inference adapter using azure-ai-inference."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.schemas import DatasetItem, InferenceOutput

logger = logging.getLogger(__name__)


class FoundryAdapter:
    """Wraps ChatCompletionsClient for a single Foundry-deployed model."""

    def __init__(self, config: AppConfig, rate_limiter: object | None = None) -> None:
        self._cfg = config
        self._client = self._build_client()
        self._rate_limit_warned = False
        self._rate_limiter = rate_limiter

    def _build_client(self) -> Any:
        try:
            from azure.ai.inference import ChatCompletionsClient  # type: ignore[import]
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("azure-ai-inference is required. Run: uv sync") from exc

        api_key = self._cfg.get_foundry_api_key()
        return ChatCompletionsClient(
            endpoint=self._cfg.azure.foundry.endpoint,
            credential=AzureKeyCredential(api_key),
        )

    def _on_response(self, response: Any) -> None:
        """Capture rate-limit headers from Azure pipeline responses."""
        try:
            headers = dict(response.http_response.headers)
        except AttributeError:
            return
        self._check_rate_limit_headers(headers)
        if self._rate_limiter is not None:
            self._rate_limiter.update_from_headers(headers)  # type: ignore[attr-defined]

    def _check_rate_limit_headers(self, headers: dict[str, str]) -> None:
        """Log a warning if actual rate limits differ significantly from config."""
        if self._rate_limit_warned:
            return

        rl = self._cfg.inference.rate_limits
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
                        "Actual TPM from Azure (%d) differs from config (%d) by >20%%",
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
                        "Actual RPM from Azure (%d) differs from config (%d) by >20%%",
                        actual_rpm,
                        rl.requests_per_minute,
                    )
            except (ValueError, TypeError):
                pass

    async def infer(self, item: DatasetItem) -> InferenceOutput:
        """Run a single inference call and return a structured output."""
        from azure.ai.inference.models import UserMessage  # type: ignore[import]

        inf = self._cfg.inference
        messages = [UserMessage(content=item.prompt)]

        timeout_s = self._cfg.concurrency.request_timeout_s

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.complete,
                    messages=messages,
                    model=inf.model,
                    temperature=inf.temperature,
                    max_tokens=inf.max_tokens,
                    top_p=inf.top_p,
                    raw_response_hook=self._on_response,
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
        choice = response.choices[0]
        usage = response.usage

        if self._rate_limiter is not None and usage:
            self._rate_limiter.update_token_usage(  # type: ignore[attr-defined]
                usage.prompt_tokens or 0, usage.completion_tokens or 0
            )

        return InferenceOutput(
            instance_id=item.instance_id,
            model_id=inf.model,
            prompt=item.prompt,
            response=choice.message.content or "",
            tokens_prompt=usage.prompt_tokens if usage else None,
            tokens_completion=usage.completion_tokens if usage else None,
            latency_s=round(latency, 3),
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            run_id=self._cfg.run.run_id,
        )
