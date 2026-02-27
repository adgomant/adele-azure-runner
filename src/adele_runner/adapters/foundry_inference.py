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

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._client = self._build_client()

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
