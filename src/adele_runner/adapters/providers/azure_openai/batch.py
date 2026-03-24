"""Azure OpenAI batch adapter."""

from __future__ import annotations

import json
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

_BATCH_POLL_INTERVAL_S = 30


class AzureOpenAIBatchAdapter:
    """Uploads a JSONL batch, waits for completion, and returns normalized responses."""

    capabilities = AdapterCapabilities(request_response=False, batch=True)

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import AzureOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("openai package is required for batch mode. Run: uv sync") from exc

        return AzureOpenAI(
            azure_endpoint=self._cfg.providers.azure_openai.endpoint,
            api_key=self._cfg.get_provider_api_key("azure_openai"),
            api_version=self._cfg.providers.azure_openai.api_version,
        )

    def _build_batch_lines(self, requests: list[ChatRequest]) -> list[str]:
        batch_cfg = self._cfg.providers.azure_openai
        lines: list[str] = []
        for request in requests:
            body = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
            }
            if request.max_tokens is not None:
                body["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                body["top_p"] = request.top_p
            lines.append(
                json.dumps(
                    {
                        "custom_id": request.request_id,
                        "method": "POST",
                        "url": batch_cfg.completion_endpoint,
                        "body": body,
                    }
                )
            )
        return lines

    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        from adele_runner.utils.batch_split import split_batch_requests

        jsonl_lines = self._build_batch_lines(requests)
        batch_cfg = self._cfg.providers.azure_openai
        chunks = split_batch_requests(
            jsonl_lines,
            max_requests=batch_cfg.max_requests_per_file,
            max_bytes=batch_cfg.max_bytes_per_file,
        )
        request_map = {request.request_id: request for request in requests}
        all_outputs: list[ChatResponse] = []

        for chunk_idx, chunk in enumerate(chunks):
            suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
            input_path = run_dir / f"batch_input{suffix}.jsonl"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with input_path.open("w", encoding="utf-8") as fh:
                for line in chunk:
                    fh.write(line + "\n")

            chunk_outputs = self._submit_and_poll(input_path, request_map, settings)
            all_outputs.extend(chunk_outputs)

        logger.info("Batch completed: %d total results.", len(all_outputs))
        return all_outputs

    def _submit_and_poll(
        self,
        input_path: Path,
        request_map: dict[str, ChatRequest],
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        batch_cfg = self._cfg.providers.azure_openai

        with input_path.open("rb") as fh:
            file_obj = self._client.files.create(file=fh, purpose="batch")

        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint=batch_cfg.completion_endpoint,
            completion_window=settings.batch_completion_window,
        )

        poll_start = time.monotonic()
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(
                    f"Batch {batch.id} polling exceeded {settings.max_poll_time_s}s timeout "
                    f"(last status: {batch.status})"
                )
            time.sleep(_BATCH_POLL_INTERVAL_S)
            batch = self._client.batches.retrieve(batch.id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

        result_content = self._client.files.content(batch.output_file_id)
        outputs: list[ChatResponse] = []
        for line in result_content.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            body = obj.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            usage = body.get("usage", {})
            content = choices[0]["message"]["content"] if choices else ""
            request = request_map.get(custom_id)
            outputs.append(
                ChatResponse(
                    request_id=custom_id,
                    content=content,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=choices[0].get("finish_reason") if choices else None,
                    raw_output=content,
                    metadata=dict(request.metadata) if request else {},
                )
            )
        return outputs
