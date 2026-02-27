"""Azure OpenAI Batch API adapter (optional fast-path)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.schemas import DatasetItem, InferenceOutput

logger = logging.getLogger(__name__)

_BATCH_POLL_INTERVAL_S = 30


class AzureOpenAIBatchAdapter:
    """Uploads a JSONL batch, waits for completion, and downloads results."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import AzureOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for batch mode. Run: uv add openai"
            ) from exc

        api_key = self._cfg.get_batch_api_key()
        return AzureOpenAI(
            azure_endpoint=self._cfg.azure.batch.endpoint,
            api_key=api_key,
            api_version=self._cfg.azure.batch.api_version,
        )

    def _build_batch_jsonl(self, items: list[DatasetItem], tmp_path: Path) -> Path:
        """Serialise items into Azure OpenAI batch JSONL format."""
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        inf = self._cfg.inference
        batch_cfg = self._cfg.azure.batch
        with tmp_path.open("w", encoding="utf-8") as fh:
            for item in items:
                request = {
                    "custom_id": item.instance_id,
                    "method": "POST",
                    "url": batch_cfg.completion_endpoint,
                    "body": {
                        "model": inf.model,
                        "messages": [{"role": "user", "content": item.prompt}],
                        "temperature": inf.temperature,
                        "max_tokens": inf.max_tokens,
                        "top_p": inf.top_p,
                    },
                }
                fh.write(json.dumps(request) + "\n")
        logger.info("Wrote %d batch requests to %s", len(items), tmp_path)
        return tmp_path

    def run_batch(
        self,
        items: list[DatasetItem],
        run_dir: Path,
    ) -> list[InferenceOutput]:
        """Upload, poll, and download a batch.  Blocking."""
        batch_input_path = run_dir / "batch_input.jsonl"
        self._build_batch_jsonl(items, batch_input_path)

        batch_cfg = self._cfg.azure.batch

        # Upload file
        with batch_input_path.open("rb") as fh:
            file_obj = self._client.files.create(file=fh, purpose="batch")
        logger.info("Uploaded batch file: %s", file_obj.id)

        # Create batch
        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint=batch_cfg.completion_endpoint,
            completion_window=self._cfg.concurrency.batch_completion_window,
        )
        logger.info("Batch created: %s (status=%s)", batch.id, batch.status)

        # Poll
        poll_start = time.monotonic()
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            if time.monotonic() - poll_start > self._cfg.concurrency.max_poll_time_s:
                raise TimeoutError(
                    f"Batch {batch.id} polling exceeded "
                    f"{self._cfg.concurrency.max_poll_time_s}s timeout "
                    f"(last status: {batch.status})"
                )
            time.sleep(_BATCH_POLL_INTERVAL_S)
            batch = self._client.batches.retrieve(batch.id)
            logger.info("Batch %s status: %s", batch.id, batch.status)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

        # Download results
        result_content = self._client.files.content(batch.output_file_id)
        outputs: list[InferenceOutput] = []
        item_map = {i.instance_id: i for i in items}

        for line in result_content.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            body = obj.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            usage = body.get("usage", {})
            content = choices[0]["message"]["content"] if choices else ""
            original_item = item_map.get(custom_id)
            outputs.append(
                InferenceOutput(
                    instance_id=custom_id,
                    model_id=self._cfg.inference.model,
                    prompt=original_item.prompt if original_item else "",
                    response=content,
                    tokens_prompt=usage.get("prompt_tokens"),
                    tokens_completion=usage.get("completion_tokens"),
                    finish_reason=choices[0].get("finish_reason") if choices else None,
                    run_id=self._cfg.run.run_id,
                )
            )

        logger.info("Batch completed: %d results downloaded.", len(outputs))
        return outputs
