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

    def _build_batch_lines(self, items: list[DatasetItem]) -> list[str]:
        """Serialise items into Azure OpenAI batch JSONL lines."""
        inf = self._cfg.inference
        batch_cfg = self._cfg.azure.batch
        lines: list[str] = []
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
            lines.append(json.dumps(request))
        return lines

    def run_batch(
        self,
        items: list[DatasetItem],
        run_dir: Path,
    ) -> list[InferenceOutput]:
        """Upload, poll, and download a batch.  Blocking.

        Large request sets are automatically split into chunks that respect
        Azure Batch API limits (request count and file size).
        """
        from adele_runner.utils.batch_split import split_batch_requests

        jsonl_lines = self._build_batch_lines(items)
        batch_cfg = self._cfg.azure.batch
        chunks = split_batch_requests(
            jsonl_lines,
            max_requests=batch_cfg.max_requests_per_file,
            max_bytes=batch_cfg.max_bytes_per_file,
        )
        item_map = {i.instance_id: i for i in items}

        all_outputs: list[InferenceOutput] = []
        for chunk_idx, chunk in enumerate(chunks):
            suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
            input_path = run_dir / f"batch_input{suffix}.jsonl"
            input_path.parent.mkdir(parents=True, exist_ok=True)

            with input_path.open("w", encoding="utf-8") as fh:
                for line in chunk:
                    fh.write(line + "\n")
            logger.info(
                "Batch chunk %d/%d: wrote %d requests to %s",
                chunk_idx + 1,
                len(chunks),
                len(chunk),
                input_path,
            )

            chunk_outputs = self._submit_and_poll(input_path, item_map, chunk_idx)
            all_outputs.extend(chunk_outputs)

        logger.info("Batch completed: %d total results.", len(all_outputs))
        return all_outputs

    def _submit_and_poll(
        self,
        input_path: Path,
        item_map: dict[str, DatasetItem],
        chunk_idx: int,
    ) -> list[InferenceOutput]:
        """Upload, create batch, poll, and download results for a single chunk."""
        batch_cfg = self._cfg.azure.batch

        # Upload file
        with input_path.open("rb") as fh:
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

        return outputs
