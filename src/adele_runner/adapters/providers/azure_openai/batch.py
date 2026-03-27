"""Azure OpenAI batch adapter."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.batch_jobs import BatchJobRecord, BatchStage, make_chunk_id
from adele_runner.runtime.types import (
    AdapterCapabilities,
    ChatRequest,
    ChatResponse,
    ExecutionSettings,
)
from adele_runner.utils.batch_split import split_items_by_limits

logger = logging.getLogger(__name__)

_BATCH_POLL_INTERVAL_S = 30


class AzureOpenAIBatchAdapter:
    """Uploads a JSONL batch, waits for completion, and returns normalized responses."""

    capabilities = AdapterCapabilities(request_response=False, batch=True)
    poll_interval_s = _BATCH_POLL_INTERVAL_S

    def __init__(
        self,
        config: AppConfig,
        budget_tracker: object | None = None,
    ) -> None:
        self._cfg = config
        self._client = self._build_client()
        self._budget_tracker = budget_tracker

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

    def _estimate_request_tokens(self, request: ChatRequest) -> int:
        prompt_chars = sum(len(message.content) for message in request.messages)
        prompt_estimate = max(1, prompt_chars // 4)
        completion_estimate = max(1, request.max_tokens or 0)
        return prompt_estimate + completion_estimate

    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
        on_chunk_completed: Callable[[list[ChatResponse]], None] | None = None,
    ) -> list[ChatResponse]:
        chunks = self.split_requests(requests, settings)
        all_outputs: list[ChatResponse] = []
        batch_budget = settings.batch_budget
        submit_interval_s = 0.0
        if batch_budget is not None and batch_budget.batch_requests_per_minute:
            submit_interval_s = 60.0 / batch_budget.batch_requests_per_minute
        last_submit_at = 0.0

        for chunk_idx, chunk in enumerate(chunks):
            if self._budget_tracker is not None:
                self._budget_tracker.can_submit_batch_chunk(chunk)  # type: ignore[attr-defined]
            if submit_interval_s > 0 and last_submit_at > 0:
                elapsed = time.monotonic() - last_submit_at
                if elapsed < submit_interval_s:
                    time.sleep(submit_interval_s - elapsed)
            last_submit_at = time.monotonic()
            submission = self.submit_chunk(
                chunk,
                run_dir,
                settings,
                stage="inference",
                run_id=self._cfg.run.run_id,
                chunk_id=make_chunk_id(
                    stage="inference",
                    provider="azure_openai",
                    request_ids=[request.request_id for request in chunk],
                ),
            )
            submission = self._wait_for_completion(submission, settings)
            chunk_outputs = self.fetch_results(submission, chunk, settings)
            if on_chunk_completed is not None:
                on_chunk_completed(chunk_outputs)
            all_outputs.extend(chunk_outputs)

        logger.info("Batch completed: %d total results.", len(all_outputs))
        return all_outputs

    def split_requests(
        self,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
    ) -> list[list[ChatRequest]]:
        batch_cfg = self._cfg.providers.azure_openai
        batch_budget = settings.batch_budget
        max_requests = batch_cfg.max_requests_per_file
        max_bytes = batch_cfg.max_bytes_per_file
        max_tokens = None
        if batch_budget is not None:
            if batch_budget.max_requests_per_batch is not None:
                max_requests = min(max_requests, batch_budget.max_requests_per_batch)
            if batch_budget.max_bytes_per_batch is not None:
                max_bytes = min(max_bytes, batch_budget.max_bytes_per_batch)
            max_tokens = batch_budget.batch_enqueued_tokens
        return split_items_by_limits(
            requests,
            size_getter=lambda request: len(
                self._build_batch_lines([request])[0].encode("utf-8")
            )
            + 1,
            token_getter=self._estimate_request_tokens,
            max_items=max_requests,
            max_bytes=max_bytes,
            max_tokens=max_tokens,
        )

    def submit_chunk(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
        *,
        stage: BatchStage,
        run_id: str,
        chunk_id: str,
        judge_name: str | None = None,
    ) -> BatchJobRecord:
        input_path = run_dir / f"{chunk_id}.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("w", encoding="utf-8") as fh:
            for line in self._build_batch_lines(requests):
                fh.write(line + "\n")

        with input_path.open("rb") as fh:
            file_obj = self._client.files.create(file=fh, purpose="batch")

        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint=self._cfg.providers.azure_openai.completion_endpoint,
            completion_window=settings.batch_completion_window,
        )
        now = datetime.utcnow()
        status = getattr(batch, "status", None)
        return BatchJobRecord(
            run_id=run_id,
            stage=stage,
            provider="azure_openai",
            judge_name=judge_name,
            chunk_id=chunk_id,
            remote_batch_id=str(getattr(batch, "id", "")),
            request_ids=[request.request_id for request in requests],
            request_count=len(requests),
            submitted_at=now,
            last_known_status=status,
            status_checked_at=now,
            completed_at=now if self._is_terminal_status(status) else None,
            input_artifact_path=str(input_path),
            provider_metadata={
                "input_file_id": getattr(file_obj, "id", None),
                "output_file_id": getattr(batch, "output_file_id", None),
            },
            is_terminal=self._is_terminal_status(status),
            is_successful=self._status_is_successful(status),
            terminal_error=(
                None
                if self._status_is_successful(status) in {None, True}
                else f"Azure OpenAI batch ended with status={status}"
            ),
        )

    def refresh_submission(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        _ = settings
        batch = self._client.batches.retrieve(submission.remote_batch_id)
        status = getattr(batch, "status", None)
        provider_metadata = dict(submission.provider_metadata)
        provider_metadata["output_file_id"] = getattr(batch, "output_file_id", None)
        now = datetime.utcnow()
        return submission.model_copy(
            update={
                "last_known_status": status,
                "status_checked_at": now,
                "completed_at": now if self._is_terminal_status(status) else None,
                "provider_metadata": provider_metadata,
                "is_terminal": self._is_terminal_status(status),
                "is_successful": self._status_is_successful(status),
                "terminal_error": (
                    None
                    if self._status_is_successful(status) in {None, True}
                    else f"Azure OpenAI batch ended with status={status}"
                ),
            }
        )

    def fetch_results(
        self,
        submission: BatchJobRecord,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        _ = settings
        output_file_id = submission.provider_metadata.get("output_file_id")
        if not output_file_id:
            batch = self._client.batches.retrieve(submission.remote_batch_id)
            output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError(f"Batch {submission.remote_batch_id} does not have an output_file_id.")

        result_content = self._client.files.content(output_file_id)
        request_map = {request.request_id: request for request in requests}
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
                    raw_output=obj,
                    metadata=dict(request.metadata) if request else {},
                )
            )
        return outputs

    def _wait_for_completion(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        current = submission
        last_logged_status = current.last_known_status
        poll_start = time.monotonic()
        while not current.is_terminal:
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(
                    f"Batch {current.remote_batch_id} polling exceeded {settings.max_poll_time_s}s timeout "
                    f"(last status: {current.last_known_status})"
                )
            time.sleep(self.poll_interval_s)
            current = self.refresh_submission(current, settings)
            if current.last_known_status != last_logged_status:
                logger.info(
                    "Azure OpenAI batch %s status=%s",
                    current.remote_batch_id,
                    current.last_known_status,
                )
                last_logged_status = current.last_known_status
        return current

    @staticmethod
    def _is_terminal_status(status: Any) -> bool:
        return str(status) in {"completed", "failed", "expired", "cancelled", "canceled"}

    @staticmethod
    def _is_success_status(status: Any) -> bool:
        return str(status) == "completed"

    @classmethod
    def _status_is_successful(cls, status: Any) -> bool | None:
        status_str = str(status)
        if not status_str or status_str == "None":
            return None
        return cls._is_success_status(status)
