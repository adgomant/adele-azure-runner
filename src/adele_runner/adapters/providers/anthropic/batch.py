"""Anthropic message batches adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
from adele_runner.utils.retry import make_retry_decorator

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 60
_CUSTOM_ID_MAX_LEN = 64
_CUSTOM_ID_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class AnthropicBatchAdapter:
    """Use Anthropic Message Batches as a normalized batch transport."""

    capabilities = AdapterCapabilities(request_response=False, batch=True)
    poll_interval_s = _POLL_INTERVAL_S

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
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("anthropic package is required for anthropic mode. Run: uv sync") from exc

        kwargs: dict[str, Any] = {"api_key": self._cfg.get_provider_api_key("anthropic")}
        if self._cfg.providers.anthropic.base_url:
            kwargs["base_url"] = self._cfg.providers.anthropic.base_url
        return anthropic.Anthropic(**kwargs)

    def run_batch(
        self,
        requests: list[ChatRequest],
        run_dir: Path,
        settings: ExecutionSettings,
        on_chunk_completed: Callable[[list[ChatResponse]], None] | None = None,
    ) -> list[ChatResponse]:
        chunks = self.split_requests(requests, settings)
        outputs: list[ChatResponse] = []
        batch_budget = settings.batch_budget
        submit_interval_s = 0.0
        if batch_budget is not None and batch_budget.batch_requests_per_minute:
            submit_interval_s = 60.0 / batch_budget.batch_requests_per_minute
        last_submit_at = 0.0

        for chunk in chunks:
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
                    provider="anthropic",
                    request_ids=[request.request_id for request in chunk],
                ),
            )
            submission = self._wait_for_completion(submission, settings)
            chunk_outputs = self.fetch_results(submission, chunk, settings)
            if on_chunk_completed is not None:
                on_chunk_completed(chunk_outputs)
            outputs.extend(chunk_outputs)

        return outputs

    def split_requests(
        self,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
    ) -> list[list[ChatRequest]]:
        batch_budget = settings.batch_budget
        return split_items_by_limits(
            requests,
            size_getter=self._estimate_request_bytes,
            max_items=(
                min(
                    batch_budget.max_requests_per_batch or len(requests),
                    batch_budget.batch_queue_requests or len(requests),
                )
                if batch_budget
                else None
            ),
            max_bytes=batch_budget.max_bytes_per_batch if batch_budget else None,
            max_tokens=batch_budget.batch_enqueued_tokens if batch_budget else None,
            token_getter=self._estimate_request_tokens,
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
        _ = run_dir
        batch = self._call_with_retry(
            settings,
            self._client.messages.batches.create,
            requests=[self._to_batch_request(request) for request in requests],
        )
        now = datetime.utcnow()
        status = getattr(batch, "processing_status", None)
        return BatchJobRecord(
            run_id=run_id,
            stage=stage,
            provider="anthropic",
            judge_name=judge_name,
            chunk_id=chunk_id,
            remote_batch_id=str(getattr(batch, "id", "")),
            request_ids=[request.request_id for request in requests],
            request_count=len(requests),
            submitted_at=now,
            last_known_status=status,
            status_checked_at=now,
            completed_at=now if self._is_terminal_status(status) else None,
            is_terminal=self._is_terminal_status(status),
            is_successful=self._status_is_successful(status),
            terminal_error=(
                None
                if self._status_is_successful(status) in {None, True}
                else f"Anthropic batch ended with status={status}"
            ),
        )

    def refresh_submission(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        batch = self._call_with_retry(
            settings,
            self._client.messages.batches.retrieve,
            submission.remote_batch_id,
        )
        status = getattr(batch, "processing_status", None)
        now = datetime.utcnow()
        return submission.model_copy(
            update={
                "last_known_status": status,
                "status_checked_at": now,
                "completed_at": now if self._is_terminal_status(status) else None,
                "is_terminal": self._is_terminal_status(status),
                "is_successful": self._status_is_successful(status),
                "terminal_error": (
                    None
                    if self._status_is_successful(status) in {None, True}
                    else f"Anthropic batch ended with status={status}"
                ),
            }
        )

    def fetch_results(
        self,
        submission: BatchJobRecord,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        return self._extract_results(submission.remote_batch_id, requests, settings)

    def _to_batch_request(self, request: ChatRequest) -> dict[str, Any]:
        system_parts = [message.content for message in request.messages if message.role == "system"]
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "assistant" if message.role == "assistant" else "user",
                    "content": message.content,
                }
                for message in request.messages
                if message.role != "system"
            ],
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
        }
        if system_parts:
            params["system"] = [{"type": "text", "text": "\n\n".join(system_parts)}]
        if request.top_p is not None and "temperature" not in params:
            params["top_p"] = request.top_p

        return {
            "custom_id": self._batch_custom_id(request.request_id),
            "params": params,
        }

    @staticmethod
    def _batch_custom_id(request_id: str) -> str:
        sanitized = _CUSTOM_ID_ALLOWED_RE.sub("-", request_id).strip("-")
        if not sanitized:
            sanitized = "request"

        if len(sanitized) <= _CUSTOM_ID_MAX_LEN:
            return sanitized

        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
        prefix_len = _CUSTOM_ID_MAX_LEN - len(digest) - 1
        return f"{sanitized[:prefix_len].rstrip('-')}-{digest}"

    def _estimate_request_tokens(self, request: ChatRequest) -> int:
        prompt_chars = sum(len(message.content) for message in request.messages)
        return max(1, prompt_chars // 4) + max(1, request.max_tokens or 0)

    def _estimate_request_bytes(self, request: ChatRequest) -> int:
        return len(json.dumps(self._to_batch_request(request)).encode("utf-8"))

    def _extract_results(
        self,
        batch_id: str,
        requests: list[ChatRequest],
        settings: ExecutionSettings,
    ) -> list[ChatResponse]:
        outputs: list[ChatResponse] = []
        request_map = {
            self._batch_custom_id(request.request_id): request for request in requests
        }
        results_iter = self._call_with_retry(
            settings,
            self._client.messages.batches.results,
            batch_id,
        )
        for result in results_iter:
            custom_id = getattr(result, "custom_id", None)
            result_body = getattr(result, "result", None)
            result_type = getattr(result_body, "type", None)
            message = getattr(result_body, "message", None)
            content_blocks = getattr(message, "content", None) or []
            text = "".join(getattr(block, "text", "") for block in content_blocks)
            usage = getattr(message, "usage", None)
            request = request_map.get(custom_id)
            raw_output = (
                result.model_dump_json()
                if hasattr(result, "model_dump_json")
                else json.dumps(result, default=str)
            )
            metadata = dict(request.metadata) if request else {}
            if result_type is not None:
                metadata["anthropic_batch_result_type"] = result_type
            if result_type != "succeeded":
                logger.warning(
                    "Anthropic batch request %s returned result_type=%s",
                    custom_id,
                    result_type,
                )
            outputs.append(
                ChatResponse(
                    request_id=request.request_id if request else str(custom_id),
                    content=text,
                    prompt_tokens=getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "output_tokens", None),
                    finish_reason=getattr(message, "stop_reason", None),
                    raw_output=raw_output,
                    metadata=metadata,
                )
            )
        return outputs

    def _wait_for_completion(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        last_logged_status = submission.last_known_status
        logger.info(
            "Anthropic batch %s created for %d requests; status=%s",
            submission.remote_batch_id,
            submission.request_count,
            last_logged_status,
        )
        poll_start = time.monotonic()
        current = submission
        while not current.is_terminal:
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(f"Anthropic batch polling exceeded {settings.max_poll_time_s}s.")
            time.sleep(self.poll_interval_s)
            current = self.refresh_submission(current, settings)
            if current.last_known_status != last_logged_status:
                logger.info(
                    "Anthropic batch %s status=%s",
                    current.remote_batch_id,
                    current.last_known_status,
                )
                last_logged_status = current.last_known_status
        logger.info(
            "Anthropic batch %s finished with status=%s",
            current.remote_batch_id,
            current.last_known_status,
        )
        return current

    @staticmethod
    def _is_terminal_status(status: Any) -> bool:
        return str(status) in {"ended", "completed", "errored", "failed", "expired", "cancelled", "canceled"}

    @staticmethod
    def _is_success_status(status: Any) -> bool:
        return str(status) in {"ended", "completed"}

    @classmethod
    def _status_is_successful(cls, status: Any) -> bool | None:
        status_str = str(status)
        if not status_str or status_str == "None":
            return None
        return cls._is_success_status(status)

    @staticmethod
    def _make_retry(settings: ExecutionSettings):
        return make_retry_decorator(
            max_retries=settings.max_retries,
            backoff_base=settings.backoff_base_s,
            backoff_max=settings.backoff_max_s,
        )

    def _call_with_retry(
        self,
        settings: ExecutionSettings,
        func,
        *args,
        **kwargs,
    ):
        retry_decorator = self._make_retry(settings)

        @retry_decorator
        def _call():
            return func(*args, **kwargs)

        return _call()
