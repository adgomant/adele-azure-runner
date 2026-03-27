"""Google Gemini batch adapter."""

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

_POLL_INTERVAL_S = 10


class GoogleGenAIBatchAdapter:
    """Use the Gemini Batch API via google-genai."""

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
        return genai.Client(api_key=self._cfg.get_provider_api_key("google_genai"))

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
                    provider="google_genai",
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
            token_getter=self._estimate_request_tokens,
            max_items=batch_budget.batch_queue_requests if batch_budget else None,
            max_bytes=batch_budget.max_bytes_per_batch if batch_budget else None,
            max_tokens=batch_budget.batch_enqueued_tokens if batch_budget else None,
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
        _ = run_dir, settings
        job = self._create_batch_job(requests)
        status = self._status_name(job)
        remote_batch_id = str(getattr(job, "name", None) or getattr(job, "id", ""))
        now = datetime.utcnow()
        return BatchJobRecord(
            run_id=run_id,
            stage=stage,
            provider="google_genai",
            judge_name=judge_name,
            chunk_id=chunk_id,
            remote_batch_id=remote_batch_id,
            request_ids=[request.request_id for request in requests],
            request_count=len(requests),
            submitted_at=now,
            last_known_status=status,
            status_checked_at=now,
            completed_at=now if self._is_terminal_status(status) else None,
            provider_metadata={"job_name": remote_batch_id},
            is_terminal=self._is_terminal_status(status),
            is_successful=self._status_is_successful(status),
            terminal_error=(
                None
                if self._status_is_successful(status) in {None, True}
                else f"Google batch job ended with status={status}"
            ),
        )

    def refresh_submission(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        _ = settings
        job = self._client.batches.get(name=submission.remote_batch_id)
        status = self._status_name(job)
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
                    else f"Google batch job ended with status={status}"
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
        job = self._client.batches.get(name=submission.remote_batch_id)
        return self._extract_results(job, requests)

    def _estimate_request_tokens(self, request: ChatRequest) -> int:
        prompt_chars = sum(len(message.content) for message in request.messages)
        return max(1, prompt_chars // 4) + max(1, request.max_tokens or 0)

    def _estimate_request_bytes(self, request: ChatRequest) -> int:
        payload = {
            "key": request.request_id,
            "request": {
                "contents": [
                    {"role": message.role, "parts": [{"text": message.content}]}
                    for message in request.messages
                ],
                "generationConfig": {
                    "temperature": request.temperature,
                    "topP": request.top_p,
                    "maxOutputTokens": request.max_tokens,
                },
            },
        }
        return len(json.dumps(payload).encode("utf-8"))

    def _create_batch_job(self, requests: list[ChatRequest]) -> Any:
        payload = [
            {
                "key": request.request_id,
                "request": {
                    "contents": [
                        {
                            "role": message.role,
                            "parts": [{"text": message.content}],
                        }
                        for message in request.messages
                    ],
                    "generationConfig": {
                        "temperature": request.temperature,
                        "topP": request.top_p,
                        "maxOutputTokens": request.max_tokens,
                    },
                },
            }
            for request in requests
        ]
        return self._client.batches.create(model=requests[0].model, src=payload)

    def _refresh_batch_job(self, job: Any) -> Any:
        job_name = getattr(job, "name", None) or getattr(job, "id", None)
        return self._client.batches.get(name=job_name)

    def _status_name(self, job: Any) -> str:
        state = getattr(job, "state", None) or getattr(job, "status", None)
        if state is None:
            return "UNKNOWN"
        return str(getattr(state, "name", state))

    def _extract_results(
        self,
        job: Any,
        requests: list[ChatRequest],
    ) -> list[ChatResponse]:
        request_map = {request.request_id: request for request in requests}
        results_iter = getattr(self._client.batches, "results", None)
        if callable(results_iter):
            raw_results = list(results_iter(job))
        else:
            raw_results = list(getattr(job, "dest", []) or [])

        outputs: list[ChatResponse] = []
        for result in raw_results:
            key = getattr(result, "key", None) or result.get("key")
            response = getattr(result, "response", None) or result.get("response", {})
            text = getattr(response, "text", None) or response.get("text", "")
            usage = getattr(response, "usage_metadata", None) or response.get("usage_metadata", {})
            request = request_map.get(key)
            outputs.append(
                ChatResponse(
                    request_id=str(key),
                    content=text or "",
                    prompt_tokens=getattr(usage, "prompt_token_count", None)
                    or usage.get("prompt_token_count"),
                    completion_tokens=getattr(usage, "candidates_token_count", None)
                    or usage.get("candidates_token_count"),
                    raw_output=response,
                    metadata=dict(request.metadata) if request else {},
                )
            )
        return outputs

    def _wait_for_completion(
        self,
        submission: BatchJobRecord,
        settings: ExecutionSettings,
    ) -> BatchJobRecord:
        poll_start = time.monotonic()
        current = submission
        while not current.is_terminal:
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(f"Google batch polling exceeded {settings.max_poll_time_s}s.")
            time.sleep(self.poll_interval_s)
            current = self.refresh_submission(current, settings)
        if current.is_successful is False:
            raise RuntimeError(f"Google batch job ended with status: {current.last_known_status}")
        return current

    @staticmethod
    def _is_terminal_status(status: Any) -> bool:
        return str(status) in {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "EXPIRED"}

    @staticmethod
    def _is_success_status(status: Any) -> bool:
        return str(status) in {"SUCCEEDED", "COMPLETED"}

    @classmethod
    def _status_is_successful(cls, status: Any) -> bool | None:
        status_str = str(status)
        if not status_str or status_str == "UNKNOWN":
            return None
        return cls._is_success_status(status)
