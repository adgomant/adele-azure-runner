"""Google Gemini batch adapter."""

from __future__ import annotations

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

_POLL_INTERVAL_S = 10


class GoogleGenAIBatchAdapter:
    """Use the Gemini Batch API via google-genai."""

    capabilities = AdapterCapabilities(request_response=False, batch=True)

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._client = self._build_client()

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
    ) -> list[ChatResponse]:
        _ = run_dir
        job = self._create_batch_job(requests)
        poll_start = time.monotonic()

        while True:
            state = self._status_name(job)
            if state in {"SUCCEEDED", "COMPLETED"}:
                break
            if state in {"FAILED", "CANCELLED", "EXPIRED"}:
                raise RuntimeError(f"Google batch job ended with status: {state}")
            if time.monotonic() - poll_start > settings.max_poll_time_s:
                raise TimeoutError(f"Google batch polling exceeded {settings.max_poll_time_s}s.")
            time.sleep(_POLL_INTERVAL_S)
            job = self._refresh_batch_job(job)

        return self._extract_results(job, requests)

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
                    raw_output=text or "",
                    metadata=dict(request.metadata) if request else {},
                )
            )
        return outputs
