"""Multi-judge evaluation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from adele_runner.config import AppConfig, JudgeConfig
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.concurrency import bounded_gather
from adele_runner.utils.io import append_jsonl, build_dedup_index, ensure_run_dir
from adele_runner.utils.retry import make_retry_decorator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_V2 = """\
TASK:
{question}

GROUND TRUTH:
{ground_truth}

Model Response:
{response}

You are an evaluation system that compares a MODEL RESPONSE to a GROUND TRUTH of a given TASK \
(as shown above), and rates the response on a scale of 1 to 5, where 5 means surely correct \
and 1 means surely incorrect. Your response must be concise, responding with exactly one \
integer score between 1 and 5 and nothing else that may be redundant.

score:
"""


_JUDGE_PROMPT_V1 = """\
You are an expert evaluator. Given the following question, reference answer, and model response, \
evaluate the model response.

Question:
{question}

Reference Answer:
{ground_truth}

Model Response:
{response}

Instructions:
- Score the response on a scale of 1-5 (1=completely wrong, 5=perfectly correct).
- Provide a verdict: "correct", "incorrect", "partial", or "unknown".
- Provide a brief reason (1-2 sentences).

Respond in JSON only, with exactly this structure:
{{"score": <int 1-5>, "verdict": "<correct|incorrect|partial|unknown>", "reason": "<string>"}}
"""


def build_judge_prompt(
    item_prompt: str,
    ground_truth: str,
    response: str,
    template: str = "v1",
) -> str:
    if template == "v1":
        return _JUDGE_PROMPT_V1.format(
            question=item_prompt,
            ground_truth=ground_truth,
            response=response,
        )
    elif template == "v2":
        return _JUDGE_PROMPT_V2.format(
            question=item_prompt,
            ground_truth=ground_truth,
            response=response,
        )
    raise ValueError(f"Unknown judge prompt template: {template}")


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------

_JSON_LIKE_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_VALID_VERDICTS = {"correct", "incorrect", "partial", "unknown"}


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse judge JSON output with repair fallback."""
    # 1) Direct parse
    try:
        obj = json.loads(raw.strip())
        return _validate_judge_obj(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) Extract first JSON-like block
    match = _JSON_LIKE_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group())
            return _validate_judge_obj(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) Regex fallback for score and verdict
    score_match = re.search(r'"score"\s*:\s*(\d)', raw)
    verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)

    score = int(score_match.group(1)) if score_match else 1
    verdict = verdict_match.group(1) if verdict_match else "unknown"
    if verdict not in _VALID_VERDICTS:
        verdict = "unknown"
    reason = reason_match.group(1) if reason_match else "Could not parse reason."

    logger.warning("Used regex fallback for judge output: score=%d verdict=%s", score, verdict)
    return {"score": max(1, min(5, score)), "verdict": verdict, "reason": reason}


def _validate_judge_obj(obj: dict[str, Any]) -> dict[str, Any]:
    score = int(obj["score"])
    if not (1 <= score <= 5):
        raise ValueError(f"Score out of range: {score}")
    verdict = str(obj.get("verdict", "unknown"))
    if verdict not in _VALID_VERDICTS:
        verdict = "unknown"
    reason = str(obj.get("reason", ""))
    return {"score": score, "verdict": verdict, "reason": reason}


# ---------------------------------------------------------------------------
# v2 bare-integer parsing
# ---------------------------------------------------------------------------


def parse_judge_v2(raw: str) -> dict[str, Any]:
    """Parse v2 judge output: expects a bare integer score 1-5."""
    # 1) Try direct integer parse
    stripped = raw.strip()
    try:
        score = int(stripped)
        score = max(1, min(5, score))
        return {
            "score": score,
            "verdict": "unknown",
            "reason": "Scored via v2 bare-integer prompt",
        }
    except ValueError:
        pass

    # 2) Fallback: regex for first integer 1-5
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        score = int(match.group(1))
        return {
            "score": score,
            "verdict": "unknown",
            "reason": "Scored via v2 bare-integer prompt",
        }

    # 3) Default to score=1
    logger.warning("Could not parse v2 judge output, defaulting to score=1: %r", raw[:200])
    return {
        "score": 1,
        "verdict": "unknown",
        "reason": "Scored via v2 bare-integer prompt",
    }


# ---------------------------------------------------------------------------
# Judge adapter — Foundry async (reuses Foundry client)
# ---------------------------------------------------------------------------


class JudgeAdapter:
    """Single-judge inference adapter (Azure AI Foundry, async)."""

    def __init__(self, judge_cfg: JudgeConfig, app_cfg: AppConfig) -> None:
        self._judge_cfg = judge_cfg
        self._app_cfg = app_cfg
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from azure.ai.inference import ChatCompletionsClient  # type: ignore[import]
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("azure-ai-inference is required.") from exc

        api_key = self._app_cfg.get_foundry_api_key()
        return ChatCompletionsClient(
            endpoint=self._app_cfg.azure.foundry.endpoint,
            credential=AzureKeyCredential(api_key),
        )

    async def judge(
        self,
        inference_output: InferenceOutput,
        ground_truth: str,
        judge_prompt: str,
        prompt_template: str = "v1",
    ) -> JudgeOutput:
        from azure.ai.inference.models import UserMessage  # type: ignore[import]

        timeout_s = self._app_cfg.concurrency.request_timeout_s

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.complete,
                    messages=[UserMessage(content=judge_prompt)],
                    model=self._judge_cfg.model,
                    temperature=0.0,
                    max_tokens=512,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Judge timed out for instance_id=%s judge=%s after %.1fs",
                inference_output.instance_id,
                self._judge_cfg.name,
                timeout_s,
            )
            raise
        raw = response.choices[0].message.content or ""

        parsed = parse_judge_v2(raw) if prompt_template == "v2" else parse_judge_json(raw)

        return JudgeOutput(
            instance_id=inference_output.instance_id,
            model_id=inference_output.model_id,
            judge_name=self._judge_cfg.name,
            score=parsed["score"],
            verdict=parsed["verdict"],
            reason=parsed["reason"],
            raw_output=raw,
            judge_prompt=judge_prompt,
            run_id=self._app_cfg.run.run_id,
        )


# ---------------------------------------------------------------------------
# Judge adapter — Azure OpenAI Batch
# ---------------------------------------------------------------------------

_BATCH_POLL_INTERVAL_S = 30


class JudgeBatchAdapter:
    """Submit judge prompts via the Azure OpenAI Batch API."""

    def __init__(self, judge_cfg: JudgeConfig, app_cfg: AppConfig) -> None:
        self._judge_cfg = judge_cfg
        self._app_cfg = app_cfg
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import AzureOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for batch judge mode. Run: uv add openai"
            ) from exc

        api_key = self._app_cfg.get_batch_api_key()
        return AzureOpenAI(
            azure_endpoint=self._app_cfg.azure.batch.endpoint,
            api_key=api_key,
            api_version=self._app_cfg.azure.batch.api_version,
        )

    def run_batch(
        self,
        requests: list[tuple[str, str]],
        run_dir: Path,
        max_poll_time_s: float = 3600.0,
    ) -> dict[str, str]:
        """Submit batch, poll, return {custom_id: raw_response_text}.

        *requests* is a list of ``(custom_id, prompt)`` tuples.
        *max_poll_time_s* is the maximum time to spend polling before raising TimeoutError.
        """
        input_path = run_dir / f"judge_batch_input_{self._judge_cfg.name}.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)

        batch_cfg = self._app_cfg.azure.batch

        with input_path.open("w", encoding="utf-8") as fh:
            for custom_id, prompt in requests:
                row = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": batch_cfg.completion_endpoint,
                    "body": {
                        "model": self._judge_cfg.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 512,
                    },
                }
                fh.write(json.dumps(row) + "\n")
        logger.info(
            "Judge batch [%s]: wrote %d requests to %s",
            self._judge_cfg.name,
            len(requests),
            input_path,
        )

        # Upload
        with input_path.open("rb") as fh:
            file_obj = self._client.files.create(file=fh, purpose="batch")
        logger.info("Judge batch [%s]: uploaded file %s", self._judge_cfg.name, file_obj.id)

        # Create batch
        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint=batch_cfg.completion_endpoint,
            completion_window=self._app_cfg.concurrency.batch_completion_window,
        )
        logger.info(
            "Judge batch [%s]: created %s (status=%s)",
            self._judge_cfg.name,
            batch.id,
            batch.status,
        )

        # Poll with timeout
        poll_start = time.monotonic()
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            elapsed = time.monotonic() - poll_start
            if elapsed >= max_poll_time_s:
                raise TimeoutError(
                    f"Judge batch [{self._judge_cfg.name}] {batch.id} polling timed out "
                    f"after {elapsed:.0f}s (limit: {max_poll_time_s:.0f}s). "
                    f"Last status: {batch.status}"
                )
            time.sleep(_BATCH_POLL_INTERVAL_S)
            batch = self._client.batches.retrieve(batch.id)
            logger.info(
                "Judge batch [%s] %s status: %s", self._judge_cfg.name, batch.id, batch.status
            )

        if batch.status != "completed":
            raise RuntimeError(
                f"Judge batch [{self._judge_cfg.name}] {batch.id} ended with status: {batch.status}"
            )

        # Download results
        result_content = self._client.files.content(batch.output_file_id)
        results: dict[str, str] = {}
        for line in result_content.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            body = obj.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            results[custom_id] = content

        logger.info("Judge batch [%s]: %d results downloaded.", self._judge_cfg.name, len(results))
        return results


# ---------------------------------------------------------------------------
# Internal: run foundry judges (async)
# ---------------------------------------------------------------------------


async def _run_foundry_judges(
    config: AppConfig,
    foundry_judges: list[JudgeConfig],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    judge_path: Path,
) -> list[JudgeOutput]:
    """Run foundry judges via async bounded concurrency."""
    retry_dec = make_retry_decorator(
        max_retries=config.concurrency.max_retries,
        backoff_base=config.concurrency.backoff_base_s,
        backoff_max=config.concurrency.backoff_max_s,
    )
    adapters = [JudgeAdapter(j, config) for j in foundry_judges]
    coroutines = []
    prompt_template = config.judging.prompt_template

    for inf_out in inference_outputs:
        gt = ground_truths.get(inf_out.instance_id, "")
        for adapter in adapters:
            key = (inf_out.instance_id, inf_out.model_id, adapter._judge_cfg.name)
            if key in done:
                continue
            judge_prompt = build_judge_prompt(
                item_prompt=inf_out.prompt,
                ground_truth=gt,
                response=inf_out.response,
                template=prompt_template,
            )

            @retry_dec
            async def _call(
                _adapter=adapter,
                _inf=inf_out,
                _gt=gt,
                _jp=judge_prompt,
                _tmpl=prompt_template,
            ) -> JudgeOutput:
                return await _adapter.judge(_inf, _gt, _jp, prompt_template=_tmpl)

            coroutines.append(_call())

    logger.info("Foundry judge tasks pending: %d", len(coroutines))
    if not coroutines:
        return []

    results = await bounded_gather(coroutines, max_concurrency=config.concurrency.max_in_flight)
    outputs: list[JudgeOutput] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.error("Judge task failed: %s", result)
            continue
        append_jsonl(judge_path, result)
        outputs.append(result)

    return outputs


# ---------------------------------------------------------------------------
# Internal: run batch judges
# ---------------------------------------------------------------------------


def _run_batch_judges(
    config: AppConfig,
    batch_judges: list[JudgeConfig],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    run_dir: Path,
    judge_path: Path,
) -> list[JudgeOutput]:
    """Run batch judges via Azure OpenAI Batch API (blocking)."""
    all_outputs: list[JudgeOutput] = []
    prompt_template = config.judging.prompt_template
    max_poll_time_s = config.concurrency.max_poll_time_s

    for judge_cfg in batch_judges:
        # Collect pending requests for this judge
        requests: list[tuple[str, str]] = []
        # Map custom_id -> (InferenceOutput, judge_prompt) for result assembly
        request_meta: dict[str, tuple[InferenceOutput, str]] = {}

        for inf_out in inference_outputs:
            key = (inf_out.instance_id, inf_out.model_id, judge_cfg.name)
            if key in done:
                continue
            gt = ground_truths.get(inf_out.instance_id, "")
            judge_prompt = build_judge_prompt(
                item_prompt=inf_out.prompt,
                ground_truth=gt,
                response=inf_out.response,
                template=prompt_template,
            )
            custom_id = f"{inf_out.instance_id}::{inf_out.model_id}"
            requests.append((custom_id, judge_prompt))
            request_meta[custom_id] = (inf_out, judge_prompt)

        if not requests:
            logger.info("Batch judge [%s]: nothing pending.", judge_cfg.name)
            continue

        logger.info("Batch judge [%s]: %d tasks pending.", judge_cfg.name, len(requests))
        adapter = JudgeBatchAdapter(judge_cfg, config)
        raw_results = adapter.run_batch(requests, run_dir, max_poll_time_s=max_poll_time_s)

        for custom_id, raw_text in raw_results.items():
            meta = request_meta.get(custom_id)
            if meta is None:
                logger.warning("Batch judge [%s]: unknown custom_id %s", judge_cfg.name, custom_id)
                continue
            inf_out, judge_prompt = meta

            if prompt_template == "v2":
                parsed = parse_judge_v2(raw_text)
            else:
                parsed = parse_judge_json(raw_text)

            output = JudgeOutput(
                instance_id=inf_out.instance_id,
                model_id=inf_out.model_id,
                judge_name=judge_cfg.name,
                score=parsed["score"],
                verdict=parsed["verdict"],
                reason=parsed["reason"],
                raw_output=raw_text,
                judge_prompt=judge_prompt,
                run_id=config.run.run_id,
            )
            append_jsonl(judge_path, output)
            all_outputs.append(output)

    return all_outputs


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_judge(
    config: AppConfig,
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
) -> list[JudgeOutput]:
    """Run all configured judges over *inference_outputs*."""
    if not config.judging.enabled:
        logger.info("Judging is disabled in config.")
        return []

    run_dir = config.run_dir()
    ensure_run_dir(run_dir)
    judge_path = config.judge_outputs_path()

    done = build_dedup_index(judge_path, "instance_id", "model_id", "judge_name")
    logger.info("Judge dedup index: %d entries.", len(done))

    # Split judges by provider
    foundry_judges = [j for j in config.judging.judges if j.provider == "foundry"]
    batch_judges = [j for j in config.judging.judges if j.provider == "batch"]

    all_outputs: list[JudgeOutput] = []

    # Run foundry and batch judges concurrently
    async def _noop() -> list[JudgeOutput]:
        return []

    foundry_coro = (
        _run_foundry_judges(
            config,
            foundry_judges,
            inference_outputs,
            ground_truths,
            done,
            judge_path,
        )
        if foundry_judges
        else _noop()
    )
    batch_coro = (
        asyncio.to_thread(
            _run_batch_judges,
            config,
            batch_judges,
            inference_outputs,
            ground_truths,
            done,
            run_dir,
            judge_path,
        )
        if batch_judges
        else _noop()
    )

    foundry_results, batch_results = await asyncio.gather(foundry_coro, batch_coro)
    all_outputs.extend(foundry_results)
    all_outputs.extend(batch_results)

    logger.info("Judging complete. %d outputs written to %s", len(all_outputs), judge_path)
    return all_outputs
