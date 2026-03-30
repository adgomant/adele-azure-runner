"""Judge pipeline orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from adele_runner.config import AppConfig
from adele_runner.runtime.batch_jobs import (
    BatchJobRecord,
    append_batch_job_record,
    latest_batch_job_records,
    make_chunk_id,
)
from adele_runner.runtime.budgeting import BudgetTracker
from adele_runner.runtime.executors import (
    RequestResponseExecutor,
    ResumableBatchTransport,
    create_rate_limiter,
)
from adele_runner.runtime.resolution import resolve_judge_plans
from adele_runner.runtime.types import ChatRequest, ChatResponse, ExecutionSettings, ResolvedJudgePlan, ResolvedJudgeTarget
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.stages.judging import (
    build_failed_judge_output,
    build_judge_output,
    build_judge_request,
)
from adele_runner.utils.io import append_jsonl, ensure_run_dir, latest_jsonl_by_key
from adele_runner.utils.retry import is_retryable

logger = logging.getLogger(__name__)


def _build_judge_budget_tracker(config: AppConfig, judge_name: str) -> BudgetTracker | None:
    judge_cfg = next((judge for judge in config.judging.judges if judge.name == judge_name), None)
    if judge_cfg is None or judge_cfg.budget_usd is None:
        return None
    model_pricing = config.pricing.get_model_pricing(judge_name)
    if model_pricing is None:
        raise ValueError(f"No pricing configured for judge '{judge_name}'.")
    return BudgetTracker(
        lane_name=f"judge:{judge_name}",
        pricing_key=judge_name,
        budget_usd=judge_cfg.budget_usd,
        model_pricing=model_pricing,
    )


async def _run_request_response_judges(
    config: AppConfig,
    plans: list[ResolvedJudgePlan],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    judge_path,
) -> list[JudgeOutput]:  # noqa: ANN001
    if not plans:
        return []

    outputs: list[JudgeOutput] = []
    for plan in plans:
        target = plan.target
        binding = plan.binding
        settings = plan.settings
        budget_tracker = _build_judge_budget_tracker(config, target.judge_name)
        rate_limiter = create_rate_limiter(settings)
        adapter = binding.create_adapter(
            config,
            rate_limiter=rate_limiter,
            configured_rate_limits=target.rate_limits,
        )

        requests = []
        request_meta: dict[str, tuple[ResolvedJudgeTarget, InferenceOutput, str]] = {}
        for inference_output in inference_outputs:
            key = (inference_output.instance_id, inference_output.model_id, target.judge_name)
            if key in done:
                continue
            ground_truth = ground_truths.get(inference_output.instance_id, "")
            if inference_output.response is None or inference_output.status != "success":
                _, judge_prompt = build_judge_request(inference_output, ground_truth, target)
                output = build_failed_judge_output(
                    inference_output,
                    target,
                    judge_prompt,
                    status="skipped",
                    error_message="Inference unavailable.",
                    run_id=config.run.run_id,
                )
                append_jsonl(judge_path, output)
                outputs.append(output)
                continue
            request, judge_prompt = build_judge_request(inference_output, ground_truth, target)
            requests.append(request)
            request_meta[request.request_id] = (target, inference_output, judge_prompt)

        logger.info(
            "Request-response judge [%s] tasks pending: %d",
            target.judge_name,
            len(requests),
        )
        if not requests:
            continue

        def _record_response(
            response: ChatResponse | BaseException,
            _request_meta: dict[str, tuple[ResolvedJudgeTarget, InferenceOutput, str]] = request_meta,
            _budget_tracker: BudgetTracker | None = budget_tracker,
        ) -> None:
            if isinstance(response, BaseException):
                request_id = getattr(response, "request_id", None)
                meta = _request_meta.get(request_id) if request_id is not None else None
                if meta is None:
                    logger.error("Judge request failed without mappable request_id: %s", response)
                    return
                resolved_target, inference_output, judge_prompt = meta
                output = build_failed_judge_output(
                    inference_output,
                    resolved_target,
                    judge_prompt,
                    status="request_failed",
                    error_message=str(response),
                    run_id=config.run.run_id,
                )
                append_jsonl(judge_path, output)
                outputs.append(output)
                return
            meta = _request_meta.get(response.request_id)
            if meta is None:
                logger.warning("Judge response had unknown request_id=%s", response.request_id)
                return
            resolved_target, inference_output, judge_prompt = meta
            output = build_judge_output(
                inference_output,
                resolved_target,
                judge_prompt,
                response,
                config.run.run_id,
            )
            append_jsonl(judge_path, output)
            outputs.append(output)
            if _budget_tracker is not None:
                _budget_tracker.record_actual_usage(
                    prompt_tokens=output.tokens_prompt,
                    completion_tokens=output.tokens_completion,
                )

        await RequestResponseExecutor().execute(
            adapter=adapter,
            requests=requests,
            settings=settings,
            rate_limiter=rate_limiter,
            on_result=_record_response,
        )
    return outputs


async def _run_batch_judges(
    config: AppConfig,
    plans: list[ResolvedJudgePlan],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    run_dir,
    judge_path,
) -> list[JudgeOutput]:  # noqa: ANN001
    if not plans:
        return []

    all_outputs: list[JudgeOutput] = []
    batch_jobs_path = config.batch_jobs_path()

    for plan in plans:
        target = plan.target
        binding = plan.binding
        settings = plan.settings
        budget_tracker = _build_judge_budget_tracker(config, target.judge_name)
        adapter = binding.create_adapter(config, budget_tracker=budget_tracker)
        if not isinstance(adapter, ResumableBatchTransport):
            raise TypeError(
                f"Batch adapter for provider '{target.provider_kind}' does not support resume primitives."
            )

        all_requests: dict[str, ChatRequest] = {}
        pending_request_ids: set[str] = set()
        pending_request_order: list[str] = []
        request_meta: dict[str, tuple[InferenceOutput, str]] = {}
        for inference_output in inference_outputs:
            ground_truth = ground_truths.get(inference_output.instance_id, "")
            key = (inference_output.instance_id, inference_output.model_id, target.judge_name)
            if key in done:
                continue
            if inference_output.response is None or inference_output.status != "success":
                _, judge_prompt = build_judge_request(inference_output, ground_truth, target)
                output = build_failed_judge_output(
                    inference_output,
                    target,
                    judge_prompt,
                    status="skipped",
                    error_message="Inference unavailable.",
                    run_id=config.run.run_id,
                )
                append_jsonl(judge_path, output)
                all_outputs.append(output)
                continue
            request, judge_prompt = build_judge_request(inference_output, ground_truth, target)
            all_requests[request.request_id] = request
            request_meta[request.request_id] = (inference_output, judge_prompt)
            pending_request_ids.add(request.request_id)
            pending_request_order.append(request.request_id)

        if not pending_request_order:
            logger.info("Batch judge [%s]: nothing pending.", target.judge_name)
            continue

        logger.info(
            "Batch judge [%s]: %d tasks pending.",
            target.judge_name,
            len(pending_request_order),
        )

        def _record_response(
            response: ChatResponse | BaseException,
            _request_meta: dict[str, tuple[InferenceOutput, str]] = request_meta,
            _target: ResolvedJudgeTarget = target,
            _budget_tracker: BudgetTracker | None = budget_tracker,
            _pending_request_ids: set[str] = pending_request_ids,
        ) -> None:
            if isinstance(response, BaseException):
                return
            meta = _request_meta.get(response.request_id)
            if meta is None:
                logger.warning(
                    "Batch judge [%s]: unknown request_id %s",
                    _target.judge_name,
                    response.request_id,
                )
                return
            inference_output, judge_prompt = meta
            key = (inference_output.instance_id, inference_output.model_id, _target.judge_name)
            if key in done:
                return
            output = build_judge_output(
                inference_output,
                _target,
                judge_prompt,
                response,
                config.run.run_id,
            )
            append_jsonl(judge_path, output)
            done.add(key)
            all_outputs.append(output)
            _pending_request_ids.discard(response.request_id)
            if _budget_tracker is not None:
                _budget_tracker.record_actual_usage(
                    prompt_tokens=output.tokens_prompt,
                    completion_tokens=output.tokens_completion,
                )

        existing_jobs = sorted(
            [
                record
                for record in latest_batch_job_records(
                    batch_jobs_path,
                    run_id=config.run.run_id,
                    stage="judging",
                    provider=target.provider_kind,
                    judge_name=target.judge_name,
                )
                if record.needs_recovery
            ],
            key=lambda record: record.submitted_at or datetime.min,
        )
        logger.info("Recovered %d pending batch jobs from disk", len(existing_jobs))

        for record in existing_jobs:
            outstanding_request_ids = [request_id for request_id in record.request_ids if request_id in pending_request_ids]
            if not outstanding_request_ids:
                completed_record = record.model_copy(
                    update={"results_downloaded_at": record.results_downloaded_at or datetime.utcnow()}
                )
                append_batch_job_record(batch_jobs_path, completed_record)
                continue

            logger.info("Batch %s resumed; status=%s", record.remote_batch_id, record.last_known_status)
            try:
                current = await _poll_submission(adapter, record, settings, batch_jobs_path)
            except Exception as exc:  # noqa: BLE001
                if is_retryable(exc):
                    raise
                failed_record = record.model_copy(
                    update={
                        "last_known_status": "local_failure",
                        "status_checked_at": datetime.utcnow(),
                        "completed_at": datetime.utcnow(),
                        "terminal_error": str(exc),
                        "is_terminal": True,
                        "is_successful": False,
                    }
                )
                append_batch_job_record(batch_jobs_path, failed_record)
                logger.warning("Batch %s terminal failure; requests re-queued", record.remote_batch_id)
                continue

            if not current.is_successful:
                logger.warning("Batch %s terminal failure; requests re-queued", current.remote_batch_id)
                continue

            chunk_requests = [all_requests[request_id] for request_id in record.request_ids if request_id in all_requests]
            responses = await asyncio.to_thread(adapter.fetch_results, current, chunk_requests, settings)
            for response in responses:
                _record_response(response)
            downloaded_record = current.model_copy(update={"results_downloaded_at": datetime.utcnow()})
            append_batch_job_record(batch_jobs_path, downloaded_record)
            logger.info("Batch %s results downloaded: %d rows", current.remote_batch_id, len(responses))

        remaining_requests = [all_requests[request_id] for request_id in pending_request_order if request_id in pending_request_ids]
        if not remaining_requests:
            logger.info("Batch judge [%s]: all pending requests recovered from existing batches.", target.judge_name)
            continue

        chunks = adapter.split_requests(remaining_requests, settings)
        submit_interval_s = 0.0
        batch_budget = settings.batch_budget
        if batch_budget is not None and batch_budget.batch_requests_per_minute:
            submit_interval_s = 60.0 / batch_budget.batch_requests_per_minute
        last_submit_at = 0.0

        for chunk in chunks:
            if budget_tracker is not None:
                budget_tracker.can_submit_batch_chunk(chunk)  # type: ignore[attr-defined]
            if submit_interval_s > 0 and last_submit_at > 0:
                elapsed = time.monotonic() - last_submit_at
                if elapsed < submit_interval_s:
                    await asyncio.sleep(submit_interval_s - elapsed)
            chunk_id = make_chunk_id(
                stage="judging",
                provider=target.provider_kind,
                request_ids=[request.request_id for request in chunk],
                judge_name=target.judge_name,
            )
            submission = await asyncio.to_thread(
                adapter.submit_chunk,
                chunk,
                run_dir,
                settings,
                stage="judging",
                run_id=config.run.run_id,
                chunk_id=chunk_id,
                judge_name=target.judge_name,
            )
            append_batch_job_record(batch_jobs_path, submission)
            last_submit_at = time.monotonic()

            try:
                current = await _poll_submission(adapter, submission, settings, batch_jobs_path)
            except Exception as exc:  # noqa: BLE001
                if is_retryable(exc):
                    raise
                failed_record = submission.model_copy(
                    update={
                        "last_known_status": "local_failure",
                        "status_checked_at": datetime.utcnow(),
                        "completed_at": datetime.utcnow(),
                        "terminal_error": str(exc),
                        "is_terminal": True,
                        "is_successful": False,
                    }
                )
                append_batch_job_record(batch_jobs_path, failed_record)
                logger.warning("Batch %s terminal failure; requests re-queued", submission.remote_batch_id)
                continue

            if not current.is_successful:
                logger.warning("Batch %s terminal failure; requests re-queued", current.remote_batch_id)
                continue

            responses = await asyncio.to_thread(adapter.fetch_results, current, chunk, settings)
            for response in responses:
                _record_response(response)
            downloaded_record = current.model_copy(update={"results_downloaded_at": datetime.utcnow()})
            append_batch_job_record(batch_jobs_path, downloaded_record)
            logger.info("Batch %s results downloaded: %d rows", current.remote_batch_id, len(responses))

    return all_outputs


async def _poll_submission(
    adapter: ResumableBatchTransport,
    submission: BatchJobRecord,
    settings: ExecutionSettings,
    batch_jobs_path,
) -> BatchJobRecord:  # noqa: ANN001
    current = submission
    last_seen_status = current.last_known_status
    poll_start = time.monotonic()

    while not current.is_terminal:
        if time.monotonic() - poll_start > settings.max_poll_time_s:
            raise TimeoutError(
                f"Batch {current.remote_batch_id} polling exceeded {settings.max_poll_time_s}s timeout "
                f"(last status: {current.last_known_status})"
            )
        await asyncio.sleep(adapter.poll_interval_s)
        updated = await asyncio.to_thread(adapter.refresh_submission, current, settings)
        if (
            updated.last_known_status != current.last_known_status
            or updated.is_terminal != current.is_terminal
            or updated.terminal_error != current.terminal_error
        ):
            append_batch_job_record(batch_jobs_path, updated)
        current = updated
        if current.last_known_status != last_seen_status:
            logger.info("Batch %s resumed; status=%s", current.remote_batch_id, current.last_known_status)
            last_seen_status = current.last_known_status
    return current


async def run_judge(
    config: AppConfig,
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    *,
    force_run: bool = False,
) -> list[JudgeOutput]:
    """Run all configured judges over *inference_outputs*."""
    if not config.judging.enabled:
        logger.info("Judging is disabled in config.")
        return []

    run_dir = config.run_dir()
    ensure_run_dir(run_dir)
    judge_path = config.judge_outputs_path()

    latest_judges = latest_jsonl_by_key(judge_path, JudgeOutput, "instance_id", "model_id", "judge_name")
    done = set() if force_run else {key for key, output in latest_judges.items() if output.status == "success"}
    logger.info("Judge dedup index: %d entries.", len(done))

    plans = resolve_judge_plans(config)
    request_response_plans = [
        plan for plan in plans if plan.binding.execution_kind == "request_response"
    ]
    batch_plans = [plan for plan in plans if plan.binding.execution_kind == "batch"]

    request_response_coro = _run_request_response_judges(
        config,
        request_response_plans,
        inference_outputs,
        ground_truths,
        done,
        judge_path,
    )
    batch_coro = _run_batch_judges(
        config,
        batch_plans,
        inference_outputs,
        ground_truths,
        done,
        run_dir,
        judge_path,
    )

    request_response_outputs, batch_outputs = await asyncio.gather(
        request_response_coro,
        batch_coro,
    )
    return request_response_outputs + batch_outputs
