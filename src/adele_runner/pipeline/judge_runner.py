"""Judge pipeline orchestration."""

from __future__ import annotations

import asyncio
import logging

from adele_runner.config import AppConfig
from adele_runner.runtime.executors import (
    BatchExecutor,
    RequestResponseExecutor,
    create_rate_limiter,
)
from adele_runner.runtime.resolution import (
    resolve_judge_binding,
    resolve_judge_execution_settings,
    resolve_judge_targets,
)
from adele_runner.runtime.types import ChatResponse, ResolvedJudgeTarget
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.stages.judging import (
    build_judge_output,
    build_judge_prompt,
    build_judge_request,
    parse_judge_json,
    parse_judge_v2,
)
from adele_runner.utils.io import append_jsonl, build_dedup_index, ensure_run_dir

logger = logging.getLogger(__name__)


async def _run_request_response_judges(
    config: AppConfig,
    targets: list[ResolvedJudgeTarget],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    judge_path,
) -> list[JudgeOutput]:  # noqa: ANN001
    if not targets:
        return []

    outputs: list[JudgeOutput] = []
    for target in targets:
        binding = resolve_judge_binding(config, target)
        settings = resolve_judge_execution_settings(config, target)
        rate_limiter = create_rate_limiter(settings, target.rate_limits)
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
        ) -> None:
            if isinstance(response, BaseException):
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
    targets: list[ResolvedJudgeTarget],
    inference_outputs: list[InferenceOutput],
    ground_truths: dict[str, str],
    done: set[tuple],
    run_dir,
    judge_path,
) -> list[JudgeOutput]:  # noqa: ANN001
    if not targets:
        return []

    all_outputs: list[JudgeOutput] = []

    for target in targets:
        binding = resolve_judge_binding(config, target)
        settings = resolve_judge_execution_settings(config, target)
        adapter = binding.create_adapter(config)
        requests = []
        request_meta: dict[str, tuple[InferenceOutput, str]] = {}

        for inference_output in inference_outputs:
            key = (inference_output.instance_id, inference_output.model_id, target.judge_name)
            if key in done:
                continue
            ground_truth = ground_truths.get(inference_output.instance_id, "")
            request, judge_prompt = build_judge_request(inference_output, ground_truth, target)
            requests.append(request)
            request_meta[request.request_id] = (inference_output, judge_prompt)

        if not requests:
            logger.info("Batch judge [%s]: nothing pending.", target.judge_name)
            continue

        logger.info("Batch judge [%s]: %d tasks pending.", target.judge_name, len(requests))
        responses = await BatchExecutor().execute(
            adapter=adapter,
            requests=requests,
            run_dir=run_dir,
            settings=settings,
        )
        for response in responses:
            meta = request_meta.get(response.request_id)
            if meta is None:
                logger.warning(
                    "Batch judge [%s]: unknown request_id %s",
                    target.judge_name,
                    response.request_id,
                )
                continue
            inference_output, judge_prompt = meta
            output = build_judge_output(
                inference_output,
                target,
                judge_prompt,
                response,
                config.run.run_id,
            )
            append_jsonl(judge_path, output)
            all_outputs.append(output)

    return all_outputs


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

    targets = resolve_judge_targets(config)
    request_response_targets = [target for target in targets if target.prompt_mode == "request_response"]
    batch_targets = [target for target in targets if target.prompt_mode == "batch"]

    request_response_coro = _run_request_response_judges(
        config,
        request_response_targets,
        inference_outputs,
        ground_truths,
        done,
        judge_path,
    )
    batch_coro = _run_batch_judges(
        config,
        batch_targets,
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


__all__ = [
    "build_judge_prompt",
    "parse_judge_json",
    "parse_judge_v2",
    "run_judge",
]
