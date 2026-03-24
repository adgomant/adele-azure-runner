"""Inference pipeline orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from adele_runner.config import AppConfig
from adele_runner.runtime.executors import BatchExecutor, RequestResponseExecutor
from adele_runner.runtime.resolution import (
    resolve_inference_execution_settings,
    resolve_inference_target,
)
from adele_runner.runtime.types import ChatResponse
from adele_runner.schemas import DatasetItem, InferenceOutput, RunManifest
from adele_runner.stages.inference import build_inference_output, build_inference_request
from adele_runner.utils.io import append_jsonl, build_dedup_index, ensure_run_dir

logger = logging.getLogger(__name__)


async def run_inference(config: AppConfig, items: list[DatasetItem]) -> list[InferenceOutput]:
    """Run inference over *items* with checkpointing and dedup."""
    target = resolve_inference_target(config)
    settings = resolve_inference_execution_settings(config)
    run_dir = config.run_dir()
    ensure_run_dir(run_dir)
    outputs_path = config.outputs_path()

    manifest = RunManifest(
        run_id=config.run.run_id,
        dataset_name=config.dataset.name,
        model_id=target.model,
        total_instances=len(items),
        start_time=datetime.utcnow(),
    )
    config.manifest_path().write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    done = build_dedup_index(outputs_path, "instance_id", "model_id")
    logger.info("Dedup index loaded: %d already completed.", len(done))

    pending = [item for item in items if (item.instance_id, target.model) not in done]
    logger.info("%d / %d items pending inference.", len(pending), len(items))

    if not pending:
        logger.info("All items already completed. Nothing to do.")
        return []

    requests = [build_inference_request(item, target) for item in pending]
    item_by_id = {item.instance_id: item for item in pending}
    completed: list[InferenceOutput] = []

    def _record_response(response: ChatResponse | BaseException) -> None:
        if isinstance(response, BaseException):
            return
        item = item_by_id.get(response.request_id)
        if item is None:
            logger.warning("Inference response had unknown request_id=%s", response.request_id)
            return
        output = build_inference_output(item, target, response, config.run.run_id)
        append_jsonl(outputs_path, output)
        completed.append(output)

    logger.info("Inference execution: adapter=%s mode=%s", target.adapter_kind, target.execution_kind)

    if target.execution_kind == "batch":
        await BatchExecutor(config).execute(
            adapter_kind=target.adapter_kind,
            requests=requests,
            run_dir=run_dir,
            settings=settings,
            on_result=_record_response,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Inference", total=len(requests))

            def _record_and_advance(response: ChatResponse | BaseException) -> None:
                _record_response(response)
                progress.advance(task_id)

            await RequestResponseExecutor(config).execute(
                adapter_kind=target.adapter_kind,
                requests=requests,
                settings=settings,
                rate_limits=target.rate_limits,
                on_result=_record_and_advance,
            )

    logger.info("Inference complete. %d outputs written to %s", len(completed), outputs_path)

    manifest.end_time = datetime.utcnow()
    manifest.completed_instances = len(completed)
    config.manifest_path().write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    return completed
