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
from adele_runner.runtime.budgeting import BudgetTracker
from adele_runner.runtime.executors import (
    BatchExecutor,
    RequestResponseExecutor,
    create_rate_limiter,
)
from adele_runner.runtime.resolution import resolve_inference_plan
from adele_runner.runtime.types import ChatResponse
from adele_runner.schemas import DatasetItem, InferenceOutput, RunManifest
from adele_runner.stages.inference import build_inference_output, build_inference_request
from adele_runner.utils.io import append_jsonl, build_dedup_index, ensure_run_dir

logger = logging.getLogger(__name__)


def _build_inference_budget_tracker(config: AppConfig) -> BudgetTracker | None:
    if config.inference.budget_usd is None:
        return None
    model_pricing = config.pricing.get_model_pricing(config.inference.model)
    if model_pricing is None:
        raise ValueError(f"No pricing configured for inference model '{config.inference.model}'.")
    return BudgetTracker(
        lane_name=f"inference:{config.inference.model}",
        pricing_key=config.inference.model,
        budget_usd=config.inference.budget_usd,
        model_pricing=model_pricing,
    )


async def run_inference(config: AppConfig, items: list[DatasetItem]) -> list[InferenceOutput]:
    """Run inference over *items* with checkpointing and dedup."""
    plan = resolve_inference_plan(config)
    target = plan.target
    binding = plan.binding
    settings = plan.settings
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
    budget_tracker = _build_inference_budget_tracker(config)

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
        if budget_tracker is not None:
            budget_tracker.record_response(response)

    logger.info("Inference execution: provider=%s mode=%s", target.provider_kind, binding.execution_kind)

    try:
        if binding.execution_kind == "batch":
            adapter = binding.create_adapter(config, budget_tracker=budget_tracker)
            await BatchExecutor().execute(
                adapter=adapter,
                requests=requests,
                run_dir=run_dir,
                settings=settings,
                on_result=_record_response,
            )
        else:
            rate_limiter = create_rate_limiter(settings)
            adapter = binding.create_adapter(
                config,
                rate_limiter=rate_limiter,
                configured_rate_limits=target.rate_limits,
            )
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
                    if not isinstance(response, BaseException):
                        progress.advance(task_id)

                await RequestResponseExecutor().execute(
                    adapter=adapter,
                    requests=requests,
                    settings=settings,
                    rate_limiter=rate_limiter,
                    on_result=_record_and_advance,
                )
    finally:
        logger.info("Inference complete. %d outputs written to %s", len(completed), outputs_path)

        manifest.end_time = datetime.utcnow()
        manifest.completed_instances = len(completed)
        config.manifest_path().write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

    return completed
