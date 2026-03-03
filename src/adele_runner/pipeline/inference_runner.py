"""Async inference runner with bounded concurrency and checkpointing."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from adele_runner.adapters.foundry_inference import FoundryAdapter
from adele_runner.config import AppConfig
from adele_runner.schemas import DatasetItem, InferenceOutput, RunManifest
from adele_runner.utils.concurrency import AsyncRateLimiter, bounded_gather
from adele_runner.utils.io import append_jsonl, build_dedup_index, ensure_run_dir
from adele_runner.utils.retry import make_retry_decorator

logger = logging.getLogger(__name__)


async def _infer_with_retry(
    adapter: FoundryAdapter,
    item: DatasetItem,
    retry_decorator,  # noqa: ANN001
) -> InferenceOutput:
    @retry_decorator
    async def _call() -> InferenceOutput:
        return await adapter.infer(item)

    return await _call()


async def _run_foundry_async(
    config: AppConfig,
    pending: list[DatasetItem],
    outputs_path: Path,
) -> list[InferenceOutput]:
    """Run inference via Azure AI Foundry with async concurrency."""
    concurrency_cfg = config.concurrency

    # Construct rate limiter if effective_rpm was computed from rate limits
    rate_limiter: AsyncRateLimiter | None = None
    if concurrency_cfg.effective_rpm is not None:
        rl = config.inference.rate_limits
        rate_limiter = AsyncRateLimiter(
            concurrency_cfg.effective_rpm,
            tpm=rl.tokens_per_minute if rl else None,
        )
        logger.info("Rate limiter enabled: %d RPM", concurrency_cfg.effective_rpm)

    adapter = FoundryAdapter(config, rate_limiter=rate_limiter)

    retry_dec = make_retry_decorator(
        max_retries=concurrency_cfg.max_retries,
        backoff_base=concurrency_cfg.backoff_base_s,
        backoff_max=concurrency_cfg.backoff_max_s,
        rate_limiter=rate_limiter,
    )

    completed: list[InferenceOutput] = []

    # Process in batches so we checkpoint frequently
    chunk_size = concurrency_cfg.max_in_flight * 4

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Inference", total=len(pending))

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            tasks = [_infer_with_retry(adapter, item, retry_dec) for item in chunk]
            results = await bounded_gather(
                tasks,
                max_concurrency=concurrency_cfg.max_in_flight,
                rate_limiter=rate_limiter,
            )

            for result in results:
                if isinstance(result, BaseException):
                    logger.error("Inference task failed (skipping): %s", result)
                    progress.advance(task_id)
                    continue
                append_jsonl(outputs_path, result)
                completed.append(result)
                progress.advance(task_id)

            logger.info(
                "Checkpoint: %d / %d done.",
                chunk_start + len(chunk),
                len(pending),
            )

    return completed


def _run_azure_openai_batch(
    config: AppConfig,
    pending: list[DatasetItem],
    run_dir: Path,
    outputs_path: Path,
) -> list[InferenceOutput]:
    """Run inference via the Azure OpenAI Batch API."""
    from adele_runner.adapters.azure_openai_batch import AzureOpenAIBatchAdapter

    adapter = AzureOpenAIBatchAdapter(config)
    results = adapter.run_batch(pending, run_dir)

    for result in results:
        append_jsonl(outputs_path, result)

    return results


async def run_inference(config: AppConfig, items: list[DatasetItem]) -> list[InferenceOutput]:
    """Run inference over *items* with checkpointing and dedup.

    Already-completed (instance_id, model_id) pairs are skipped.
    Results are appended to ``outputs.jsonl`` as they complete.
    """
    mode = config.resolve_inference_mode()
    run_dir = config.run_dir()
    ensure_run_dir(run_dir)
    outputs_path = config.outputs_path()

    model_id = config.inference.model

    # --- Write initial RunManifest ---
    manifest = RunManifest(
        run_id=config.run.run_id,
        dataset_name=config.dataset.name,
        model_id=model_id,
        total_instances=len(items),
        start_time=datetime.utcnow(),
    )
    config.manifest_path().write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    done = build_dedup_index(outputs_path, "instance_id", "model_id")
    logger.info("Dedup index loaded: %d already completed.", len(done))

    pending = [i for i in items if (i.instance_id, model_id) not in done]
    logger.info("%d / %d items pending inference.", len(pending), len(items))

    if not pending:
        logger.info("All items already completed. Nothing to do.")
        return []

    logger.info("Inference mode: %s", mode)

    if mode == "batch":
        completed = await asyncio.to_thread(
            _run_azure_openai_batch,
            config,
            pending,
            run_dir,
            outputs_path,
        )
    else:
        completed = await _run_foundry_async(config, pending, outputs_path)

    logger.info("Inference complete. %d outputs written to %s", len(completed), outputs_path)

    # --- Update RunManifest with completion info ---
    manifest.end_time = datetime.utcnow()
    manifest.completed_instances = len(completed)
    config.manifest_path().write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    return completed
