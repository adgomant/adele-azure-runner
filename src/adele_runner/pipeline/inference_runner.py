from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from adele_runner.adapters.foundry_inference import FoundryChatAdapter
from adele_runner.config import AppConfig
from adele_runner.datasets.adele import load_adele_examples
from adele_runner.models import InferenceOutput
from adele_runner.utils.io import append_jsonl, ensure_run_dir, load_dedup_keys

logger = logging.getLogger(__name__)


async def run_inference(config: AppConfig) -> Path:
    run_dir = ensure_run_dir(config.run.output_dir, config.run.run_id)
    outputs_path = run_dir / "outputs.jsonl"
    dedup = load_dedup_keys(outputs_path, ("instance_id", "model_id"))

    adapter = FoundryChatAdapter(config.inference.foundry)
    semaphore = asyncio.Semaphore(config.concurrency.max_in_flight)

    async def process(example) -> None:
        key = (example.instance_id, config.inference.foundry.model)
        if key in dedup:
            return
        async with semaphore:
            start = time.perf_counter()
            response = await asyncio.to_thread(adapter.complete, example.prompt)
            latency_ms = (time.perf_counter() - start) * 1000.0
            output = InferenceOutput(
                run_id=config.run.run_id,
                instance_id=example.instance_id,
                model_id=config.inference.foundry.model,
                prompt=example.prompt,
                response_text=response["text"],
                latency_ms=latency_ms,
                raw_response=response["raw"],
                metadata={"ground_truth": example.ground_truth, **example.metadata},
            )
            append_jsonl(outputs_path, output.model_dump(mode="json"))
            dedup.add(key)
            logger.info("inference_complete instance_id=%s", example.instance_id)

    tasks = [asyncio.create_task(process(ex)) for ex in load_adele_examples(config.dataset)]
    if tasks:
        await asyncio.gather(*tasks)
    return outputs_path
