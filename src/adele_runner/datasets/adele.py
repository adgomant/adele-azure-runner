from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from adele_runner.config import DatasetConfig
from adele_runner.models import NormalizedExample


def load_adele_examples(config: DatasetConfig) -> Iterator[NormalizedExample]:
    dataset = load_dataset(config.hf_id, split=config.split)
    dataset = dataset.sort("instance_id") if "instance_id" in dataset.column_names else dataset

    count = 0
    for row in dataset:
        if config.limit is not None and count >= config.limit:
            break
        instance_id = str(row.get("instance_id", row.get("id", count)))
        prompt = str(row.get("prompt", row.get("question", "")))
        ground_truth = row.get("ground_truth", row.get("answer"))
        metadata = {k: v for k, v in row.items() if k not in {"instance_id", "id", "prompt", "question", "ground_truth", "answer"}}
        yield NormalizedExample(
            instance_id=instance_id,
            prompt=prompt,
            ground_truth=None if ground_truth is None else str(ground_truth),
            metadata=metadata,
        )
        count += 1
