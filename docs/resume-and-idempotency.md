# Resume and Idempotency

The runner is designed to be safe to interrupt and re-run. Completed work is never lost and never re-processed.

## Append-Only JSONL

Both `outputs.jsonl` (inference) and `judge_outputs.jsonl` (judging) are append-only files. Each completed instance is written as a single JSON line immediately after processing. This means:

- Progress is saved after every instance, not just at the end
- Interrupting a run (Ctrl+C, crash, timeout) loses at most one in-flight instance
- The files grow monotonically and are never rewritten

## Dedup Index

At the start of every run, the runner builds a **dedup index** by scanning the existing output file:

### Inference dedup

Key: `(instance_id, model_id)`

```
1. Read outputs.jsonl
2. Extract (instance_id, model_id) from each line
3. Build a set of seen keys
4. Filter dataset items: skip any where (item.instance_id, model_id) is in the set
5. Process only the remaining items
```

### Judge dedup

Key: `(instance_id, model_id, judge_name)`

```
1. Read judge_outputs.jsonl
2. Extract (instance_id, model_id, judge_name) from each line
3. Build a set of seen keys
4. For each judge: skip instances already evaluated by that judge
5. Process only the remaining combinations
```

## Re-Run Semantics

| Scenario | Behavior |
|---|---|
| Re-run same model | Skips all completed instances, processes only missing ones |
| Add a second model | Processes all instances for the new model (different model_id) |
| Add a new judge | Processes all instances for the new judge (different judge_name) |
| Resume after crash | Picks up where it left off, skipping completed instances |
| Re-run after completion | Skips everything (all instances already done) |

## Ground Truth Caching

When running judges, the runner needs ground truth answers from the dataset. To avoid re-downloading the HuggingFace dataset on every run:

1. First run: downloads dataset, extracts ground truths, writes `ground_truths.json` to the run directory
2. Subsequent runs: loads ground truths from the cached file
3. If the cache is corrupted or unreadable, falls back to re-downloading

The cache file is a JSON object mapping `instance_id` to `ground_truth` string.

## Run Manifest

The runner writes a `run_manifest.json` at the start of inference:

```json
{
  "run_id": "adele_dev_run",
  "dataset_name": "adele",
  "model_id": "gpt-4o",
  "total_instances": 50,
  "completed_instances": 0,
  "start_time": "2025-01-15T10:30:00",
  "end_time": null,
  "code_version": "0.1.0",
  "params": {}
}
```

After inference completes, the manifest is updated with `end_time` and `completed_instances`.

The same finalization also happens when inference stops early because a configured runtime budget is exhausted, so resumed runs still have accurate checkpoint metadata.

## Budget Stops

Runtime budget enforcement is designed to work with the existing append-only and dedup-based resume flow.

- request-response runs write each completed output first, then charge the budget
- batch runs stop before submitting a chunk whose estimated worst-case cost would exceed the remaining budget
- if a completed response or chunk pushes actual spend over budget, the runner exits and leaves all completed outputs on disk

To continue after a budget stop, increase or remove the budget in config and rerun the same command with the same `run_id`. The dedup index skips everything already written.

## Safe Practices

- Always use the same `run_id` when resuming a run. A different `run_id` creates a new output directory.
- Do not manually edit `outputs.jsonl` or `judge_outputs.jsonl`. The dedup index depends on valid JSON lines.
- Deleting an output file resets progress for that stage. The runner will re-process everything.
- The `merged_results.parquet` and `metrics.json` files are regenerated from scratch on every `merge-results` / `summarize` call. They are not append-only.
