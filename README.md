# ADeLe Azure Runner

Production-oriented benchmark runner for the ADeLe dataset using Azure AI Foundry inference and multi-judge evaluation.

## Architecture (short)
- **Dataset layer**: pluggable loaders (`datasets/`) normalize records to `{instance_id, prompt, ground_truth}`.
- **Adapter layer**: model/provider specific clients (`adapters/`) with a common chat interface.
- **Pipeline layer**: streaming inference, judging, merge, and summary stages (`pipeline/`).
- **Storage layer**: append-only JSONL artifacts + merged parquet in `runs/<run_id>/`.

## Artifacts
- `outputs.jsonl`: model outputs (dedup key: `instance_id + model_id`)
- `judge_outputs.jsonl`: one row per `{instance_id, model_id, judge_name}`
- `merged_results.parquet`: merged inference + judge records
- `run_manifest.json`: immutable run metadata

## Quickstart
```bash
uv venv
uv sync --extra dev
cp config.example.yaml config.yaml
uv run adele-runner run-inference --config config.yaml
uv run adele-runner run-judge --config config.yaml
uv run adele-runner merge-results --config config.yaml
uv run adele-runner summarize --config config.yaml
```

## Config
Use YAML + environment variables for secrets and endpoint auth. See `config.example.yaml`.

## Notes
- Idempotent reruns: completed records are skipped.
- Streaming JSONL IO is used to avoid high memory usage.
