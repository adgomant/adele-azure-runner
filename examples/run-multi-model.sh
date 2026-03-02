#!/usr/bin/env bash
# Compare two models side-by-side with the same judges.
#
# Since --model accepts a single value, this script runs each model
# sequentially and then judges + merges + summarizes once at the end.
#
# Prerequisites:
#   - config.yaml with endpoint configured and judging.judges set
#   - AZURE_AI_API_KEY set in environment
#
# Usage:
#   bash examples/run-multi-model.sh

set -euo pipefail

MODELS=("gpt-4o" "llama-3-70b")

for model in "${MODELS[@]}"; do
  echo "=== Inference: $model ==="
  uv run adele-runner run-inference --model "$model"
done

echo "=== Judging ==="
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus

echo "=== Merge ==="
uv run adele-runner merge-results

echo "=== Summarize ==="
uv run adele-runner summarize

echo "=== Done ==="
echo "Results are in runs/<run_id>/merged_results.parquet"
