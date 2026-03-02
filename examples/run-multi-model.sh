#!/usr/bin/env bash
# Compare two models side-by-side with the same judges.
#
# Prerequisites:
#   - config.yaml with endpoint configured
#   - AZURE_AI_API_KEY set in environment
#
# Usage:
#   bash examples/run-multi-model.sh

set -euo pipefail

echo "=== Running full pipeline for two models ==="
uv run adele-runner run-all \
  --model gpt-4o \
  --model llama-3-70b \
  --judge gpt-4o \
  --judge claude-3-opus

echo "=== Done ==="
echo "Results are in runs/<run_id>/merged_results.parquet"
