#!/usr/bin/env bash
# Run a subset of ADeLe: 50 items, one model, two judges.
#
# Prerequisites:
#   - config.yaml with endpoint configured (see examples/config.foundry-only.yaml)
#   - AZURE_AI_API_KEY set in environment
#
# Usage:
#   bash examples/run-subset.sh

set -euo pipefail

echo "=== Step 1: Inference ==="
uv run adele-runner run-inference --model gpt-4o

echo "=== Step 2: Judging ==="
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus

echo "=== Step 3: Merge ==="
uv run adele-runner merge-results

echo "=== Step 4: Summarize ==="
uv run adele-runner summarize

echo "=== Done ==="
