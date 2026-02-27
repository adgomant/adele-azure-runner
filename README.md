# ADeLe Azure Runner

Run the [ADeLe benchmark](https://huggingface.co/datasets/CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0) at scale against Azure AI Foundry models, then evaluate outputs with multiple LLM judges.

## Features

- **Universal inference** -- any Foundry-deployed model (GPT, Claude, Llama, Qwen, DeepSeek, ...) via the `azure-ai-inference` SDK
- **Azure OpenAI Batch API** fast-path for inference and judging when available
- **Multi-judge evaluation** -- run multiple judges (e.g. GPT-4o + Claude) per instance, mix Foundry async and Batch judges in the same run
- **Two judge prompt templates** -- structured JSON (v1) or bare-integer (v2) scoring
- **Async bounded concurrency** with exponential backoff, smart retry filtering, and per-request timeouts
- **Idempotent runs** -- resume from any interruption; completed instances are skipped automatically
- **Durable artifacts** -- `outputs.jsonl`, `judge_outputs.jsonl`, `merged_results.parquet`, `run_manifest.json`, `metrics.json`
- **Metrics** -- per-judge score distributions, verification pass rate, inter-rater agreement (Cohen's kappa), token usage, optional cost estimation

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- Azure AI Foundry deployment(s)
- (Optional) Azure OpenAI resource for batch mode

## Quick Start

```bash
# Clone
git clone <repo-url>
cd adele-azure-runner

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync                     # add --extra batch for Azure OpenAI Batch API support

# Copy and edit config
cp config.example.yaml config.yaml
# Edit config.yaml: set your endpoint, model, and judge settings

# Set API keys (never hardcode)
export AZURE_AI_API_KEY="<your-foundry-api-key>"
export AZURE_OPENAI_API_KEY="<your-azure-openai-key>"   # only if using batch mode
```

## Example End-to-End Run

```bash
# 1. Run inference on the first 50 ADeLe items
uv run adele-runner run-inference --model gpt-4o

# 2. Judge outputs with two models
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus

# 3. Merge into Parquet
uv run adele-runner merge-results

# 4. Print metrics summary
uv run adele-runner summarize
```

Or run the full pipeline in one command:

```bash
uv run adele-runner run-all --model gpt-4o --judge gpt-4o --judge claude-3-opus
```

Use `--dry-run` on any command to preview the plan without making API calls:

```bash
uv run adele-runner run-inference --model gpt-4o --dry-run
```

## Output Artifacts

All outputs are written to `runs/<run_id>/`:

| File | Description |
|---|---|
| `outputs.jsonl` | Inference results (append-only, one JSON object per line) |
| `judge_outputs.jsonl` | Judge evaluations (append-only) |
| `merged_results.parquet` | Combined view with judge scores and verification metrics |
| `run_manifest.json` | Run provenance: model, dataset, timestamps, instance counts |
| `metrics.json` | Summary statistics, inter-rater agreement, token usage |
| `ground_truths.json` | Cached ground truths (avoids re-downloading the dataset) |

## Documentation

| Guide | Contents |
|---|---|
| [Configuration](docs/configuration.md) | Full config reference -- every field, type, default, env vars |
| [CLI Reference](docs/cli-reference.md) | All commands and flags with examples |
| [Architecture](docs/architecture.md) | Module map, data flow, adapter pattern, concurrency model |
| [Judging](docs/judging.md) | Prompt templates, multi-judge setup, JSON parsing pipeline |
| [Batch Mode](docs/batch-mode.md) | Azure OpenAI Batch API lifecycle and configuration |
| [Resume & Idempotency](docs/resume-and-idempotency.md) | Dedup index, safe resume, re-run semantics |
| [Output Artifacts](docs/output-artifacts.md) | File schemas and how to consume them |
| [Development](docs/development.md) | Dev setup, tests, lint, project structure |

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type-check
uv run mypy src/
```

See [docs/development.md](docs/development.md) for the full project structure and test descriptions.
