# ADeLe Azure Runner

Run the [ADeLe benchmark](https://huggingface.co/datasets/CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0) against multiple model SDKs and execution modes, then evaluate outputs with one or more LLM judges.

## Features

- **Provider / mode split** -- configure the SDK provider independently from the execution mode
- **Multiple providers** -- `azure_ai_inference`, `azure_openai`, `google_genai`, and `anthropic`
- **Multiple modes** -- `request_response`, `batch`, and provider-local `auto`
- **Multi-judge evaluation** -- mix request-response and batch judges in the same run
- **Two judge prompt templates** -- structured JSON (v1) or bare-integer (v2) scoring
- **Async bounded concurrency** with exponential backoff, smart retry filtering, and per-request timeouts
- **Rate-limit auto-tuning** -- provide TPM/RPM and the runner computes optimal concurrency settings
- **Idempotent runs** -- resume from any interruption; completed instances are skipped automatically
- **Durable artifacts** -- `outputs.jsonl`, `judge_outputs.jsonl`, `merged_results.parquet`, `run_manifest.json`, `metrics.json`
- **Metrics** -- per-judge score distributions, verification pass rate, inter-rater agreement (Cohen's kappa), token usage, optional cost estimation

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- Credentials and target endpoints for whichever providers you use

## Quick Start

```bash
# Clone
git clone <repo-url>
cd adele-azure-runner

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync --extra dev

# Copy and edit config
cp config.example.yaml config.yaml
# Edit config.yaml: set provider credentials, targets, and judge settings

# Set API keys -- copy .env to .env.local and fill in your real keys
cp .env .env.local
# Edit .env.local with your actual API keys (auto-loaded by the CLI)
```

## Example End-to-End Run

```bash
# 1. Run inference on the first 50 ADeLe items
uv run adele-runner run-inference --provider azure_ai_inference --model gpt-4o

# 2. Judge outputs with two models
uv run adele-runner run-judge --judge gpt-4o:azure_ai_inference:request_response --judge claude-sonnet-4-5:anthropic:batch

# 3. Merge into Parquet
uv run adele-runner merge-results

# 4. Print metrics summary
uv run adele-runner summarize
```

Or run the full pipeline in one command:

```bash
uv run adele-runner run-all --provider azure_ai_inference --model gpt-4o --judge gpt-4o:azure_ai_inference:request_response --judge claude-sonnet-4-5:anthropic:batch
```

Use `--dry-run` on any command to preview the plan without making API calls:

```bash
uv run adele-runner run-inference --provider google_genai --mode request_response --model gemini-2.5-flash --dry-run
```

Use `--run-id` before any command to override the run ID (useful for comparing models):

```bash
uv run adele-runner --run-id gpt4o_run run-all --model gpt-4o --judge gpt-4o
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
| [Configuration](docs/configuration.md) | `providers.*`, targets, inference, judges, validation, and compatibility |
| [CLI Reference](docs/cli-reference.md) | Commands, `--provider`, `--mode`, and judge grammar |
| [Architecture](docs/architecture.md) | Provider registry, transport contracts, stages, and executors |
| [Judging](docs/judging.md) | Judge config, parsing, and mixed-mode execution |
| [Batch Mode](docs/batch-mode.md) | Batch support across Azure OpenAI, Google GenAI, and Anthropic |
| [Resume & Idempotency](docs/resume-and-idempotency.md) | Dedup index, safe resume, re-run semantics |
| [Output Artifacts](docs/output-artifacts.md) | File schemas and how to consume them |
| [Development](docs/development.md) | Dev setup, tests, lint, project structure |

## Development

```bash
# Run tests
./.venv/bin/pytest tests/ -v

# Lint and format
./.venv/bin/ruff check src/ tests/
./.venv/bin/ruff format src/ tests/

# Type-check
./.venv/bin/mypy src/
```

See [docs/development.md](docs/development.md) for the full project structure and test descriptions.
