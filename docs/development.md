# Development

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)

## Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install all dependencies (core + batch + dev)
uv sync --extra batch --extra dev
```

The `batch` extra installs the `openai` package for Azure OpenAI Batch API support. The `dev` extra installs `pytest`, `ruff`, `mypy`, and related tools.

## Project Structure

```
src/adele_runner/
  __init__.py
  cli.py                      # Typer CLI entry point (5 commands)
  config.py                   # Pydantic config models + YAML loader
  schemas.py                  # Data models: InferenceOutput, JudgeOutput, DatasetItem, RunManifest

  adapters/
    __init__.py
    foundry_inference.py      # Azure AI Foundry async inference adapter
    azure_openai_batch.py     # Azure OpenAI Batch API adapter

  datasets/
    __init__.py
    adele.py                  # ADeLe dataset loader (HuggingFace datasets)

  pipeline/
    __init__.py
    inference_runner.py       # Inference orchestration: dedup, dispatch, checkpoint
    judge_runner.py           # Multi-judge evaluation: prompts, parsing, batch+foundry
    merge.py                  # Merge inference + judge outputs → Parquet
    metrics.py                # Score stats, Cohen's kappa, token usage, cost

  utils/
    __init__.py
    io.py                     # JSONL/Parquet I/O, dedup index builder
    retry.py                  # Tenacity retry with smart error classification
    concurrency.py            # Bounded async concurrency (semaphore)
    batch_split.py            # Split large batch request files to respect Azure limits
```

## Running Tests

```bash
uv run pytest tests/ -v
```

### Test Files

| File | What it tests |
|---|---|
| `test_batch_split.py` | Batch request file splitting logic, Azure limits, edge cases |
| `test_cli_overrides.py` | `--run-id`, `--model`, `--mode`, `--judge`, `--judge-template`, `--tpm`, `--rpm`, `max_tokens` override logic |
| `test_config_validation.py` | Placeholder detection, missing endpoints, invalid templates |
| `test_dotenv_loading.py` | `.env` / `.env.local` loading and precedence |
| `test_dry_run.py` | `--dry-run` flag: summary output, validation gating |
| `test_inter_rater.py` | Cohen's kappa: perfect/zero/chance agreement, known values |
| `test_judge_v2.py` | v2 bare-integer parsing: clean int, clamping, preamble, fallback |
| `test_mode_resolution.py` | `resolve_inference_mode()` with different config combinations |
| `test_parsing.py` | v1 judge JSON parsing: valid, malformed, markdown fences, regex |
| `test_pick_col.py` | Case-insensitive column detection, alias resolution |
| `test_pricing.py` | Token aggregation, cost calculation, pricing disabled |
| `test_rate_limit_compute.py` | Rate limit auto-tuning, `get_max_judge_max_tokens()` |
| `test_resume_dedup.py` | JSONL append, dedup index, resume filtering |
| `test_retry_filter.py` | `is_retryable()`: transport errors, HTTP status codes, logic errors |
| `test_run_manifest.py` | Manifest creation and update |
| `test_schema_validation.py` | Pydantic schema validation for InferenceOutput, JudgeOutput |
| `test_verification_score.py` | Verification score computation |

### Running Specific Tests

```bash
# Run a single test file
uv run pytest tests/test_parsing.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_retry" -v
```

## Linting

```bash
# Check for errors
uv run ruff check src/ tests/

# Auto-fix where possible
uv run ruff check --fix src/ tests/
```

Ruff is configured in `pyproject.toml`:

- Line length: 100
- Target: Python 3.11
- Rules: E, F, I, UP, B, SIM
- Ignored: E501 (line length -- handled by formatter), B008 (Typer default pattern)

## Formatting

```bash
uv run ruff format src/ tests/
```

## Type Checking

```bash
uv run mypy src/
```

mypy is configured with `strict = true` and `ignore_missing_imports = true`.

## Adding New Tests

Test files live in `tests/` and follow the `test_*.py` naming convention. The test suite uses:

- `pytest` for test discovery and execution
- `pytest-asyncio` for async test functions
- `pytest-mock` for mocking
- `tmp_path` fixture for temporary directories

Example:

```python
"""Tests for my new feature."""
from adele_runner.schemas import InferenceOutput

def test_something(tmp_path):
    # tmp_path is a pytest fixture providing a temporary directory
    output = InferenceOutput(
        instance_id="test",
        model_id="gpt-4o",
        prompt="hello",
        response="world",
    )
    assert output.instance_id == "test"
```
