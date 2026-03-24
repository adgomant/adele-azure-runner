# Development

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

The runtime now depends directly on these provider SDKs:

- `azure-ai-inference`
- `openai`
- `google-genai`
- `anthropic`

If you are working in an offline environment, prefer `./.venv/bin/...` commands over `uv run`, since `uv` may attempt dependency resolution.

## Project Structure

```text
src/adele_runner/
  cli.py
  config.py
  schemas.py

  adapters/
    providers/
      azure_ai_inference/
      azure_openai/
      google_genai/
      anthropic/
    foundry.py
    google_genai.py
    azure_openai.py
    factory.py

  runtime/
    types.py
    registry.py
    resolution.py
    executors.py

  stages/
    inference.py
    judging.py

  pipeline/
    inference_runner.py
    judge_runner.py
    merge.py
    metrics.py
```

Rules for extending the runtime:

- add providers in `runtime/registry.py`
- add SDK-specific mode adapters under `adapters/providers/<provider>/`
- keep prompt and parsing logic in `stages/`
- keep provider branching out of executors

## Tests

```bash
./.venv/bin/pytest -q
./.venv/bin/pytest tests/test_resolution.py -q
```

Important test areas:

- config normalization and validation
- provider/mode resolution
- executor behavior
- stage request/output mapping
- dedup and resume

## Lint and Format

```bash
./.venv/bin/ruff check src/ tests/
./.venv/bin/ruff format src/ tests/
```

## Type Checking

```bash
./.venv/bin/mypy src/
```

## Adding a New Provider

Typical steps:

1. add provider connection config under `providers.*`
2. implement `request_response` and/or `batch` adapters
3. register the provider descriptor
4. add capability validation rules
5. add resolution and executor tests
