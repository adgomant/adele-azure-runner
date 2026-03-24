# CLI Reference

The CLI keeps the same top-level commands, but the runtime selection model is now explicit: use `--provider` and `--mode` instead of overloading `mode` with provider meaning.

```bash
uv run adele-runner [GLOBAL OPTIONS] <command> [OPTIONS]
```

## Global Options

| Flag | Short | Description |
|---|---|---|
| `--run-id` | `-r` | Override `run.run_id` |

## Common Options

| Flag | Short | Description |
|---|---|---|
| `--config` | `-c` | Path to YAML config |
| `--dry-run` | | Print the resolved plan and exit |

## `run-inference`

```bash
uv run adele-runner run-inference [OPTIONS]
```

| Flag | Type | Description |
|---|---|---|
| `--model`, `-m` | `TEXT` | Model name or deployment |
| `--provider` | `TEXT` | `azure_openai`, `azure_ai_inference`, `google_genai`, or `anthropic` |
| `--mode` | `TEXT` | `request_response`, `batch`, or `auto` |
| `--tpm` | `INT` | Request-response TPM override |
| `--rpm` | `INT` | Request-response RPM override |

Examples:

```bash
uv run adele-runner run-inference --provider azure_ai_inference --model gpt-4o
uv run adele-runner run-inference --provider azure_openai --mode batch --model gpt-4o-batch
uv run adele-runner run-inference --provider google_genai --mode request_response --model gemini-2.5-flash
uv run adele-runner run-inference --provider anthropic --mode request_response --model claude-sonnet-4-5
```

## `run-judge`

```bash
uv run adele-runner run-judge [OPTIONS]
```

| Flag | Type | Description |
|---|---|---|
| `--judge`, `-j` | `TEXT` | Repeatable judge spec |
| `--judge-template` | `TEXT` | `v1` or `v2` |

### Judge Flag Grammar

Preferred form:

```text
MODEL:PROVIDER:MODE[:TPM:RPM[:MAX_TOKENS]]
```

Examples:

```bash
--judge gpt-4o:azure_ai_inference:request_response
--judge gpt-4o:azure_openai:batch
--judge gemini-2.5-flash:google_genai:batch
--judge claude-sonnet-4-5:anthropic:request_response:40000:200:512
```

Notes:

- TPM/RPM are only for request-response lanes
- `MAX_TOKENS` is optional
- legacy forms are still accepted and normalized

## `merge-results`

```bash
uv run adele-runner merge-results [OPTIONS]
```

Reads `outputs.jsonl` and `judge_outputs.jsonl`, then writes `merged_results.parquet`.

## `summarize`

```bash
uv run adele-runner summarize [OPTIONS]
```

Writes `metrics.json` and prints summary metrics.

## `run-all`

```bash
uv run adele-runner run-all [OPTIONS]
```

Combines the flags from `run-inference` and `run-judge`.

Examples:

```bash
uv run adele-runner run-all \
  --provider azure_ai_inference \
  --model gpt-4o \
  --judge gpt-4o:azure_ai_inference:request_response \
  --judge claude-sonnet-4-5:anthropic:batch

uv run adele-runner run-all \
  --provider azure_openai \
  --mode batch \
  --model gpt-4o-batch \
  --judge gpt-4o:azure_openai:batch
```

## Legacy CLI Aliases

The CLI still accepts older aliases for one release cycle:

- `--provider foundry` -> `azure_ai_inference`
- `--provider google` -> `google_genai`
- `--mode foundry` -> `provider=azure_ai_inference`, `mode=request_response`
- `--mode google` -> `provider=google_genai`, `mode=request_response`
- `--mode batch` without an explicit provider -> `provider=azure_openai`, `mode=batch`
