# Configuration Reference

ADeLe Runner loads configuration from a YAML file and optionally overrides values with CLI flags.

## Loading Order

1. **`.env` files** -- environment variables are loaded from `.env.local` and `.env` at CLI startup (before any config is read). `.env.local` values take priority over `.env`. Shell-exported variables are never overwritten
2. **YAML file** -- `config.yaml` by default, or the path given via `--config / -c`
3. **CLI overrides** -- flags like `--model`, `--mode`, `--judge`, `--judge-template` mutate the loaded config in-place
4. **Environment variables** -- API keys are read from env vars named in the config (never stored in the YAML)

If no `--config` is provided and `config.yaml` does not exist in the working directory, the CLI exits with an error.

### `.env` files

The CLI automatically loads environment variables from `.env.local` and `.env` in the working directory:

| File | Purpose | Git-tracked? |
|---|---|---|
| `.env` | Placeholder template showing required variables | Yes (committed with placeholder values) |
| `.env.local` | Your real API keys and secrets | No (gitignored) |

To get started, copy `.env` to `.env.local` and fill in your real keys:

```bash
cp .env .env.local
# Edit .env.local with your actual API keys
```

Precedence: **shell env** > `.env.local` > `.env`. Both files use `override=False`, so whichever value is set first wins.

## Config Sections

### `azure`

Connection settings for Azure services. Set once per environment.

#### `azure.foundry`

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | `""` | Azure AI Foundry endpoint URL (e.g. `https://<resource>.services.ai.azure.com/models`) |
| `api_key_env` | `str` | `"AZURE_AI_API_KEY"` | Name of the environment variable holding the API key |

#### `azure.batch`

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | `""` | Azure OpenAI resource endpoint (e.g. `https://<resource>.openai.azure.com`) |
| `api_key_env` | `str` | `"AZURE_OPENAI_API_KEY"` | Env var for the batch API key |
| `api_version` | `str` | `""` | Azure OpenAI API version string |
| `completion_endpoint` | `str` | `"/chat/completions"` | Completion endpoint path used in batch requests |
| `max_requests_per_file` | `int` | `50000` | Maximum requests per batch file. Azure hard limit: 100,000 |
| `max_bytes_per_file` | `int` | `100000000` | Maximum bytes per batch file (100 MB default). Azure hard limit: 200 MB |

### `run`

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | `"adele_run"` | Identifier for this run. Output directory becomes `runs/<run_id>/` |
| `output_dir` | `str` | `"runs"` | Parent directory for all run outputs |

### `dataset`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"adele"` | Dataset name (for manifest metadata) |
| `hf_id` | `str` | `"CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"` | HuggingFace dataset identifier |
| `split` | `str` | `"train"` | Dataset split to load |
| `limit` | `int \| null` | `null` | Load only the first N items. `null` loads all |

### `inference`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `"auto" \| "foundry" \| "batch"` | `"auto"` | Inference mode. See [Mode resolution](#mode-resolution) below |
| `model` | `str` | `""` | Model name or deployment ID. Overridden by `--model` |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Maximum completion tokens |
| `top_p` | `float` | `1.0` | Top-p (nucleus) sampling |
| `rate_limits` | `object \| null` | `null` | Optional rate limits for auto-tuning concurrency in foundry mode (see [Auto-tuning concurrency](#auto-tuning-concurrency)) |

##### `inference.rate_limits`

When set and inference mode is `foundry`, concurrency parameters (`max_in_flight`, `request_timeout_s`, `backoff_base_s`, `backoff_max_s`) are automatically computed from the deployment's rate limits.

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens_per_minute` | `int` | *(required)* | TPM limit from the Azure portal for this deployment |
| `requests_per_minute` | `int` | *(required)* | RPM limit from the Azure portal for this deployment |

### `concurrency`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_in_flight` | `int` | `16` | Maximum concurrent async requests |
| `request_timeout_s` | `float` | `120.0` | Per-request HTTP timeout in seconds |
| `max_retries` | `int` | `6` | Maximum retry attempts per request |
| `backoff_base_s` | `float` | `1.0` | Exponential backoff base (seconds) |
| `backoff_max_s` | `float` | `30.0` | Maximum backoff wait (seconds) |
| `max_poll_time_s` | `float` | `3600.0` | Maximum time to poll a batch job before raising `TimeoutError` |
| `batch_completion_window` | `str` | `"24h"` | Batch API completion window |

### `judging`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Enable the judging stage |
| `prompt_template` | `str` | `"v1"` | Judge prompt template: `"v1"` (structured JSON) or `"v2"` (bare integer). See [Judging](judging.md) |
| `judges` | `list` | `[]` | List of judge configurations (see below) |

Each judge entry:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Display name for this judge |
| `provider` | `"foundry" \| "batch"` | `"foundry"` | Whether to use Foundry async or Azure OpenAI Batch API |
| `model` | `str` | *(required)* | Model name or deployment to use for judging |
| `rate_limits` | `object \| null` | `null` | Optional per-judge rate limits (foundry judges only). Same structure as `inference.rate_limits` |

### `logging`

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### `pricing`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable cost estimation in metrics summary |
| `models` | `dict` | `{}` | Per-model pricing (see below) |

Each model pricing entry (keyed by model name or judge name):

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_per_1k` | `float` | `0.0` | Cost in USD per 1,000 prompt tokens |
| `completion_per_1k` | `float` | `0.0` | Cost in USD per 1,000 completion tokens |

Cost estimation covers both inference and judging. Inference models are keyed by `inference.model` (the model name). Judge models are keyed by `judging.judges[].name` (the judge display name). For example:

```yaml
pricing:
  enabled: true
  models:
    gpt-4o:                # inference model
      prompt_per_1k: 2.50
      completion_per_1k: 10.00
    gpt4o:                 # judge (matches judging.judges[].name)
      prompt_per_1k: 2.50
      completion_per_1k: 10.00
```

## Environment Variables

| Variable | Config field | Used by |
|---|---|---|
| `AZURE_AI_API_KEY` | `azure.foundry.api_key_env` | Foundry inference and Foundry judges |
| `AZURE_OPENAI_API_KEY` | `azure.batch.api_key_env` | Batch inference and Batch judges |

The variable names are configurable. The runner reads the env var whose name is specified in the config, not a hardcoded name.

> **Tip:** Instead of exporting variables in your shell, place them in a `.env.local` file in the project root. See [Loading Order](#loading-order) above.

## Mode Resolution

The `inference.mode` field determines how inference requests are dispatched:

| `inference.mode` | Effective mode |
|---|---|
| `"auto"` | `foundry` |
| `"foundry"` | `foundry` |
| `"batch"` | `batch` |

CLI shorthand: `--mode foundry` and `--mode batch` set the mode directly.

## Config Validation

The runner validates config before making API calls. Validation checks:

- Foundry endpoint is set and does not contain `<YOUR-` placeholders (when using Foundry mode)
- Batch endpoint is set when batch mode is active
- Model is specified (`inference.model`)
- At least one judge is configured when judging is enabled
- `prompt_template` is `"v1"` or `"v2"`

Use `--dry-run` to test config without hitting APIs. Dry-run skips API key validation.

## Auto-tuning Concurrency

When `inference.rate_limits` is configured and the inference mode is `foundry`, the runner automatically computes optimal concurrency parameters from TPM, RPM, and `inference.max_tokens`. This replaces the need to manually tune `concurrency.*` fields and helps avoid 429 (rate limit) errors.

Rate-limit auto-tuning only applies to the **foundry provider**. Batch mode uses a different strategy: requests are automatically split into chunks that respect configurable limits (`azure.batch.max_requests_per_file` and `azure.batch.max_bytes_per_file`). These default to 50K requests / 100 MB per file, well within Azure's hard limits of 100K requests / 200 MB.

### Inference auto-tuning

When `inference.rate_limits` is set and `inference.mode` resolves to `foundry`:

1. **Effective RPM** = min(RPM, TPM / max_tokens) — the tighter constraint
2. **max_in_flight** = effective_rpm x estimated_duration / 60, with 80% safety margin
3. **request_timeout_s** = max_tokens / 50 + 10s overhead, capped 30-600s
4. **backoff_base_s** = 60 / effective_rpm, capped 0.5-10s
5. **backoff_max_s** = backoff_base x 10, capped 30-120s

### Judge auto-tuning

When foundry judges have `rate_limits` configured, the runner uses the **most restrictive** judge's rate limits to tune concurrency before the judging phase. Judge requests use a fixed `max_tokens=512`.

Since inference and judging run sequentially, concurrency parameters are recomputed between phases.

Non-rate-limit params (`max_retries`, `max_poll_time_s`, `batch_completion_window`) are preserved from the user config.

### Example computed values

| TPM | RPM | max_tokens | max_in_flight | timeout_s | backoff |
|-----|-----|------------|---------------|-----------|---------|
| 80k | 300 | 2048 | 17 | 51s | 1.5–15s |
| 80k | 300 | 50000 | 2 | 600s | 0.5–30s |
| 200k | 600 | 2048 | 42 | 51s | 0.6–30s |
| 30k | 100 | 512 | 7 | 30s | 4.1–41s |

### Runtime header validation

On the first API response, the runner reads `x-ratelimit-limit-tokens` and `x-ratelimit-limit-requests` headers from Azure. If either value differs from the config by more than 20%, a warning is logged so you can update your config.

### Retry-After awareness

When a 429 response includes a `Retry-After` or `x-ratelimit-reset-tokens` header, the retry system uses that value instead of generic exponential backoff — avoiding both unnecessary waits and premature retries.

## Annotated Example

See [config.example.yaml](../config.example.yaml) for a fully annotated config file, or the [examples/](../examples/) directory for scenario-specific configs.

## CLI Flags

All CLI flags override the corresponding config-file values.

| Flag | Type | Commands | Description |
|---|---|---|---|
| `--config` / `-c` | `Path` | All | Path to YAML config file |
| `--model` / `-m` | `str` | `run-inference`, `run-all` | Model name or deployment |
| `--mode` | `str` | `run-inference`, `run-all` | Inference mode: `foundry`, `batch`, or `auto` |
| `--tpm` | `int` | `run-inference`, `run-all` | Tokens per minute rate limit (foundry only, requires `--rpm`) |
| `--rpm` | `int` | `run-inference`, `run-all` | Requests per minute rate limit (foundry only, requires `--tpm`) |
| `--judge` / `-j` | `str` | `run-judge`, `run-all` | Judge model (repeatable). See format below |
| `--judge-template` | `str` | `run-judge`, `run-all` | Judge prompt template: `v1` or `v2` |
| `--dry-run` | `bool` | `run-inference`, `run-judge`, `run-all` | Print plan and exit without API calls |

### `--judge` format

```
MODEL                       → foundry provider, no rate limits
MODEL:PROVIDER              → explicit provider (foundry or batch)
MODEL:PROVIDER:TPM:RPM      → explicit provider + rate limits (foundry only)
```

Examples:
```bash
--judge gpt-4o                          # foundry, no rate limits
--judge gpt-4o:batch                    # batch provider
--judge gpt-4o:foundry:80000:300        # foundry with TPM=80000 RPM=300
```

Rate limits for batch judges are not supported (batch uses file splitting instead). Providing TPM/RPM for a batch judge is an error.
