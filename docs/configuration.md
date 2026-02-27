# Configuration Reference

ADeLe Runner loads configuration from a YAML file and optionally overrides values with CLI flags.

## Loading Order

1. **YAML file** -- `config.yaml` by default, or the path given via `--config / -c`
2. **CLI overrides** -- flags like `--model`, `--mode`, `--judge`, `--judge-template` mutate the loaded config in-place
3. **Environment variables** -- API keys are read from env vars named in the config (never stored in the YAML)

If no `--config` is provided and `config.yaml` does not exist in the working directory, the CLI exits with an error.

## Config Sections

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
| `mode` | `"auto" \| "foundry_async" \| "azure_openai_batch"` | `"auto"` | Inference mode. See [Mode resolution](#mode-resolution) below |

#### `inference.foundry`

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | `""` | Azure AI Foundry endpoint URL (e.g. `https://<resource>.services.ai.azure.com/models`) |
| `api_key_env` | `str` | `"AZURE_AI_API_KEY"` | Name of the environment variable holding the API key |
| `model` | `str` | `""` | Model name or deployment ID. Overridden by `--model` |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Maximum completion tokens |
| `top_p` | `float` | `1.0` | Top-p (nucleus) sampling |

#### `inference.batch`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable Azure OpenAI Batch API path |
| `azure_openai_endpoint` | `str` | `""` | Azure OpenAI resource endpoint |
| `api_key_env` | `str` | `"AZURE_OPENAI_API_KEY"` | Env var for the batch API key |
| `deployment` | `str` | `""` | Azure OpenAI deployment name |
| `api_version` | `str` | `""` | Azure OpenAI API version string |
| `completion_endpoint` | `str` | `"/chat/completions"` | Completion endpoint path used in batch requests |
| `completion_window` | `str` | `"24h"` | Batch completion window |

### `concurrency`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_in_flight` | `int` | `16` | Maximum concurrent async requests |
| `request_timeout_s` | `float` | `120.0` | Per-request HTTP timeout in seconds |
| `max_retries` | `int` | `6` | Maximum retry attempts per request |
| `backoff_base_s` | `float` | `1.0` | Exponential backoff base (seconds) |
| `backoff_max_s` | `float` | `30.0` | Maximum backoff wait (seconds) |
| `max_poll_time_s` | `float` | `3600.0` | Maximum time to poll a batch job before raising `TimeoutError` |

### `judging`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Enable the judging stage |
| `prompt_template` | `str` | `"v1"` | Judge prompt template: `"v1"` (structured JSON) or `"v2"` (bare integer). See [Judging](judging.md) |
| `require_json` | `bool` | `true` | Whether to require JSON output from judges |
| `judges` | `list` | `[]` | List of judge configurations (see below) |

Each judge entry:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Display name for this judge |
| `provider` | `"foundry" \| "batch"` | `"foundry"` | Whether to use Foundry async or Azure OpenAI Batch API |
| `model` | `str` | *(required)* | Model name or deployment to use for judging |

### `logging`

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### `pricing`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable cost estimation in metrics summary |
| `models` | `dict` | `{}` | Per-model pricing (see below) |

Each model pricing entry (keyed by model name):

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_per_1k` | `float` | `0.0` | Cost in USD per 1,000 prompt tokens |
| `completion_per_1k` | `float` | `0.0` | Cost in USD per 1,000 completion tokens |

## Environment Variables

| Variable | Config field | Used by |
|---|---|---|
| `AZURE_AI_API_KEY` | `inference.foundry.api_key_env` | Foundry inference and Foundry judges |
| `AZURE_OPENAI_API_KEY` | `inference.batch.api_key_env` | Batch inference and Batch judges |

The variable names are configurable. The runner reads the env var whose name is specified in the config, not a hardcoded name.

## Mode Resolution

The `inference.mode` field determines how inference requests are dispatched:

| `inference.mode` | `inference.batch.enabled` | Effective mode |
|---|---|---|
| `"auto"` | `true` | `azure_openai_batch` |
| `"auto"` | `false` | `foundry_async` |
| `"foundry_async"` | *(ignored)* | `foundry_async` |
| `"azure_openai_batch"` | *(ignored)* | `azure_openai_batch` |

CLI shorthand: `--mode foundry` maps to `foundry_async`, `--mode batch` maps to `azure_openai_batch`.

## Config Validation

The runner validates config before making API calls. Validation checks:

- Foundry endpoint is set and does not contain `<YOUR-` placeholders
- Batch endpoint is set when batch mode is active
- Model or deployment is specified
- At least one judge is configured when judging is enabled
- `prompt_template` is `"v1"` or `"v2"`

Use `--dry-run` to test config without hitting APIs. Dry-run skips API key validation.

## Annotated Example

See [config.example.yaml](../config.example.yaml) for a fully annotated config file, or the [examples/](../examples/) directory for scenario-specific configs.
