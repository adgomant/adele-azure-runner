# Configuration Reference

The public config now models runtime selection explicitly:

- `provider`: which SDK to use
- `mode`: how to query that SDK

This applies to both inference and judging.

## Loading Order

1. `.env.local` and `.env`
2. YAML config file
3. CLI overrides
4. API keys are read from the env var names stored in config

## Top-Level Sections

- `providers`
- `run`
- `dataset`
- `targets`
- `inference`
- `concurrency`
- `judging`
- `logging`
- `pricing`

## `providers`

### `providers.azure_ai_inference`

| Field | Type | Default |
|---|---|---|
| `endpoint` | `str` | `""` |
| `api_key_env` | `str` | `AZURE_AI_API_KEY` |

Supports `request_response` only.

### `providers.azure_openai`

| Field | Type | Default |
|---|---|---|
| `endpoint` | `str` | `""` |
| `api_key_env` | `str` | `AZURE_OPENAI_API_KEY` |
| `api_version` | `str` | `""` |
| `completion_endpoint` | `str` | `/chat/completions` |
| `max_requests_per_file` | `int` | `100000` |
| `max_bytes_per_file` | `int` | `200000000` |

Supports `request_response` and `batch`, but batch requires target capability hints.

### `providers.google_genai`

| Field | Type | Default |
|---|---|---|
| `api_key_env` | `str` | `GEMINI_API_KEY` |
| `backend` | `gemini_api \| vertex_ai` | `gemini_api` |
| `project` | `str` | `""` |
| `location` | `str` | `""` |

Supports `request_response` and `batch`.

When `backend=vertex_ai`, both `project` and `location` are required.

### `providers.anthropic`

| Field | Type | Default |
|---|---|---|
| `api_key_env` | `str` | `ANTHROPIC_API_KEY` |
| `base_url` | `str` | `""` |

Supports `request_response` and `batch`.

## `targets`

`targets` stores optional deployment capability metadata.

```yaml
targets:
  gpt-4o-batch:
    supported_modes: [batch]
    batch_capable: true
    deployment_type: global_batch
```

Fields:

| Field | Type |
|---|---|
| `supported_modes` | `list[request_response \| batch] \| null` |
| `batch_capable` | `bool \| null` |
| `deployment_type` | `str \| null` |

This is especially important for `azure_openai` batch validation.

## `inference`

| Field | Type | Default |
|---|---|---|
| `provider` | `azure_openai \| azure_ai_inference \| google_genai \| anthropic` | `azure_ai_inference` |
| `mode` | `request_response \| batch \| auto` | `auto` |
| `model` | `str` | `""` |
| `temperature` | `float` | `0.0` |
| `max_tokens` | `int` | `2048` |
| `top_p` | `float` | `1.0` |
| `rate_limits` | `object \| null` | `null` |

`mode=auto` resolves within the chosen provider, not across providers.

`rate_limits` supports both the legacy minute budgets and provider-specific extensions:

- `tokens_per_minute`
- `requests_per_minute`
- `input_tokens_per_minute`
- `output_tokens_per_minute`
- `requests_per_day`
- `tokens_per_day`
- `concurrent_requests`
- `batch_requests_per_minute`
- `batch_queue_requests`
- `batch_enqueued_tokens`

For `google_genai` with `backend: gemini_api`, daily budgets such as `requests_per_day` follow the Gemini API reset window at midnight Pacific time.

## `concurrency`

| Field | Type | Default |
|---|---|---|
| `max_in_flight` | `int` | `16` |
| `request_timeout_s` | `float` | `120.0` |
| `max_retries` | `int` | `6` |
| `backoff_base_s` | `float` | `1.0` |
| `backoff_max_s` | `float` | `30.0` |
| `max_poll_time_s` | `float` | `3600.0` |
| `batch_completion_window` | `str` | `24h` |

These are base values. The runtime resolves per-stage `ExecutionSettings` instead of mutating this object.

## `judging`

| Field | Type | Default |
|---|---|---|
| `enabled` | `bool` | `true` |
| `prompt_template` | `v1 \| v2` | `v1` |
| `judges` | `list` | `[]` |

Each judge:

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `provider` | provider enum | `azure_ai_inference` |
| `mode` | `request_response \| batch \| auto` | `request_response` |
| `model` | `str` | required |
| `rate_limits` | `object \| null` | `null` |
| `max_tokens` | `int` | `512` |

Request-response judges use `rate_limits` for adaptive pacing. Batch judges use the same block for queue and enqueued-token budgets when configured.

## Environment Variables

| Variable | Typical config field |
|---|---|
| `AZURE_AI_API_KEY` | `providers.azure_ai_inference.api_key_env` |
| `AZURE_OPENAI_API_KEY` | `providers.azure_openai.api_key_env` |
| `GEMINI_API_KEY` | `providers.google_genai.api_key_env` |
| `ANTHROPIC_API_KEY` | `providers.anthropic.api_key_env` |

## Validation Rules

Validation happens before execution.

Examples:

- `azure_ai_inference + batch` is rejected
- `azure_openai + batch` requires target metadata proving batch capability
- `google_genai + vertex_ai` requires `project` and `location`
- inference requires a model
- judging requires at least one judge when enabled

Use `--dry-run` to validate a plan without making API calls.

## Rate-Limit Auto-Tuning

When request-response `rate_limits` are set, the runner computes stage-local execution settings from the tightest configured budget, including:

- TPM / RPM
- input and output TPM
- requests or tokens per day
- `concurrent_requests`
- `max_tokens`

That affects:

- `effective_rpm`
- `max_in_flight`
- `request_timeout_s`
- retry backoff

Batch lanes use provider-specific file or payload splitting plus optional queue budgets such as `batch_queue_requests` and `batch_enqueued_tokens`.

## Example

```yaml
providers:
  azure_ai_inference:
    endpoint: "https://my-foundry.services.ai.azure.com/models"
  azure_openai:
    endpoint: "https://my-openai-resource.openai.azure.com"
    api_version: "2024-10-21"
  google_genai:
    backend: gemini_api
  anthropic:
    api_key_env: ANTHROPIC_API_KEY

targets:
  gpt-4o-batch:
    supported_modes: [batch]
    batch_capable: true
    deployment_type: global_batch

inference:
  provider: azure_ai_inference
  mode: request_response
  model: gpt-4o
  rate_limits:
    tokens_per_minute: 80000
    requests_per_minute: 300

judging:
  enabled: true
  prompt_template: v1
  judges:
    - name: gpt4o-judge
      provider: azure_ai_inference
      mode: request_response
      model: gpt-4o
    - name: claude-batch
      provider: anthropic
      mode: batch
      model: claude-sonnet-4-5
      rate_limits:
        batch_queue_requests: 50000
```

## Breaking Change

Older config aliases and root compatibility blocks have been removed. Config files must use the current `providers.*` plus explicit `provider` and `mode` fields.
