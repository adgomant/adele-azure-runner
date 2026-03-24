# Batch Mode

Batch is now an execution mode, not a provider name.

Supported batch providers in the current runtime:

- `azure_openai`
- `google_genai`
- `anthropic`

`azure_ai_inference` is request-response only.

## When to Use It

Batch is appropriate when:

- the provider supports batch
- the target supports batch
- throughput matters more than low latency

Request-response is better when:

- you need fast interactive responses
- the target is not batch-capable
- you want request-level adaptive pacing

## Provider Notes

### Azure OpenAI

Configured under `providers.azure_openai`.

Batch validation requires target hints such as:

```yaml
targets:
  gpt-4o-batch:
    supported_modes: [batch]
    batch_capable: true
    deployment_type: global_batch
```

### Google GenAI

Configured under `providers.google_genai`.

Batch is supported for:

- `backend: gemini_api`
- `backend: vertex_ai`

`vertex_ai` requires `project` and `location`.

### Anthropic

Configured under `providers.anthropic`.

Batch uses the Anthropic Message Batches API.

## Inference Example

```yaml
providers:
  azure_openai:
    endpoint: "https://my-resource.openai.azure.com"
    api_version: "2024-10-21"

targets:
  gpt-4o-batch:
    supported_modes: [batch]
    batch_capable: true

inference:
  provider: azure_openai
  mode: batch
  model: gpt-4o-batch
```

## Judge Example

```yaml
judging:
  judges:
    - name: gpt4o-batch
      provider: azure_openai
      mode: batch
      model: gpt-4o-batch
    - name: gemini-batch
      provider: google_genai
      mode: batch
      model: gemini-2.5-flash
    - name: claude-batch
      provider: anthropic
      mode: batch
      model: claude-sonnet-4-5
```

## Internal Lifecycle

1. the stage builds normalized `ChatRequest` values
2. `BatchExecutor` receives a concrete provider batch adapter
3. the adapter performs provider-specific upload, create, poll, and result collection
4. the adapter returns normalized `ChatResponse` values
5. the stage maps them into public outputs

The executor itself contains no provider branching.

## Polling and Limits

Batch polling is controlled by:

```yaml
concurrency:
  max_poll_time_s: 3600
  batch_completion_window: 24h
```

Azure OpenAI file splitting uses:

```yaml
providers:
  azure_openai:
    max_requests_per_file: 50000
    max_bytes_per_file: 100000000
```
