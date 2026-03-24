# Batch Mode

Azure OpenAI Batch API provides a cost-effective way to run large-scale inference and judging. The runner supports batch mode for both inference and judge evaluation.

## When to Use Batch Mode

Use batch mode when:

- Your target model is deployed as an Azure OpenAI resource (not just Foundry)
- The deployment supports the [Batch API](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch)
- You prioritize throughput and cost over latency (batch jobs can take minutes to hours)

Use Foundry async mode when:

- Your model is deployed only in Azure AI Foundry
- You need results quickly (async responses are near-real-time)
- The model/deployment does not support the Batch API

## Config Requirements

```yaml
azure:
  batch:
    endpoint: "https://<YOUR-AOAI-RESOURCE>.openai.azure.com"
    api_key_env: "AZURE_OPENAI_API_KEY"
    api_version: "<API_VERSION>"
    max_requests_per_file: 50000     # Max requests per batch file (Azure limit: 100,000)
    max_bytes_per_file: 100000000    # Max bytes per batch file (Azure limit: 200 MB)

inference:
  mode: "batch"
  model: "<DEPLOYMENT_NAME>"
```

Set the API key:

```bash
export AZURE_OPENAI_API_KEY="<your-key>"
```

## Batch Lifecycle

Internally, batch transport is handled by `adapters/azure_openai.py`, while the stage-specific request content comes from `stages/inference.py` or `stages/judging.py`. The shared `BatchExecutor` runs the blocking batch workflow in a worker thread so the pipeline can still coordinate request-response and batch lanes concurrently.

### Inference

```
1. Build JSONL
   Each DatasetItem becomes a batch request line:
   {
     "custom_id": "<instance_id>",
     "method": "POST",
     "url": "/chat/completions",
     "body": {
       "model": "<deployment>",
       "messages": [{"role": "user", "content": "<prompt>"}],
       "temperature": 0.0,
       "max_tokens": 2048,
       "top_p": 1.0
     }
   }

2. Upload JSONL file
   → client.files.create(file=..., purpose="batch")

3. Create batch job
   → client.batches.create(input_file_id=..., endpoint=..., completion_window=...)

4. Poll for completion
   → client.batches.retrieve(batch_id) every ~30 seconds
   → Status progression: validating → in_progress → completed / failed / expired

5. Download results
   → client.files.content(output_file_id)
   → Parse each line: extract response content, token usage, finish reason

6. Convert to InferenceOutput
   → Append to outputs.jsonl
```

### Judging

The same lifecycle applies to batch judges. Each judge with `provider: "batch"` submits its own batch job. The request body includes the judge prompt with the model output and ground truth embedded.

## Polling and Timeouts

The runner polls the Batch API every ~30 seconds. The maximum poll time is controlled by:

```yaml
concurrency:
  max_poll_time_s: 3600    # 1 hour (default)
```

If the batch job does not complete within this window, the runner raises `TimeoutError`. Increase `max_poll_time_s` for large batch jobs or set a generous `batch_completion_window` in the concurrency config.

## Inference Parameters

Batch requests use the same inference parameters as Foundry mode:

| Parameter | Config field | Default |
|---|---|---|
| `temperature` | `inference.temperature` | `0.0` |
| `max_tokens` | `inference.max_tokens` | `2048` |
| `top_p` | `inference.top_p` | `1.0` |

## Mixing Batch and Foundry Judges

You can mix both provider types in the same run:

```yaml
judging:
  judges:
    - name: "gpt4o-batch"
      provider: "batch"
      model: "gpt-4o"
    - name: "claude-foundry"
      provider: "foundry"
      model: "claude-3-opus"
```

Or from the CLI:

```bash
uv run adele-runner run-judge --judge gpt-4o:batch --judge claude-3-opus
```

Foundry judges run as async coroutines while batch judges run in a separate thread. Both execute concurrently via `asyncio.gather()`.

Internally, this is implemented as two executor lanes:

- `RequestResponseExecutor` for Foundry judges
- `BatchExecutor` for Azure OpenAI batch judges

## Automatic File Splitting

Azure imposes hard limits on batch files: 100,000 requests and 200 MB per file. When a batch request set exceeds the configured limits, the runner automatically splits it into multiple chunks.

Splitting is controlled by two config fields under `azure.batch`:

| Field | Default | Azure hard limit | Description |
|---|---|---|---|
| `max_requests_per_file` | `50,000` | `100,000` | Maximum requests per batch file |
| `max_bytes_per_file` | `100,000,000` (100 MB) | `200,000,000` (200 MB) | Maximum bytes per batch file |

The defaults are conservative (50% of Azure's hard limits) to provide a safety margin. Each chunk is uploaded and processed as a separate batch job, and results are combined automatically.

This applies to both inference and judge batch operations. Config validation rejects values that exceed Azure's hard limits.

## Optional Dependency

Batch mode requires the `openai` package. Install it with:

```bash
uv sync --extra batch
```

If `openai` is not installed and batch mode is requested, the runner will raise an `ImportError`.
