# Architecture

## Design Axes

The runner now separates its internal logic across three independent axes:

- **Adapters** — provider/SDK integrations (`foundry`, `google_genai`, `azure_openai`)
- **Stages** — benchmark logic (`inference`, `judging`)
- **Execution modes** — transport style (`request_response`, `batch`)

This keeps provider SDK code out of the stage logic and keeps stage-specific prompting/parsing out of the transport layer.

## Module Map

```
src/adele_runner/
  __init__.py
  cli.py                    ← Typer CLI entry point
  config.py                 ← Pydantic config models + YAML loader
  schemas.py                ← Public output schemas

  adapters/
    foundry.py              ← Azure AI Foundry request-response transport
    google_genai.py         ← Google Gemini request-response transport
    azure_openai.py         ← Azure OpenAI batch transport
    factory.py              ← Adapter construction helpers
    foundry_inference.py    ← Compatibility shim for older imports
    azure_openai_batch.py   ← Compatibility shim for older imports

  runtime/
    types.py                ← Internal transport contracts and resolved targets
    resolution.py           ← Config → internal targets/execution settings
    executors.py            ← Shared request-response and batch executors

  stages/
    inference.py            ← Build inference requests and map responses
    judging.py              ← Judge prompts, parsing, and output mapping

  datasets/
    adele.py                ← ADeLe HuggingFace dataset loader

  pipeline/
    inference_runner.py     ← Thin orchestration for inference
    judge_runner.py         ← Thin orchestration for judging
    merge.py                ← Merge inference + judge → Parquet
    metrics.py              ← Score stats, Cohen's kappa, token usage

  utils/
    io.py                   ← JSONL/Parquet I/O, dedup index
    retry.py                ← Tenacity retry with smart error classification
    concurrency.py          ← Bounded async concurrency + adaptive rate limiter
    batch_split.py          ← Split large batch request files to respect Azure limits
```

## Internal Flow

```
config.yaml / CLI flags
          │
          ▼
runtime.resolution
  ├── ResolvedInferenceTarget
  ├── ResolvedJudgeTarget[]
  └── ExecutionSettings
          │
          ├──────────────────────────────┐
          ▼                              ▼
stages.inference                    stages.judging
  build ChatRequest                   build ChatRequest
  map ChatResponse → InferenceOutput  parse ChatResponse → JudgeOutput
          │                              │
          └──────────────┬───────────────┘
                         ▼
                 runtime.executors
           ├── RequestResponseExecutor
           └── BatchExecutor
                         │
                         ▼
                      adapters
           ├── FoundryAdapter
           ├── GoogleGenAIAdapter
           └── AzureOpenAIAdapter
```

The pipeline runners coordinate dedup, artifact writing, and run-level flow, but they no longer embed judge prompt templates, judge parsers, or provider-specific transport code.

## Transport Contracts

The adapter layer uses normalized internal contracts:

- `ChatRequest` — request id, model, messages, generation params, metadata
- `ChatResponse` — request id, content, token usage, finish reason, raw metadata
- `AdapterCapabilities` — whether the adapter supports `request_response` and/or `batch`
- `ResolvedInferenceTarget` / `ResolvedJudgeTarget` — explicit internal targets derived from the public config
- `ExecutionSettings` — per-stage resolved concurrency, timeout, retry, and batch polling settings

These are internal-only. Public output schemas remain `InferenceOutput` and `JudgeOutput`.

## Adapter Pattern

All provider-specific SDK code lives in `adapters/`.

| Adapter | Module | Capabilities | SDK |
|---|---|---|---|
| `FoundryAdapter` | `adapters/foundry.py` | `request_response` | `azure-ai-inference` |
| `GoogleGenAIAdapter` | `adapters/google_genai.py` | `request_response` | `google-genai` |
| `AzureOpenAIAdapter` | `adapters/azure_openai.py` | `batch` | `openai` |

The same Foundry adapter is used for both inference and judging. The difference between those stages is in how requests are built and how responses are interpreted, not in which adapter is used.

## Stage Logic

### Inference

`stages/inference.py` is responsible for:

- turning each `DatasetItem` into a `ChatRequest`
- mapping each `ChatResponse` into an `InferenceOutput`

### Judging

`stages/judging.py` is responsible for:

- rendering judge prompts (`v1` JSON, `v2` bare integer)
- parsing judge responses with fallback logic
- mapping each `ChatResponse` into a `JudgeOutput`

This keeps prompt templates and parsing out of `pipeline/judge_runner.py`.

## Execution Model

### Request-response

`RequestResponseExecutor` handles:

- bounded async concurrency via `bounded_gather()`
- retries via `utils.retry.make_retry_decorator()`
- adaptive rate limiting via `AsyncRateLimiter`
- per-response callbacks for checkpointing

This is the internal name for the non-batch path. The executor is async even when the underlying SDK exposes blocking calls.

### Batch

`BatchExecutor` handles:

- adapter selection for batch-capable providers
- running the batch workflow in a worker thread
- per-response callbacks after batch completion

Provider-specific upload/poll/download details stay inside the batch adapter.

## Config Resolution

The public config surface remains unchanged, but runtime execution no longer mutates `config.concurrency` between stages. Instead, `runtime.resolution` builds explicit internal targets and per-stage `ExecutionSettings`.

Current mappings:

| Public field | Internal target |
|---|---|
| `inference.mode=auto` | `adapter=foundry`, `execution=request_response` |
| `inference.mode=foundry` | `adapter=foundry`, `execution=request_response` |
| `inference.mode=google` | `adapter=google_genai`, `execution=request_response` |
| `inference.mode=batch` | `adapter=azure_openai`, `execution=batch` |
| `judging.judges[].provider=foundry` | `adapter=foundry`, `execution=request_response` |
| `judging.judges[].provider=batch` | `adapter=azure_openai`, `execution=batch` |

Inference and judging each get their own resolved `ExecutionSettings`, so rate-limit tuning is phase-local and does not overwrite shared config state.

## Retry Strategy

The retry module (`utils/retry.py`) wraps [tenacity](https://tenacity.readthedocs.io/) with smart error classification:

**Retryable** (transient):

- connection errors, timeouts
- HTTP 429 (rate limit)
- HTTP 500, 502, 503, 504 (server errors)

**Not retryable** (permanent):

- Python logic errors (`ValueError`, `TypeError`, `KeyError`, `AttributeError`)
- HTTP 400, 401, 403, 404, 405, 422

**Unknown** exceptions are retried (fail-open).

Backoff is exponential, starting at `backoff_base_s` and capped at `backoff_max_s`. When a 429 response includes `Retry-After` or `x-ratelimit-reset-tokens` headers, the retry system uses the server-specified wait time instead of generic exponential backoff.
