# Architecture

## Module Map

```
src/adele_runner/
  __init__.py
  cli.py                  ← Typer CLI entry point
  config.py               ← Pydantic config models + YAML loader
  schemas.py              ← Data models (InferenceOutput, JudgeOutput, ...)

  adapters/
    foundry_inference.py  ← Azure AI Foundry async adapter
    azure_openai_batch.py ← Azure OpenAI Batch API adapter

  datasets/
    adele.py              ← ADeLe HuggingFace dataset loader

  pipeline/
    inference_runner.py   ← Inference orchestration (async + batch dispatch)
    judge_runner.py       ← Multi-judge evaluation (foundry + batch)
    merge.py              ← Merge inference + judge → Parquet
    metrics.py            ← Score stats, Cohen's kappa, token usage

  utils/
    io.py                 ← JSONL/Parquet I/O, dedup index
    retry.py              ← Tenacity retry with smart error classification
    concurrency.py        ← Bounded async concurrency (semaphore)
```

## Data Flow

```
                  ┌──────────┐
                  │  config  │
                  │  .yaml   │
                  └────┬─────┘
                       │
                       ▼
┌────────────┐     ┌─────────────┐     ┌───────────────────┐
│ HuggingFace│ ──▶ │ load_adele()│ ──▶ │ list[DatasetItem] │
│ dataset    │     └─────────────┘     └────────┬──────────┘
└────────────┘                                  │
                                                ▼
                                    ┌────────────────────┐
                                    │  run_inference()   │
                                    │  (inference_runner)│
                                    └────────┬───────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                                     ▼
                  ┌────────────────┐                   ┌─────────────────┐
                  │ FoundryAdapter │                   │ AzureOpenAI     │
                  │ (async)        │                   │ BatchAdapter    │
                  └───────┬────────┘                   └────────┬────────┘
                          │                                     │
                          └──────────────┬──────────────────────┘
                                         ▼
                               ┌──────────────────┐
                               │  outputs.jsonl   │
                               └────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │   run_judge()     │
                              │  (judge_runner)   │
                              └────────┬──────────┘
                                       │
                       ┌───────────────┼─────────────┐
                       ▼                             ▼
                ┌──────────────┐             ┌────────────────┐
                │ JudgeAdapter │             │ JudgeBatch     │
                │ (foundry)    │             │ Adapter (batch)│
                └──────┬───────┘             └───────┬────────┘
                       │                             │
                       └──────────┬──────────────────┘
                                  ▼
                        ┌───────────────────┐
                        │judge_outputs.jsonl│
                        └────────┬──────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ merge_results()  │
                       └────────┬─────────┘
                                │
                                ▼
                   ┌───────────────────────┐
                   │ merged_results.parquet│
                   └───────────┬───────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │   summarize()    │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  metrics.json    │
                      │  (+ console)     │
                      └──────────────────┘
```

## Adapter Pattern

All infrastructure-specific code lives in `adapters/`. The pipeline modules depend on adapter interfaces, not SDK internals.

### Inference Adapters

| Adapter | Module | SDK | Use case |
|---|---|---|---|
| `FoundryAdapter` | `foundry_inference.py` | `azure-ai-inference` | Any model deployed in Azure AI Foundry |
| `AzureOpenAIBatchAdapter` | `azure_openai_batch.py` | `openai` | Azure OpenAI deployments with Batch API support |

Both adapters produce `InferenceOutput` objects. The `inference_runner` dispatches to the appropriate adapter based on `config.resolve_inference_mode()`.

### Judge Adapters

| Adapter | Module | Transport |
|---|---|---|
| `JudgeAdapter` | `judge_runner.py` | Foundry async (same SDK as inference) |
| `JudgeBatchAdapter` | `judge_runner.py` | Azure OpenAI Batch API |

Both produce `JudgeOutput` objects. Foundry and batch judges run concurrently via `asyncio.gather()`.

## Concurrency Model

Foundry async operations use **bounded concurrency** via `asyncio.Semaphore`:

```
bounded_gather(tasks, max_concurrency=16)
    │
    ├── Semaphore(16) controls parallelism
    ├── Tasks execute as coroutines
    ├── Results returned in input order
    └── Exceptions captured (not raised immediately)
```

The semaphore limit is set by `concurrency.max_in_flight` (default: 16). Items are processed in chunks of `max_in_flight * 4` to avoid loading all tasks into memory at once.

Batch operations run synchronously (in a thread) since the Batch API is a submit-poll-download workflow.

## Retry Strategy

The retry module (`utils/retry.py`) wraps [tenacity](https://tenacity.readthedocs.io/) with smart error classification:

**Retryable** (transient):
- Connection errors, timeouts
- HTTP 429 (rate limit)
- HTTP 500, 502, 503, 504 (server errors)

**Not retryable** (permanent):
- Python logic errors (`ValueError`, `TypeError`, `KeyError`, `AttributeError`)
- HTTP 400, 401, 403, 404, 405, 422

**Unknown** exceptions are retried (fail-open).

Backoff is exponential, starting at `backoff_base_s` and capped at `backoff_max_s`.

## Config System

```
config.yaml
    │
    ▼
load_config()           ← YAML → dict
    │
    ▼
AppConfig.model_validate()  ← Pydantic validation
    │
    ▼
apply_cli_overrides()   ← --model, --mode, --judge, --judge-template
    │
    ▼
validate_config()       ← Placeholder detection, required fields
```

Pydantic models define all fields with defaults, so a minimal YAML works. CLI flags override specific values. Validation catches common mistakes (placeholder endpoints, missing models) before API calls.
