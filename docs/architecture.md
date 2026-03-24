# Architecture

## Core Model

The runner now separates three concerns that used to be mixed together:

- `provider`: the exact SDK family
- `mode`: how that SDK is queried
- `stage`: what the runner is doing with the model

Concrete examples:

- provider: `azure_ai_inference`, `azure_openai`, `google_genai`, `anthropic`
- mode: `request_response`, `batch`, `auto` before resolution
- stage: `inference`, `judging`

This means:

- adapters own SDK calls
- stages own prompt construction and response interpretation
- executors own concurrency, retries, polling, and rate limiting

## Package Layout

```text
src/adele_runner/
  cli.py
  config.py
  schemas.py

  adapters/
    providers/
      azure_ai_inference/
        request_response.py
      azure_openai/
        request_response.py
        batch.py
      google_genai/
        request_response.py
        batch.py
      anthropic/
        request_response.py
        batch.py
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

The top-level adapter modules are compatibility shims. Real implementations live under `adapters/providers/<provider>/<mode>.py`.

## Dataflow

```text
config.yaml / CLI
    ↓
config normalization + validation
    ↓
runtime.resolution
    ↓
ResolvedProviderTarget + ResolvedModeBinding + ExecutionSettings
    ↓
stage builds ChatRequest
    ↓
executor runs concrete adapter
    ↓
adapter returns ChatResponse
    ↓
stage maps to public output schema
    ↓
pipeline runner persists artifacts
```

The normalized transport flow is:

- `DatasetItem` or `InferenceOutput`
- `ChatRequest`
- provider SDK call
- `ChatResponse`
- `InferenceOutput` or `JudgeOutput`

## Transport Contracts

Internal transport contracts live in [types.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/runtime/types.py).

- `ChatMessage`: normalized `{role, content}`
- `ChatRequest`: request id, model, messages, generation params, metadata
- `ChatResponse`: normalized content, token usage, finish reason, raw output, metadata
- `AdapterCapabilities`: static support flags
- `ResolvedProviderTarget`: provider + model + target metadata + rate limits
- `ResolvedModeBinding`: resolved mode plus adapter factory
- `ExecutionSettings`: per-stage execution settings

Adapters only speak `ChatRequest` and `ChatResponse`. They do not know about `InferenceOutput` or `JudgeOutput`.

## Provider Registry

[registry.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/runtime/registry.py) is the decoupling point between provider and mode.

Each provider is described by a `ProviderDescriptor` with:

- request-response adapter factory
- batch adapter factory
- capability resolver

The capability resolver is where static and dynamic support checks happen.

Examples:

- `azure_ai_inference` supports request-response only
- `azure_openai` supports both modes, but batch requires target metadata showing the deployment is batch-capable
- `google_genai` supports both modes and can run against `gemini_api` or `vertex_ai`
- `anthropic` supports both modes

Executors never branch on provider. They receive a concrete adapter instance that was already selected by the registry and resolution layers.

## Resolution

[resolution.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/runtime/resolution.py) turns public config into executable internal targets.

For inference it resolves:

- provider
- requested mode
- effective mode after `auto`
- model/deployment target metadata
- rate limits
- execution settings

For judging it does the same per judge, plus:

- judge name
- judge prompt template
- judge `max_tokens`

The runtime no longer mutates shared concurrency config between stages. Each stage gets its own resolved `ExecutionSettings`.

## Stages

### Inference

[stages/inference.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/stages/inference.py):

- converts a dataset item into `ChatRequest`
- maps `ChatResponse` into `InferenceOutput`

### Judging

[stages/judging.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/stages/judging.py):

- renders judge prompts
- parses `v1` and `v2` responses
- maps `ChatResponse` into `JudgeOutput`

Stage logic contains no SDK calls.

## Executors

[executors.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/runtime/executors.py) contains the shared runtime engines.

`RequestResponseExecutor` handles:

- bounded async concurrency
- retries
- adaptive rate limiting
- result callbacks for checkpointing

`BatchExecutor` handles:

- running a concrete batch adapter
- moving blocking batch work to a worker thread
- emitting results back to the pipeline

Neither executor chooses the provider.

## Pipeline Runners

- [inference_runner.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/pipeline/inference_runner.py)
- [judge_runner.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/pipeline/judge_runner.py)

These files are now orchestration-only. They manage:

- dedup and resume
- run directory setup
- artifact persistence
- dispatching resolved lanes

They no longer embed provider-specific SDK logic or judge parsing logic.

## Compatibility

The new public model is `provider + mode`, but the code still accepts older config/CLI forms for one release cycle:

- `foundry`
- `google`
- `batch` as a pseudo-provider

Those are normalized immediately into the new model and should be considered deprecated.
