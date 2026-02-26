# ADeLe Runner + Multi-Judge Framework (Azure AI Foundry)

## Goal
Build a production-grade Python framework to run the ADeLe benchmark at scale against models deployed in Azure AI Foundry, and to evaluate outputs using multiple LLM judges (at least GPT-4o and Claude). The framework must support:

1) Inference execution (primary)
- Default universal path: Azure AI Foundry Models via Azure AI Model Inference API (chat completions).
- Same code should run prompts against any Foundry-deployed model (GPT, Llama, Qwen, DeepSeek, Grok, Claude, etc.) by changing config only.
- Clean adapter layer so providers/deployments can be swapped by config.

2) Batch execution (fast-path when available)
- If the target model is Azure OpenAI and supports Batch API, implement batch mode using JSONL + files upload + batches create + results download.
- Detect batch capability via config flag or capability metadata.
- If batch is not supported, emulate batch via asyncio bounded concurrency, retries with exponential backoff, and durable checkpointing (safe resume).

3) Judging
- Run judges as a second stage over model outputs.
- Support multiple judges per instance (e.g., gpt-4o AND claude) and store each judge result separately.
- Enforce robust parseable judge output format. Prefer strict JSON schema:
  {"score": int (1..5), "verdict": "correct|incorrect|partial|unknown", "reason": str}
- Provide a fallback parser if malformed JSON (attempt repair + strict validation).
- Store raw judge outputs and the full judge prompt for auditability.

4) Data + storage
- Load ADeLe from Hugging Face dataset "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0" (train split by default), but implement a dataset interface so other benchmarks can be plugged in.
- Stable internal schemas and durable artifacts:
  - outputs.jsonl for model outputs (append-only; resume supported; de-duplicate by instance_id + model_id)
  - judge_outputs.jsonl for judge results
  - merged_results.parquet containing prompt, ground_truth, model_response, judge scores, metadata
  - run_manifest.json capturing dataset revision, model identifiers, parameters, timestamps, code version.
- Idempotency: reruns should skip completed instances and only process missing ones.

5) Architecture requirements
- Provide a CLI (Typer preferred) with commands:
  - prepare-requests (optional; can be a no-op if streaming)
  - run-inference
  - run-judge
  - merge-results
  - summarize (metrics + judge agreement)
- Python 3.11+, structured logging, type hints, and clear module boundaries.

Suggested structure:
src/
  config.py (pydantic settings)
  datasets/adele.py
  adapters/
    foundry_inference.py
    azure_openai_batch.py (optional fast path)
  pipeline/
    inference_runner.py (batch + asyncio emulate)
    judge_runner.py
    merge.py
    metrics.py
  utils/
    io.py (jsonl/parquet)
    retry.py
    concurrency.py

Tests:
tests/
  test_parsing.py
  test_resume_dedup.py
  test_schema_validation.py

6) Operational constraints
- Handle rate limits and transient failures.
- Bounded memory: stream read/write, avoid loading huge outputs in memory.
- Run on laptop and on server: sensible defaults.

7) Output deliverables
- Complete codebase with README: setup, authentication, config, example commands.
- Minimal example run:
  - run ADeLe subset (e.g., first 50 instances) on one model
  - judge with two judges
  - produce metrics summary.

Important constraints
- Do NOT hardcode secrets.
- Use environment variables or Azure identity.
- Avoid vendor-specific logic outside adapter modules.
- Do not invent endpoints/IDs: use config placeholders.
- Prefer Foundry inference via azure-ai-inference.
- If structured outputs not supported by a judge model, fall back to regex parsing.

Tooling requirement
- Use `uv` for dependency management.
- Provide `pyproject.toml` and generate `uv.lock`.
- Document `uv venv`, `uv sync`, and `uv run` commands in README.

Workflow requirement
- Before coding, propose design, schemas, and config format.
- Then implement step-by-step (small commits / incremental).
