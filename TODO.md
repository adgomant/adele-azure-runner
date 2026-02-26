# Implementation TODO (execute in order)

## Phase 0 — Repo scaffold
- [ ] Create `pyproject.toml` (project metadata + deps + tools) for `uv`
- [ ] Create `src/` package layout and `tests/`
- [ ] Add `README.md` with quickstart placeholders
- [ ] Add `.gitignore` (python, outputs/, .venv, etc.)

## Phase 1 — Core schemas & config
- [ ] Implement `src/config.py` with pydantic settings (load YAML + env overrides)
- [ ] Define internal schemas (TypedDict/dataclasses/pydantic models):
      InferenceOutput, JudgeOutput, Manifest, etc.

## Phase 2 — Dataset loader
- [ ] Implement `src/datasets/adele.py`:
      load HF dataset, normalize columns (instance_id, prompt, ground_truth),
      support subset (first N), and deterministic ordering.

## Phase 3 — IO + resume/dedup
- [ ] Implement `src/utils/io.py` JSONL append, streaming read, dedup index
- [ ] Implement run directories: `runs/<run_id>/...` with consistent filenames
- [ ] Implement idempotent skip logic: instance_id+model_id

## Phase 4 — Inference adapters
- [ ] Implement Foundry adapter using `azure-ai-inference` ChatCompletionsClient
- [ ] Implement asyncio runner with bounded concurrency + retries + checkpointing
- [ ] (Optional fast path) Implement Azure OpenAI Batch adapter (JSONL + files + batches)

## Phase 5 — Judge stage
- [ ] Implement judge prompt template(s)
- [ ] Implement judge runner (multi-judge, per instance)
- [ ] Implement strict JSON parsing + fallback repair parser
- [ ] Store raw outputs + prompt text for audit

## Phase 6 — Merge + metrics + reporting
- [ ] Merge inference + judge outputs into parquet
- [ ] Metrics summary:
      - map score to binary (configurable threshold)
      - agreement between judges (percent + Cohen’s kappa)
      - list top disagreement cases

## Phase 7 — CLI
- [ ] Typer CLI with commands:
      run-inference, run-judge, merge-results, summarize
- [ ] Provide example commands in README

## Phase 8 — Tests
- [ ] Parsers + schema validation tests
- [ ] Resume/dedup tests
- [ ] Small “dry run” test with mocked adapters

## Phase 9 — Polishing
- [ ] Ruff/formatting config
- [ ] Logging configuration
- [ ] Final README quickstart + example run
