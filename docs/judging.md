# Judging

The judge stage evaluates model outputs by sending them (along with the original prompt and ground truth) to one or more LLM judges. Each judge scores instances on a 1--5 scale.

## Judge Providers

| Provider | Transport | Config |
|---|---|---|
| `foundry` | Azure AI Foundry async (same as inference) | Uses `azure.foundry.endpoint` and `AZURE_AI_API_KEY` |
| `batch` | Azure OpenAI Batch API | Uses `azure.batch.endpoint` and `AZURE_OPENAI_API_KEY` |

Configure judges in `config.yaml`:

```yaml
judging:
  enabled: true
  prompt_template: "v1"
  judges:
    - name: "gpt4o-judge"
      provider: "foundry"
      model: "gpt-4o"
      # max_tokens: 512         # Max completion tokens (default: 512)
      # rate_limits:            # Optional per-judge rate limits (foundry only)
      #   tokens_per_minute: 80000
      #   requests_per_minute: 300
    - name: "claude-judge"
      provider: "foundry"
      model: "claude-3-opus"
    - name: "gpt4o-batch-judge"
      provider: "batch"
      model: "gpt-4o"
```

Or override from the CLI:

```bash
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus:foundry --judge gpt-4o:batch

# With per-judge rate limits and max_tokens
uv run adele-runner run-judge --judge gpt-4o:foundry:80000:300:1024 --judge claude-3-opus
```

## Prompt Templates

### v1 -- Structured JSON

The judge is asked to return a JSON object:

```json
{"score": 4, "verdict": "correct", "reason": "The answer matches the expected output."}
```

- `score`: integer 1--5
- `verdict`: one of `correct`, `incorrect`, `partial`, `unknown`
- `reason`: free-text explanation

This is the default template (`prompt_template: "v1"`).

### v2 -- Bare Integer

The judge is asked to return only an integer (1--5). This is simpler and works well with models that struggle with structured output.

Set with `prompt_template: "v2"` in config or `--judge-template v2` on the CLI.

## Parsing Pipeline

Judge output parsing is designed to handle malformed responses:

### v1 Parsing

```
raw output
    │
    ├── 1. Try json.loads(raw) directly
    │       → Success: validate fields, return
    │
    ├── 2. Extract JSON block from markdown fences / surrounding text
    │       → Regex: find {...} pattern
    │       → Try json.loads on extracted block
    │
    ├── 3. Regex fallback
    │       → Extract score, verdict, reason via patterns
    │
    └── 4. Default fallback
            → score=1, verdict="unknown", reason="Failed to parse judge output"
```

### v2 Parsing

```
raw output
    │
    ├── 1. Try int(raw.strip())
    │       → Clamp to 1--5
    │
    ├── 2. Regex: find first digit 1-5 in text
    │       → re.search(r'\b([1-5])\b', raw)
    │
    └── 3. Default fallback
            → score=1, verdict="unknown"
```

## Concurrent Execution

When a run includes both Foundry and Batch judges, they execute concurrently:

```
run_judge()
    │
    ├── asyncio.gather(
    │       _run_request_response_judges(...),
    │       _run_batch_judges(...)
    │   )
    │
    └── All judge outputs appended to judge_outputs.jsonl
```

Internally, request building and parsing live in `stages/judging.py`, while `pipeline/judge_runner.py` only handles orchestration. Foundry judges share the same provider adapter used by inference; the only stage-specific differences are the rendered prompt and the response parser.

Foundry judges use the shared `RequestResponseExecutor` with bounded async concurrency and retry handling. Batch judges use the shared `BatchExecutor`, which delegates the Azure OpenAI batch lifecycle to the batch adapter.

## Dedup

Judge results are deduplicated by the tuple `(instance_id, model_id, judge_name)`. On re-run, only missing combinations are processed. This allows:

- Adding a new judge to an existing run
- Resuming after interruption
- Re-running without duplicating work

## Score Schema

Every judge output follows this schema:

| Field | Type | Description |
|---|---|---|
| `instance_id` | `str` | Dataset instance identifier |
| `model_id` | `str` | Model that produced the inference output |
| `judge_name` | `str` | Display name of this judge |
| `score` | `int` (1--5) | Quality score |
| `verdict` | `str` | `"correct"`, `"incorrect"`, `"partial"`, or `"unknown"` |
| `reason` | `str` | Free-text explanation from the judge |
| `raw_output` | `str` | Unprocessed judge response (for auditability) |
| `judge_prompt` | `str` | The full prompt sent to the judge |
| `tokens_prompt` | `int \| null` | Prompt token count (if reported by API) |
| `tokens_completion` | `int \| null` | Completion token count |
| `timestamp` | `datetime` | When the evaluation was recorded |
| `run_id` | `str` | Run identifier |
