# Judging

The judging stage evaluates inference outputs with one or more judge models. Each judge is configured independently with:

- `provider`
- `mode`
- `model`

## Example Config

```yaml
judging:
  enabled: true
  prompt_template: v1
  judges:
    - name: gpt4o-judge
      provider: azure_ai_inference
      mode: request_response
      model: gpt-4o
      rate_limits:
        tokens_per_minute: 80000
        requests_per_minute: 300
    - name: gemini-batch-judge
      provider: google_genai
      mode: batch
      model: gemini-2.5-flash
    - name: claude-batch-judge
      provider: anthropic
      mode: batch
      model: claude-sonnet-4-5
```

Equivalent CLI:

```bash
uv run adele-runner run-judge \
  --judge gpt-4o:azure_ai_inference:request_response:80000:300:512 \
  --judge gemini-2.5-flash:google_genai:batch \
  --judge claude-sonnet-4-5:anthropic:batch
```

## Prompt Templates

### `v1`

Asks the judge for structured output with:

- `score`
- `verdict`
- `reason`

### `v2`

Asks the judge for a single integer from `1` to `5`.

## Parsing

Parsing lives in [stages/judging.py](/Users/alvar/proyects/adele-azure-runner/src/adele_runner/stages/judging.py).

`v1` parsing tries:

1. direct JSON
2. extracting a JSON block from surrounding text
3. regex fallback
4. default fallback

`v2` parsing tries:

1. direct integer parsing
2. the first `1-5` found in text
3. default fallback

## Execution Model

For each judge:

1. stage logic builds a judge `ChatRequest`
2. resolution picks provider, effective mode, and execution settings
3. a concrete adapter is created from the provider registry
4. the matching executor runs it
5. the stage maps the `ChatResponse` into `JudgeOutput`

The judge pipeline runs request-response and batch lanes concurrently when both are present.

## Rate Limits

Judge `rate_limits` auto-tune request-response pacing and can also define batch queue budgets such as `batch_queue_requests` and `batch_enqueued_tokens`.

## Dedup

Judge rows are deduplicated on:

`(instance_id, model_id, judge_name)`

This makes judge runs resumable and safe to rerun.
