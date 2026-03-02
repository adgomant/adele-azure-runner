# CLI Reference

The ADeLe Runner CLI is built with [Typer](https://typer.tiangolo.com/). All commands are invoked through the `adele-runner` entry point.

```bash
uv run adele-runner [GLOBAL OPTIONS] <command> [OPTIONS]
```

## Global Options

These options go **before** the subcommand:

```bash
uv run adele-runner --run-id my_run run-inference --model gpt-4o
```

| Flag | Short | Type | Description |
|---|---|---|---|
| `--run-id` | `-r` | `TEXT` | Run ID. Overrides `run.run_id` in config |

## Commands

| Command | Description |
|---|---|
| [`run-inference`](#run-inference) | Run model inference on the ADeLe dataset |
| [`run-judge`](#run-judge) | Evaluate inference outputs with LLM judges |
| [`merge-results`](#merge-results) | Merge inference + judge outputs into Parquet |
| [`summarize`](#summarize) | Print metrics summary and judge agreement |
| [`run-all`](#run-all) | Run the full pipeline: inference, judge, merge, summarize |

## Common Options

These options are shared across multiple commands:

| Flag | Short | Description |
|---|---|---|
| `--config` | `-c` | Path to YAML config file. Defaults to `config.yaml` in the working directory |
| `--dry-run` | | Print a summary of what would run, then exit without making API calls |

---

## `run-inference`

Run inference over the ADeLe dataset. Loads the dataset, builds a dedup index from prior outputs, and processes only pending instances.

```bash
uv run adele-runner run-inference [OPTIONS]
```

**Options:**

| Flag | Short | Type | Description |
|---|---|---|---|
| `--config` | `-c` | `PATH` | Path to YAML config |
| `--model` | `-m` | `TEXT` | Model name or deployment |
| `--mode` | | `TEXT` | Inference mode: `foundry`, `batch`, or `auto` |
| `--tpm` | | `INT` | Tokens per minute rate limit (foundry only, requires `--rpm`) |
| `--rpm` | | `INT` | Requests per minute rate limit (foundry only, requires `--tpm`) |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Single model, Foundry mode
uv run adele-runner run-inference --model gpt-4o

# Batch mode
uv run adele-runner run-inference --model gpt-4o --mode batch

# With rate limits for auto-tuned concurrency
uv run adele-runner run-inference --model gpt-4o --tpm 80000 --rpm 300

# Dry-run preview
uv run adele-runner run-inference --model gpt-4o --dry-run
```

---

## `run-judge`

Run judge evaluation over existing inference outputs. Reads `outputs.jsonl` from the run directory, loads ground truths (cached if available), and dispatches to configured judges.

```bash
uv run adele-runner run-judge [OPTIONS]
```

**Options:**

| Flag | Short | Type | Description |
|---|---|---|---|
| `--config` | `-c` | `PATH` | Path to YAML config |
| `--judge` | `-j` | `TEXT` | Judge model. Repeatable. See [Judge flag format](#judge-flag-format) below |
| `--judge-template` | | `TEXT` | Judge prompt template: `v1` or `v2` |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Two Foundry judges
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus

# Mix Foundry and Batch judges
uv run adele-runner run-judge --judge gpt-4o:batch --judge claude-3-opus

# Foundry judge with rate limits
uv run adele-runner run-judge --judge gpt-4o:foundry:80000:300

# Use v2 (bare integer) prompt template
uv run adele-runner run-judge --judge gpt-4o --judge-template v2

# Use judges from config.yaml
uv run adele-runner run-judge
```

When `--judge` flags are provided, they **replace** the judges list from `config.yaml`.

### Judge Flag Format

```
MODEL                              → foundry provider, no rate limits
MODEL:PROVIDER                     → explicit provider (foundry or batch)
MODEL:PROVIDER:TPM:RPM             → explicit provider + rate limits (foundry only)
MODEL:PROVIDER:TPM:RPM:MAX_TOKENS  → provider + rate limits + max tokens (foundry only)
```

Examples:

```bash
--judge gpt-4o                          # foundry, no rate limits
--judge gpt-4o:batch                    # batch provider
--judge gpt-4o:foundry:80000:300        # foundry with TPM=80000 RPM=300
--judge gpt-4o:foundry:80000:300:1024   # foundry with rate limits + max_tokens=1024
```

Rate limits for batch judges are not supported (batch uses file splitting instead). Providing TPM/RPM for a batch judge is an error.

---

## `merge-results`

Merge inference outputs and judge evaluations into a single Parquet file.

```bash
uv run adele-runner merge-results [OPTIONS]
```

**Options:**

| Flag | Short | Type | Description |
|---|---|---|---|
| `--config` | `-c` | `PATH` | Path to YAML config |

Reads `outputs.jsonl` and `judge_outputs.jsonl`, joins them by `(instance_id, model_id)`, and writes `merged_results.parquet`. Computed columns include `avg_judge_score` and `verification_score`.

---

## `summarize`

Print a metrics summary to the console and write `metrics.json` to the run directory.

```bash
uv run adele-runner summarize [OPTIONS]
```

**Options:**

| Flag | Short | Type | Description |
|---|---|---|---|
| `--config` | `-c` | `PATH` | Path to YAML config |

Output includes:

- Per-judge statistics (count, mean score, score distribution)
- Verification metrics (pass rate, mean average score)
- Inter-rater agreement (Cohen's kappa for each judge pair)
- Token usage per model
- Estimated cost per model (when pricing is configured)

---

## `run-all`

Run the full pipeline in sequence: inference, judge, merge, summarize. Accepts all flags from `run-inference` and `run-judge`.

```bash
uv run adele-runner run-all [OPTIONS]
```

**Options:**

| Flag | Short | Type | Description |
|---|---|---|---|
| `--config` | `-c` | `PATH` | Path to YAML config |
| `--model` | `-m` | `TEXT` | Model name or deployment |
| `--mode` | | `TEXT` | Inference mode: `foundry`, `batch`, or `auto` |
| `--tpm` | | `INT` | Tokens per minute rate limit (foundry only, requires `--rpm`) |
| `--rpm` | | `INT` | Requests per minute rate limit (foundry only, requires `--tpm`) |
| `--judge` | `-j` | `TEXT` | Judge model. Repeatable. See [Judge flag format](#judge-flag-format) |
| `--judge-template` | | `TEXT` | Judge prompt template: `v1` or `v2` |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Full pipeline with one model and two judges
uv run adele-runner run-all --model gpt-4o --judge gpt-4o --judge claude-3-opus

# With rate limits
uv run adele-runner run-all --model gpt-4o --tpm 80000 --rpm 300 --judge gpt-4o

# Dry-run the full pipeline
uv run adele-runner run-all --model gpt-4o --judge gpt-4o --dry-run
```

The `run-all` command skips judging if `judging.enabled` is `false` in config.
