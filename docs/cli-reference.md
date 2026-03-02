# CLI Reference

The ADeLe Runner CLI is built with [Typer](https://typer.tiangolo.com/). All commands are invoked through the `adele-runner` entry point.

```bash
uv run adele-runner <command> [options]
```

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
| `--model` | `-m` | `TEXT` | Model name or deployment. Repeatable for multi-model runs |
| `--mode` | | `TEXT` | Inference mode: `foundry`, `batch`, or `auto` |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Single model, Foundry mode
uv run adele-runner run-inference --model gpt-4o

# Batch mode
uv run adele-runner run-inference --model gpt-4o --mode batch

# Multiple models in one run
uv run adele-runner run-inference --model gpt-4o --model llama-3-70b

# Dry-run preview
uv run adele-runner run-inference --model gpt-4o --dry-run
```

When multiple `--model` flags are provided, inference runs sequentially for each model. The dedup index prevents re-processing instances that are already complete for a given model.

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
| `--judge` | `-j` | `TEXT` | Judge model. Repeatable. Format: `MODEL` (Foundry) or `MODEL:batch` |
| `--judge-template` | | `TEXT` | Judge prompt template: `v1` or `v2` |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Two Foundry judges
uv run adele-runner run-judge --judge gpt-4o --judge claude-3-opus

# Mix Foundry and Batch judges
uv run adele-runner run-judge --judge gpt-4o:batch --judge claude-3-opus

# Use v2 (bare integer) prompt template
uv run adele-runner run-judge --judge gpt-4o --judge-template v2

# Use judges from config.yaml
uv run adele-runner run-judge
```

When `--judge` flags are provided, they **replace** the judges list from `config.yaml`.

### Judge Flag Format

- `MODEL` -- uses Foundry async (e.g. `gpt-4o`, `claude-3-opus`)
- `MODEL:batch` -- uses Azure OpenAI Batch API (e.g. `gpt-4o:batch`)
- `MODEL:foundry` -- explicitly Foundry (same as bare `MODEL`)

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
| `--model` | `-m` | `TEXT` | Model name(s). Repeatable |
| `--mode` | | `TEXT` | Inference mode: `foundry`, `batch`, or `auto` |
| `--judge` | `-j` | `TEXT` | Judge model. Repeatable. Format: `MODEL` or `MODEL:batch` |
| `--judge-template` | | `TEXT` | Judge prompt template: `v1` or `v2` |
| `--dry-run` | | `bool` | Preview plan without API calls |

**Examples:**

```bash
# Full pipeline with one model and two judges
uv run adele-runner run-all --model gpt-4o --judge gpt-4o --judge claude-3-opus

# Multi-model comparison
uv run adele-runner run-all --model gpt-4o --model llama-3-70b --judge gpt-4o

# Dry-run the full pipeline
uv run adele-runner run-all --model gpt-4o --judge gpt-4o --dry-run
```

The `run-all` command skips judging if `judging.enabled` is `false` in config.
