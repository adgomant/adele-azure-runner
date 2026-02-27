"""Typer CLI for ADeLe Runner."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from adele_runner.config import AppConfig, JudgeConfig, load_config
from adele_runner.schemas import InferenceOutput

# Maps user-friendly CLI mode names to internal config values.
_MODE_MAP: dict[str, str] = {
    "foundry": "foundry_async",
    "batch": "azure_openai_batch",
    "auto": "auto",
}

app = typer.Typer(
    name="adele-runner",
    help="ADeLe benchmark inference + multi-judge evaluation on Azure AI Foundry.",
    add_completion=False,
)
console = Console()

# Load .env files. Order matters: .env.local first so its values take
# precedence over .env. override=False means shell env vars always win.
load_dotenv(dotenv_path=".env.local", override=False)
load_dotenv(dotenv_path=".env", override=False)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _get_config(config_path: Path | None):
    if config_path is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            typer.echo("No --config provided and config.yaml not found.", err=True)
            raise typer.Exit(1)
    return load_config(config_path)


_VALID_JUDGE_PROVIDERS = {"foundry", "batch"}


def _parse_judge_flag(value: str) -> JudgeConfig:
    """Parse a ``--judge`` value into a :class:`JudgeConfig`.

    Accepted formats: ``MODEL`` (foundry) or ``MODEL:batch``.
    """
    if ":" in value:
        model, provider = value.rsplit(":", 1)
        if provider not in _VALID_JUDGE_PROVIDERS:
            raise typer.BadParameter(
                f"Invalid judge provider '{provider}' in '{value}'. "
                f"Use MODEL or MODEL:batch."
            )
    else:
        model, provider = value, "foundry"
    return JudgeConfig(name=model, provider=provider, model=model)  # type: ignore[arg-type]


def apply_cli_overrides(
    cfg: AppConfig,
    *,
    mode: str | None = None,
    models: list[str] | None = None,
    judges: list[str] | None = None,
    judge_template: str | None = None,
) -> None:
    """Mutate *cfg* in-place with CLI flag overrides.

    *models* sets the first model only (multi-model loop is handled by the caller).
    """
    if mode is not None:
        internal = _MODE_MAP.get(mode)
        if internal is None:
            raise typer.BadParameter(
                f"Invalid mode '{mode}'. Choose from: {', '.join(_MODE_MAP)}"
            )
        cfg.inference.mode = internal  # type: ignore[assignment]
        if mode == "batch":
            cfg.inference.batch.enabled = True

    if models:
        model = models[0]
        resolved = cfg.resolve_inference_mode()
        if resolved == "azure_openai_batch":
            cfg.inference.batch.deployment = model
        else:
            cfg.inference.foundry.model = model

    if judges:
        cfg.judging.judges = [_parse_judge_flag(j) for j in judges]
        cfg.judging.enabled = True

    if judge_template is not None:
        cfg.judging.prompt_template = judge_template


def _validate_or_exit(cfg: AppConfig, *, dry_run: bool = False) -> None:
    """Run config validation and exit on errors (skip API-key checks in dry-run)."""
    errors = cfg.validate_config(dry_run=dry_run)
    if errors:
        for err in errors:
            typer.echo(f"Config error: {err}", err=True)
        raise typer.Exit(1)


def _load_ground_truths(cfg: AppConfig) -> dict[str, str]:
    """Load ground truths, preferring a cached file in the run dir.

    Falls back to downloading from HF and caching the result.
    """
    cache_path = cfg.ground_truths_cache_path()

    # Try cache first
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if isinstance(cached, dict) and cached:
                logging.getLogger(__name__).info(
                    "Loaded %d ground truths from cache: %s", len(cached), cache_path,
                )
                return cached
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to read ground truth cache %s, re-downloading.", cache_path,
            )

    # Download from HF
    from adele_runner.datasets.adele import load_adele

    items = load_adele(
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
    )
    gt_map = {item.instance_id: item.ground_truth for item in items}

    # Cache for future runs
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(gt_map, fh, ensure_ascii=False)
        logging.getLogger(__name__).info("Cached %d ground truths to %s", len(gt_map), cache_path)
    except Exception:
        logging.getLogger(__name__).warning("Could not write ground truth cache to %s", cache_path)

    return gt_map


def _set_model_for_run(cfg: AppConfig, model: str) -> None:
    """Point config at *model* for the next inference run."""
    resolved = cfg.resolve_inference_mode()
    if resolved == "azure_openai_batch":
        cfg.inference.batch.deployment = model
    else:
        cfg.inference.foundry.model = model


def _print_dry_run(cfg: AppConfig, *, items_count: int | None = None) -> None:
    """Print a dry-run summary and exit."""
    from adele_runner.utils.io import build_dedup_index

    mode = cfg.resolve_inference_mode()
    model = (
        cfg.inference.batch.deployment
        if mode == "azure_openai_batch"
        else cfg.inference.foundry.model
    )

    console.print("\n[bold cyan]--- Dry-Run Summary ---[/bold cyan]")
    console.print(f"  Run ID:         {cfg.run.run_id}")
    console.print(f"  Dataset:        {cfg.dataset.hf_id} (split={cfg.dataset.split}, limit={cfg.dataset.limit})")
    if items_count is not None:
        console.print(f"  Items loaded:   {items_count}")
    console.print(f"  Mode:           {mode}")
    console.print(f"  Model:          {model}")

    # Dedup info
    outputs_path = cfg.outputs_path()
    if outputs_path.exists():
        done = build_dedup_index(outputs_path, "instance_id", "model_id")
        console.print(f"  Already done:   {len(done)} (instance, model) pairs in {outputs_path}")
    else:
        console.print("  Already done:   0 (no outputs file yet)")

    # Judging info
    if cfg.judging.enabled:
        judge_names = [j.name for j in cfg.judging.judges]
        console.print(f"  Judging:        enabled (template={cfg.judging.prompt_template})")
        console.print(f"  Judges:         {', '.join(judge_names) if judge_names else '(none configured)'}")
    else:
        console.print("  Judging:        disabled")

    if cfg.pricing.enabled:
        console.print(f"  Pricing:        enabled ({len(cfg.pricing.models)} models configured)")

    console.print("[bold cyan]--- (no API calls made) ---[/bold cyan]\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run_inference(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    model: list[str] | None = typer.Option(None, "--model", "-m", help="Model name(s). Repeatable."),
    mode: str | None = typer.Option(None, "--mode", help="Inference mode: foundry, batch, or auto."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan and exit without API calls."),
) -> None:
    """Run inference over the ADeLe dataset."""
    from adele_runner.datasets.adele import load_adele
    from adele_runner.pipeline.inference_runner import run_inference as _run

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    apply_cli_overrides(cfg, mode=mode, models=model)

    items = load_adele(
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
    )

    if dry_run:
        _print_dry_run(cfg, items_count=len(items))
        return

    _validate_or_exit(cfg)

    # Multi-model support: loop over each model
    models_to_run = model if model and len(model) > 1 else None

    if models_to_run:
        for m in models_to_run:
            _set_model_for_run(cfg, m)
            console.print(f"\n[bold]Running inference for model: {m}[/bold]")
            asyncio.run(_run(cfg, items))
    else:
        asyncio.run(_run(cfg, items))

    console.print("[bold green]Inference complete.[/bold green]")


@app.command()
def run_judge(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    judge: list[str] | None = typer.Option(None, "--judge", "-j", help="Judge model (MODEL or MODEL:batch). Repeatable."),
    judge_template: str | None = typer.Option(None, "--judge-template", help="Judge prompt template: v1 or v2."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan and exit without API calls."),
) -> None:
    """Run judge evaluation over existing inference outputs."""
    from adele_runner.pipeline.judge_runner import run_judge as _run
    from adele_runner.utils.io import read_jsonl

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    apply_cli_overrides(cfg, judges=judge, judge_template=judge_template)

    outputs_path = cfg.outputs_path()
    if not outputs_path.exists():
        typer.echo(f"No inference outputs found at {outputs_path}. Run run-inference first.", err=True)
        raise typer.Exit(1)

    inference_outputs = read_jsonl(outputs_path, InferenceOutput)

    if dry_run:
        _print_dry_run(cfg, items_count=len(inference_outputs))
        return

    _validate_or_exit(cfg)

    # Load ground truths (with caching)
    ground_truths = _load_ground_truths(cfg)

    asyncio.run(_run(cfg, inference_outputs, ground_truths))
    console.print("[bold green]Judging complete.[/bold green]")


@app.command()
def merge_results(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Merge inference + judge outputs into merged_results.parquet."""
    from adele_runner.pipeline.merge import merge_results as _merge

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)

    path = _merge(cfg)
    console.print(f"[bold green]Merged results written to {path}[/bold green]")


@app.command()
def summarize(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Print metrics summary and judge agreement statistics."""
    from adele_runner.pipeline.metrics import summarize as _summarize

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    _summarize(cfg)


@app.command()
def run_all(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    model: list[str] | None = typer.Option(None, "--model", "-m", help="Model name(s). Repeatable."),
    mode: str | None = typer.Option(None, "--mode", help="Inference mode: foundry, batch, or auto."),
    judge: list[str] | None = typer.Option(None, "--judge", "-j", help="Judge model (MODEL or MODEL:batch). Repeatable."),
    judge_template: str | None = typer.Option(None, "--judge-template", help="Judge prompt template: v1 or v2."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan and exit without API calls."),
) -> None:
    """Run the full pipeline: inference → judge → merge → summarize."""
    from adele_runner.datasets.adele import load_adele
    from adele_runner.pipeline.inference_runner import run_inference as _run_inference
    from adele_runner.pipeline.judge_runner import run_judge as _run_judge
    from adele_runner.pipeline.merge import merge_results as _merge
    from adele_runner.pipeline.metrics import summarize as _summarize
    from adele_runner.utils.io import read_jsonl

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    apply_cli_overrides(cfg, mode=mode, models=model, judges=judge, judge_template=judge_template)

    items = load_adele(
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
    )

    if dry_run:
        _print_dry_run(cfg, items_count=len(items))
        return

    _validate_or_exit(cfg)

    # 1. Inference (loop over models if multiple)
    models_to_run = model if model and len(model) > 1 else None
    if models_to_run:
        for m in models_to_run:
            _set_model_for_run(cfg, m)
            console.print(f"\n[bold]Running inference for model: {m}[/bold]")
            asyncio.run(_run_inference(cfg, items))
    else:
        asyncio.run(_run_inference(cfg, items))
    console.print("[bold green]Inference complete.[/bold green]")

    # 2. Judging
    if cfg.judging.enabled:
        inference_outputs = read_jsonl(cfg.outputs_path(), InferenceOutput)
        ground_truths = _load_ground_truths(cfg)
        asyncio.run(_run_judge(cfg, inference_outputs, ground_truths))
        console.print("[bold green]Judging complete.[/bold green]")

    # 3. Merge
    path = _merge(cfg)
    console.print(f"[bold green]Merged results written to {path}[/bold green]")

    # 4. Summarize
    _summarize(cfg)
    console.print("[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    app()
