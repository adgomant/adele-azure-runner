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

from adele_runner.config import AppConfig, JudgeConfig, RateLimitsConfig, load_config
from adele_runner.runtime.resolution import (
    resolve_inference_execution_settings,
    resolve_judge_execution_settings,
    resolve_judge_targets,
)
from adele_runner.runtime.types import ExecutionSettings
from adele_runner.schemas import InferenceOutput

_VALID_MODES = {"request_response", "batch", "auto"}
_VALID_PROVIDERS = {"azure_openai", "azure_ai_inference", "google_genai", "anthropic"}
_LEGACY_PROVIDER_ALIASES = {"foundry": "azure_ai_inference", "google": "google_genai"}
_LEGACY_MODE_ALIASES = {
    "foundry": ("azure_ai_inference", "request_response"),
    "google": ("google_genai", "request_response"),
    "batch": ("azure_openai", "batch"),
}

app = typer.Typer(
    name="adele-runner",
    add_completion=False,
)
console = Console()

# Module-level state set by @app.callback (global options).
_cli_run_id: str | None = None


@app.callback()
def _main(
    run_id: str | None = typer.Option(
        None, "--run-id", "-r", help="Run ID. Overrides config run.run_id."
    ),
) -> None:
    """ADeLe benchmark inference + multi-judge evaluation on Azure AI Foundry."""
    global _cli_run_id  # noqa: PLW0603
    _cli_run_id = run_id


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


def _normalize_provider(value: str) -> str:
    if value in _LEGACY_PROVIDER_ALIASES:
        logging.getLogger(__name__).warning(
            "Deprecated provider alias '%s'; use '%s'.",
            value,
            _LEGACY_PROVIDER_ALIASES[value],
        )
        return _LEGACY_PROVIDER_ALIASES[value]
    if value not in _VALID_PROVIDERS:
        raise typer.BadParameter(
            f"Invalid provider '{value}'. Use one of: {', '.join(sorted(_VALID_PROVIDERS))}."
        )
    return value


def _normalize_provider_mode(
    provider: str | None,
    mode: str | None,
) -> tuple[str | None, str | None]:
    resolved_provider = _normalize_provider(provider) if provider is not None else None

    if mode is None:
        return resolved_provider, None

    if mode in _LEGACY_MODE_ALIASES:
        mapped_provider, mapped_mode = _LEGACY_MODE_ALIASES[mode]
        logging.getLogger(__name__).warning(
            "Deprecated mode alias '%s'; use provider='%s' and mode='%s'.",
            mode,
            mapped_provider,
            mapped_mode,
        )
        if resolved_provider is not None and resolved_provider != mapped_provider:
            raise typer.BadParameter(
                f"Mode '{mode}' implies provider '{mapped_provider}', but --provider was '{resolved_provider}'."
            )
        return mapped_provider, mapped_mode

    if mode not in _VALID_MODES:
        raise typer.BadParameter(
            f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(_VALID_MODES))}"
        )
    return resolved_provider, mode


def _parse_judge_flag(value: str) -> JudgeConfig:
    """Parse a ``--judge`` value into a :class:`JudgeConfig`.

    Accepted formats::

        MODEL                                   → legacy default
        MODEL:PROVIDER                          → provider + request_response mode
        MODEL:PROVIDER:MODE                     → explicit provider + mode
        MODEL:PROVIDER:MODE:TPM:RPM             → explicit provider + mode + rate limits
        MODEL:PROVIDER:MODE:TPM:RPM:MAXTOK      → explicit provider + mode + rate limits + max_tokens

    Legacy formats are still accepted for one release cycle.
    """
    parts = value.split(":")
    max_tokens = 512  # default

    if len(parts) == 1:
        model = parts[0]
        provider, mode = "azure_ai_inference", "request_response"
        rate_limits = None
    elif len(parts) == 2:
        model, provider_token = parts
        if provider_token == "batch":
            provider, mode = "azure_openai", "batch"
        else:
            provider = _normalize_provider(provider_token)
            mode = "request_response"
        rate_limits = None
    elif len(parts) == 3:
        model, provider_token, mode_token = parts
        provider = _normalize_provider(provider_token)
        provider, mode = _normalize_provider_mode(provider, mode_token)
        rate_limits = None
    elif len(parts) in (4, 5, 6):
        # Legacy: MODEL:PROVIDER[:MODE]:TPM:RPM[:MAXTOK]
        if parts[2].isdigit():
            model, provider_token, tpm_str, rpm_str = parts[:4]
            if provider_token == "batch":
                provider, mode = "azure_openai", "batch"
            else:
                provider = _normalize_provider(provider_token)
                mode = "request_response"
            max_idx = 4
        else:
            model, provider_token, mode_token, tpm_str, rpm_str = parts[:5]
            provider = _normalize_provider(provider_token)
            provider, mode = _normalize_provider_mode(provider, mode_token)
            max_idx = 5
        try:
            tpm = int(tpm_str)
            rpm = int(rpm_str)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid rate limits in '{value}'. TPM and RPM must be integers."
            ) from None
        rate_limits = RateLimitsConfig(tokens_per_minute=tpm, requests_per_minute=rpm)
        if len(parts) == max_idx + 1:
            try:
                max_tokens = int(parts[max_idx])
            except ValueError:
                raise typer.BadParameter(
                    f"Invalid max_tokens in '{value}'. MAX_TOKENS must be an integer."
                ) from None
    else:
        raise typer.BadParameter(
            f"Invalid judge format '{value}'. "
            "Use MODEL:PROVIDER:MODE[:TPM:RPM[:MAX_TOKENS]] "
            "(legacy forms are still accepted)."
        )

    if rate_limits is not None and mode == "batch":
        raise typer.BadParameter(
            f"Rate limits are not supported for batch mode judges ('{value}'). "
            "Batch judges use file splitting, not rate limiting."
        )

    return JudgeConfig(
        name=model,
        provider=provider,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        model=model,
        rate_limits=rate_limits,
        max_tokens=max_tokens,
    )  # type: ignore[arg-type]


def apply_cli_overrides(
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    provider: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    judges: list[str] | None = None,
    judge_template: str | None = None,
    tpm: int | None = None,
    rpm: int | None = None,
) -> None:
    """Mutate *cfg* in-place with CLI flag overrides.

    CLI values always take precedence over config-file values.
    """
    if run_id is not None:
        cfg.run.run_id = run_id

    if provider is not None or mode is not None:
        resolved_provider, resolved_mode = _normalize_provider_mode(provider, mode)
        if resolved_provider is not None:
            cfg.inference.provider = resolved_provider  # type: ignore[assignment]
        if resolved_mode is not None:
            cfg.inference.mode = resolved_mode  # type: ignore[assignment]

    if model is not None:
        cfg.inference.model = model

    if judges:
        cfg.judging.judges = [_parse_judge_flag(j) for j in judges]
        cfg.judging.enabled = True

    if judge_template is not None:
        cfg.judging.prompt_template = judge_template

    # --tpm and --rpm must both be provided or both omitted
    if tpm is not None or rpm is not None:
        if tpm is None or rpm is None:
            raise typer.BadParameter("--tpm and --rpm must both be provided together.")
        cfg.inference.rate_limits = RateLimitsConfig(tokens_per_minute=tpm, requests_per_minute=rpm)


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
                    "Loaded %d ground truths from cache: %s",
                    len(cached),
                    cache_path,
                )
                return cached
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to read ground truth cache %s, re-downloading.",
                cache_path,
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


def _print_dry_run(
    cfg: AppConfig,
    *,
    items_count: int | None = None,
    execution_settings: ExecutionSettings | None = None,
) -> None:
    """Print a dry-run summary and exit."""
    from adele_runner.utils.io import build_dedup_index

    mode = cfg.resolve_inference_mode()
    model = cfg.inference.model

    console.print("\n[bold cyan]--- Dry-Run Summary ---[/bold cyan]")
    console.print(f"  Run ID:         {cfg.run.run_id}")
    console.print(
        f"  Dataset:        {cfg.dataset.hf_id} (split={cfg.dataset.split}, limit={cfg.dataset.limit})"
    )
    if items_count is not None:
        console.print(f"  Items loaded:   {items_count}")
    console.print(f"  Provider:       {cfg.inference.provider}")
    console.print(f"  Mode:           {cfg.inference.mode} (effective transport={mode})")
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
        judge_names = [f"{j.name}[{j.provider}/{j.mode}]" for j in cfg.judging.judges]
        console.print(f"  Judging:        enabled (template={cfg.judging.prompt_template})")
        console.print(
            f"  Judges:         {', '.join(judge_names) if judge_names else '(none configured)'}"
        )
    else:
        console.print("  Judging:        disabled")

    if cfg.pricing.enabled:
        console.print(f"  Pricing:        enabled ({len(cfg.pricing.models)} models configured)")

    # Concurrency info (especially useful when rate-limit auto-tuning is active)
    rl = cfg.inference.rate_limits
    if rl is not None:
        console.print(
            f"  Rate limits:    TPM={rl.tokens_per_minute:,}  RPM={rl.requests_per_minute:,}"
        )
    judge_targets = resolve_judge_targets(cfg) if cfg.judging.enabled else []
    judge_request_limits = [target.rate_limits for target in judge_targets if target.prompt_mode == "request_response" and target.rate_limits is not None]
    if judge_request_limits:
        judge_rl = min(judge_request_limits, key=lambda rl: rl.tokens_per_minute)
        console.print(
            f"  Judge limits:   TPM={judge_rl.tokens_per_minute:,}  RPM={judge_rl.requests_per_minute:,}"
        )
    cc = execution_settings or cfg.concurrency
    console.print(
        f"  Concurrency:    max_in_flight={cc.max_in_flight}  "
        f"timeout={cc.request_timeout_s:.0f}s  "
        f"backoff={cc.backoff_base_s:.1f}–{cc.backoff_max_s:.1f}s"
    )

    console.print("[bold cyan]--- (no API calls made) ---[/bold cyan]\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run_inference(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider SDK: azure_openai, azure_ai_inference, google_genai, or anthropic.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name or deployment."),
    mode: str | None = typer.Option(
        None, "--mode", help="Execution mode: request_response, batch, or auto."
    ),
    tpm: int | None = typer.Option(None, "--tpm", help="Tokens per minute rate limit."),
    rpm: int | None = typer.Option(None, "--rpm", help="Requests per minute rate limit."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan and exit without API calls."),
) -> None:
    """Run inference over the ADeLe dataset."""
    from adele_runner.datasets.adele import load_adele
    from adele_runner.pipeline.inference_runner import run_inference as _run

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    apply_cli_overrides(
        cfg,
        run_id=_cli_run_id,
        provider=provider,
        mode=mode,
        model=model,
        tpm=tpm,
        rpm=rpm,
    )

    items = load_adele(
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
    )

    if dry_run:
        _print_dry_run(
            cfg,
            items_count=len(items),
            execution_settings=resolve_inference_execution_settings(cfg),
        )
        return

    _validate_or_exit(cfg)
    asyncio.run(_run(cfg, items))
    console.print("[bold green]Inference complete.[/bold green]")


@app.command()
def run_judge(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    judge: list[str] | None = typer.Option(
        None,
        "--judge",
        "-j",
        help="Judge model. Format: MODEL:PROVIDER:MODE[:TPM:RPM[:MAX_TOKENS]]. Legacy forms are still accepted.",
    ),
    judge_template: str | None = typer.Option(
        None, "--judge-template", help="Judge prompt template: v1 or v2."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan and exit without API calls."),
) -> None:
    """Run judge evaluation over existing inference outputs."""
    from adele_runner.pipeline.judge_runner import run_judge as _run
    from adele_runner.utils.io import read_jsonl

    cfg = _get_config(config)
    _setup_logging(cfg.logging.level)
    apply_cli_overrides(cfg, run_id=_cli_run_id, judges=judge, judge_template=judge_template)

    outputs_path = cfg.outputs_path()
    if not outputs_path.exists():
        typer.echo(
            f"No inference outputs found at {outputs_path}. Run run-inference first.", err=True
        )
        raise typer.Exit(1)

    inference_outputs = read_jsonl(outputs_path, InferenceOutput)

    if dry_run:
        request_targets = [
            target for target in resolve_judge_targets(cfg) if target.prompt_mode == "request_response"
        ]
        _print_dry_run(
            cfg,
            items_count=len(inference_outputs),
            execution_settings=(
                None
                if not request_targets
                else resolve_judge_execution_settings(
                    cfg,
                    max(request_targets, key=lambda target: target.max_tokens),
                )
            ),
        )
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
    apply_cli_overrides(cfg, run_id=_cli_run_id)

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
    apply_cli_overrides(cfg, run_id=_cli_run_id)
    _summarize(cfg)


@app.command()
def run_all(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider SDK: azure_openai, azure_ai_inference, google_genai, or anthropic.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name or deployment."),
    mode: str | None = typer.Option(
        None, "--mode", help="Execution mode: request_response, batch, or auto."
    ),
    tpm: int | None = typer.Option(None, "--tpm", help="Tokens per minute rate limit."),
    rpm: int | None = typer.Option(None, "--rpm", help="Requests per minute rate limit."),
    judge: list[str] | None = typer.Option(
        None,
        "--judge",
        "-j",
        help="Judge model. Format: MODEL:PROVIDER:MODE[:TPM:RPM[:MAX_TOKENS]]. Legacy forms are still accepted.",
    ),
    judge_template: str | None = typer.Option(
        None, "--judge-template", help="Judge prompt template: v1 or v2."
    ),
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
    apply_cli_overrides(
        cfg,
        run_id=_cli_run_id,
        provider=provider,
        mode=mode,
        model=model,
        judges=judge,
        judge_template=judge_template,
        tpm=tpm,
        rpm=rpm,
    )

    items = load_adele(
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
    )

    if dry_run:
        _print_dry_run(
            cfg,
            items_count=len(items),
            execution_settings=resolve_inference_execution_settings(cfg),
        )
        return

    _validate_or_exit(cfg)

    # 1. Inference
    asyncio.run(_run_inference(cfg, items))
    console.print("[bold green]Inference complete.[/bold green]")

    # 2. Judging (re-tune concurrency for judge rate limits)
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
