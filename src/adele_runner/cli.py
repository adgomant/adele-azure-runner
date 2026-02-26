from __future__ import annotations

import asyncio
import json
import logging

import typer

from adele_runner.config import load_config
from adele_runner.pipeline.inference_runner import run_inference
from adele_runner.pipeline.judge_runner import run_judging
from adele_runner.pipeline.manifest import write_manifest
from adele_runner.pipeline.merge import merge_results
from adele_runner.pipeline.metrics import summarize_results

app = typer.Typer(help="ADeLe benchmark runner")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command("prepare-requests")
def prepare_requests(config: str = typer.Option("config.example.yaml", "--config")) -> None:
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    manifest_path = write_manifest(cfg)
    typer.echo(f"prepared manifest at {manifest_path}")


@app.command("run-inference")
def run_inference_cmd(config: str = typer.Option("config.example.yaml", "--config")) -> None:
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    write_manifest(cfg)
    out = asyncio.run(run_inference(cfg))
    typer.echo(f"wrote inference outputs to {out}")


@app.command("run-judge")
def run_judge_cmd(config: str = typer.Option("config.example.yaml", "--config")) -> None:
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    out = run_judging(cfg)
    typer.echo(f"wrote judge outputs to {out}")


@app.command("merge-results")
def merge_results_cmd(config: str = typer.Option("config.example.yaml", "--config")) -> None:
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    out = merge_results(cfg)
    typer.echo(f"wrote merged parquet to {out}")


@app.command("summarize")
def summarize_cmd(config: str = typer.Option("config.example.yaml", "--config")) -> None:
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    summary = summarize_results(cfg)
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
