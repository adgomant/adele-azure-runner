from __future__ import annotations

from pathlib import Path

from adele_runner.config import AppConfig
from adele_runner.models import RunManifest
from adele_runner.utils.io import ensure_run_dir


def write_manifest(config: AppConfig) -> Path:
    run_dir = ensure_run_dir(config.run.output_dir, config.run.run_id)
    path = run_dir / "run_manifest.json"
    manifest = RunManifest(
        run_id=config.run.run_id,
        dataset_name=config.dataset.name,
        dataset_split=config.dataset.split,
        config_snapshot=config.model_dump(mode="json"),
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
