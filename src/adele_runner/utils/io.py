"""JSONL / Parquet IO with append, streaming read, and dedup support."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append a single Pydantic model as a JSONL line. Not atomic; partial writes possible on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()


def iter_jsonl(path: Path, model: type[M]) -> Iterator[M]:
    """Stream-read a JSONL file, yielding parsed Pydantic models."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    """Read all records from a JSONL file into a list."""
    return list(iter_jsonl(path, model))


# ---------------------------------------------------------------------------
# Dedup index
# ---------------------------------------------------------------------------


def build_dedup_index(path: Path, *key_fields: str) -> set[tuple]:
    """Return a set of composite key tuples already present in a JSONL file.

    Example
    -------
    >>> done = build_dedup_index(outputs_path, "instance_id", "model_id")
    >>> if (item.instance_id, cfg.model_id) in done:
    ...     continue
    """
    seen: set[tuple] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = tuple(str(obj.get(f, "")) for f in key_fields)
                seen.add(key)
            except json.JSONDecodeError:
                continue
    logger.debug("Dedup index for %s has %d entries.", path, len(seen))
    return seen


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def write_parquet(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a Parquet file."""
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pandas is required for Parquet support.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(records), path)


def read_parquet(path: Path) -> pd.DataFrame:  # type: ignore[name-defined]  # noqa: F821
    """Read a Parquet file into a DataFrame."""
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pandas is required for Parquet support.") from exc

    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------


def ensure_run_dir(run_dir: Path) -> None:
    """Create the run directory (and parents) if it doesn't exist."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Run directory ensured: %s", run_dir)
