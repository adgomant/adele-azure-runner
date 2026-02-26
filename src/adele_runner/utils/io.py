from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import orjson
import pyarrow as pa
import pyarrow.parquet as pq


def ensure_run_dir(output_dir: str, run_id: str) -> Path:
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(record))
        handle.write(b"\n")


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def load_dedup_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for item in stream_jsonl(path):
        keys.add(tuple(str(item.get(field, "")) for field in fields))
    return keys


def write_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        table = pa.table({})
    else:
        table = pa.Table.from_pylist(rows_list)
    pq.write_table(table, path)
