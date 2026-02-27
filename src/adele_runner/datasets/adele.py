"""ADeLe dataset loader — wraps HuggingFace datasets."""

from __future__ import annotations

import hashlib
import logging

from adele_runner.schemas import DatasetItem

logger = logging.getLogger(__name__)

# Columns we look for in the raw HF dataset (in priority order).
# Matching is case-insensitive.
_PROMPT_COLS = ["prompt", "question", "input", "task"]
_GT_COLS = ["groundtruth", "ground_truth", "answer", "expected", "reference"]
_ID_COLS = ["instance_id", "id", "item_id", "question_id"]


def _pick_col(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column, case-insensitive."""
    col_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        original = col_lower.get(candidate.lower())
        if original is not None:
            return original
    return None


def _make_id(row: dict, idx: int, id_col: str | None) -> str:
    if id_col and row.get(id_col) is not None:
        return str(row[id_col])
    # Stable fallback: hash of the row index
    return hashlib.sha1(str(idx).encode()).hexdigest()[:12]


def load_adele(
    hf_id: str = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0",
    split: str = "train",
    limit: int | None = None,
) -> list[DatasetItem]:
    """Load and normalise ADeLe from HuggingFace.

    Parameters
    ----------
    hf_id:
        HuggingFace dataset identifier.
    split:
        Dataset split to load.
    limit:
        If set, return only the first *limit* rows (deterministic HF order).

    Returns
    -------
    list[DatasetItem]
        Normalised dataset items, ordered deterministically.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: uv sync"
        ) from exc

    logger.info("Loading dataset '%s' split='%s' …", hf_id, split)
    ds = load_dataset(hf_id, split=split, trust_remote_code=False)

    columns: list[str] = ds.column_names  # type: ignore[union-attr]
    logger.debug("Dataset columns: %s", columns)

    id_col = _pick_col(columns, _ID_COLS)
    prompt_col = _pick_col(columns, _PROMPT_COLS)
    gt_col = _pick_col(columns, _GT_COLS)

    if prompt_col is None:
        raise ValueError(
            f"Cannot find a prompt column in {columns}. "
            f"Expected one of {_PROMPT_COLS}."
        )
    if gt_col is None:
        raise ValueError(
            f"Cannot find a ground-truth column in {columns}. "
            f"Expected one of {_GT_COLS}."
        )

    logger.info(
        "Column mapping — id: %s | prompt: %s | ground_truth: %s",
        id_col,
        prompt_col,
        gt_col,
    )

    items: list[DatasetItem] = []
    meta_cols = [c for c in columns if c not in {id_col, prompt_col, gt_col}]

    iterable = ds if limit is None else ds.select(range(min(limit, len(ds))))  # type: ignore[arg-type]

    for idx, row in enumerate(iterable):
        row = dict(row)
        instance_id = _make_id(row, idx, id_col)
        metadata = {c: row.get(c) for c in meta_cols}
        items.append(
            DatasetItem(
                instance_id=instance_id,
                prompt=str(row[prompt_col]),
                ground_truth=str(row[gt_col]),
                metadata=metadata,
            )
        )

    logger.info("Loaded %d items from '%s'.", len(items), hf_id)
    return items
