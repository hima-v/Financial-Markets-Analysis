from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..core.config import settings
from ..core.errors import AppError


DATASET_ID_RE = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str
    created_at: str
    row_count: int
    symbol_count: int
    columns: list[str]


def datasets_root() -> Path:
    root = Path(settings.data_dir).resolve() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(dataset_id: str) -> Path:
    _validate_dataset_id(dataset_id)
    return datasets_root() / dataset_id


def list_dataset_ids() -> list[str]:
    root = datasets_root()
    ids: list[str] = []
    for p in root.iterdir():
        if p.is_dir() and DATASET_ID_RE.fullmatch(p.name):
            ids.append(p.name)
    return sorted(ids)


def enforce_dataset_quota() -> None:
    existing = list_dataset_ids()
    if len(existing) >= settings.max_saved_datasets:
        raise AppError(
            code="dataset_quota_exceeded",
            message="Dataset storage quota exceeded.",
            details={"max_saved_datasets": settings.max_saved_datasets},
        )


def save_dataset(df: pd.DataFrame) -> DatasetMeta:
    enforce_dataset_quota()

    dataset_id = uuid.uuid4().hex
    ddir = dataset_dir(dataset_id)
    ddir.mkdir(parents=True, exist_ok=False)

    parquet_path = ddir / "data.parquet"
    meta_path = ddir / "meta.json"

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        raise AppError(code="storage_write_failed", message="Failed to store dataset.") from e

    meta = DatasetMeta(
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        row_count=int(len(df)),
        symbol_count=int(df["symbol"].nunique(dropna=True)) if "symbol" in df.columns else 0,
        columns=[str(c) for c in df.columns],
    )
    _atomic_write_json(meta_path, meta.__dict__)
    return meta


def load_dataset(dataset_id: str) -> pd.DataFrame:
    ddir = dataset_dir(dataset_id)
    parquet_path = ddir / "data.parquet"
    if not parquet_path.exists():
        raise AppError(code="dataset_not_found", message="Dataset not found.", status_code=404)
    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        raise AppError(code="storage_read_failed", message="Failed to read stored dataset.", status_code=500) from e


def load_meta(dataset_id: str) -> DatasetMeta:
    ddir = dataset_dir(dataset_id)
    meta_path = ddir / "meta.json"
    if not meta_path.exists():
        raise AppError(code="dataset_not_found", message="Dataset not found.", status_code=404)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetMeta(
        dataset_id=str(data["dataset_id"]),
        created_at=str(data["created_at"]),
        row_count=int(data["row_count"]),
        symbol_count=int(data["symbol_count"]),
        columns=list(data["columns"]),
    )


def _validate_dataset_id(dataset_id: str) -> None:
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise AppError(code="invalid_dataset_id", message="Invalid dataset_id.", status_code=400)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

