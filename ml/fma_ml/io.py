from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import DATASET_ID_RE
from .schema import REQUIRED_COLUMNS, normalize_column_name


@dataclass(frozen=True)
class LoadConfig:
    max_rows: int = 500_000
    max_columns: int = 300


def load_frame(*, dataset_id: str | None, path: str | None, cfg: LoadConfig) -> pd.DataFrame:
    if bool(dataset_id) == bool(path):
        raise ValueError("Provide exactly one of dataset_id or path.")

    if dataset_id:
        return _load_by_dataset_id(dataset_id, cfg=cfg)
    assert path is not None
    return _load_by_path(path, cfg=cfg)


def _load_by_dataset_id(dataset_id: str, *, cfg: LoadConfig) -> pd.DataFrame:
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError("Invalid dataset_id.")

    parquet = Path("data") / "datasets" / dataset_id / "data.parquet"
    df = pd.read_parquet(parquet)
    return _postprocess(df, cfg=cfg)


def _load_by_path(path: str, *, cfg: LoadConfig) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, nrows=cfg.max_rows, engine="c")
    return _postprocess(df, cfg=cfg)


def _postprocess(df: pd.DataFrame, *, cfg: LoadConfig) -> pd.DataFrame:
    if df.shape[1] > cfg.max_columns:
        raise ValueError("Too many columns.")

    out = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
    out.columns = [str(c).strip() for c in out.columns]

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)

    for c in ("open", "high", "low", "close", "vwap"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = out.dropna(subset=["date", "symbol", "close"]).copy()
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    return out

