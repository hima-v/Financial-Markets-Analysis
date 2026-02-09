from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


REPO_DEFAULT_CSV = Path(__file__).resolve().parents[2] / "nse_sensex (1).csv"


def load_csv_bytes(data: bytes, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data), nrows=nrows)


def load_repo_default(*, nrows: int | None = None) -> pd.DataFrame:
    if not REPO_DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Missing default dataset at {REPO_DEFAULT_CSV}")
    return pd.read_csv(REPO_DEFAULT_CSV, nrows=nrows)


def read_repo_default_bytes() -> bytes:
    if not REPO_DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Missing default dataset at {REPO_DEFAULT_CSV}")
    return REPO_DEFAULT_CSV.read_bytes()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "DATE": "date",
        "Date": "date",
        "date": "date",
        "SYMBOL": "symbol",
        "Symbol": "symbol",
        "symbol": "symbol",
        "OPEN": "open",
        "Open": "open",
        "open": "open",
        "HIGH": "high",
        "High": "high",
        "high": "high",
        "LOW": "low",
        "Low": "low",
        "low": "low",
        "CLOSE": "close",
        "Close": "close",
        "close": "close",
        "VOLUME": "volume",
        "Volume": "volume",
        "volume": "volume",
        "VWAP": "vwap",
        "vwap": "vwap",
    }

    renamed = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    renamed.columns = [str(c).strip() for c in renamed.columns]
    return renamed


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)

    for c in ("open", "high", "low", "close", "vwap"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip()

    return out

