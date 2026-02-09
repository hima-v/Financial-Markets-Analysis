from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.errors import AppError
from ..domain.dataset_schema import REQUIRED_COLUMNS, normalize_column_name


@dataclass(frozen=True)
class IngestLimits:
    max_rows: int = 250_000
    max_columns: int = 200


def load_csv_bytes(data: bytes, *, limits: IngestLimits) -> pd.DataFrame:
    if not data:
        raise AppError(code="empty_upload", message="Uploaded file is empty.")

    try:
        df = pd.read_csv(
            pd.io.common.BytesIO(data),
            engine="c",
            nrows=limits.max_rows,
        )
    except Exception as e:
        raise AppError(code="invalid_csv", message="Could not parse file as CSV.", details={"reason": type(e).__name__}) from e

    if df.shape[1] > limits.max_columns:
        raise AppError(code="too_many_columns", message="Too many columns in uploaded dataset.")

    df = normalize_columns(df)
    df = coerce_types(df)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise AppError(
            code="missing_required_columns",
            message="Missing required columns.",
            details={"missing": missing_required, "required": list(REQUIRED_COLUMNS)},
        )

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={c: normalize_column_name(str(c)) for c in df.columns})
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

