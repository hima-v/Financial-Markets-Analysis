from __future__ import annotations

import pandas as pd

from .types import ValidationReport


REQUIRED_COLUMNS = ("date", "symbol", "close")
RECOMMENDED_COLUMNS = ("open", "high", "low", "volume")


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")
        return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))

    missing_recommended = [c for c in RECOMMENDED_COLUMNS if c not in df.columns]
    if missing_recommended:
        warnings.append(f"Missing recommended columns: {', '.join(missing_recommended)}")

    if df.empty:
        errors.append("Dataset is empty.")
        return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))

    if df["date"].isna().any():
        n = int(df["date"].isna().sum())
        errors.append(f"Unparseable dates in `date`: {n} rows.")

    if df["symbol"].isna().any() or (df["symbol"].astype(str).str.strip() == "").any():
        errors.append("Empty `symbol` values found.")

    if df["close"].isna().any():
        n = int(df["close"].isna().sum())
        errors.append(f"Non-numeric `close` values: {n} rows.")

    dup = df.duplicated(subset=["date", "symbol"], keep=False)
    if dup.any():
        n = int(dup.sum())
        errors.append(f"Duplicate (date, symbol) rows: {n}.")

    if {"open", "high", "low", "close"}.issubset(df.columns):
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        bad = (l > h) | (o < l) | (o > h) | (c < l) | (c > h)
        if bad.any():
            n = int(bad.sum())
            warnings.append(f"OHLC consistency issues: {n} rows.")

    if "volume" in df.columns:
        v = df["volume"]
        if (v < 0).fillna(False).any():
            n = int(((v < 0).fillna(False)).sum())
            warnings.append(f"Negative volume values: {n} rows.")

    symbols = int(df["symbol"].nunique(dropna=True))
    if symbols < 1:
        errors.append("No valid symbols found.")
    elif symbols == 1:
        warnings.append("Only one symbol found; comparisons/correlation will be limited.")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))

