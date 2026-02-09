from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import AppError


@dataclass(frozen=True)
class AnalyticsLimits:
    max_points: int = 5_000
    max_symbols: int = 60
    min_corr_periods: int = 30


def filter_date_range(df: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    if start is None and end is None:
        return df

    if "date" not in df.columns:
        raise AppError(code="missing_date", message="Missing `date` column.")

    start_ts = _parse_date(start) if start else None
    end_ts = _parse_date(end) if end else None

    out = df
    if start_ts is not None:
        out = out[out["date"] >= start_ts]
    if end_ts is not None:
        out = out[out["date"] <= end_ts]
    return out


def movers(df: pd.DataFrame, *, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_n < 1 or top_n > 50:
        raise AppError(code="invalid_top_n", message="Invalid top_n. Expected 1..50.")

    m = period_returns_by_symbol(df)
    if m.empty:
        return m, m
    gainers = m.head(top_n).copy()
    losers = m.tail(top_n).sort_values("period_return", ascending=True, kind="mergesort").copy()
    return gainers, losers


def period_returns_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "symbol", "close"}.issubset(df.columns):
        raise AppError(code="missing_columns", message="Expected columns: date, symbol, close")

    sub = df[["date", "symbol", "close"]].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["symbol", "start_close", "end_close", "period_return"])

    sub = sub.sort_values(["symbol", "date"], kind="mergesort")
    first = sub.groupby("symbol", sort=False).first(numeric_only=False)
    last = sub.groupby("symbol", sort=False).last(numeric_only=False)

    out = pd.DataFrame(
        {
            "symbol": first.index.astype(str),
            "start_close": first["close"].astype(float),
            "end_close": last["close"].astype(float),
        }
    )
    out["period_return"] = (out["end_close"] / out["start_close"]) - 1.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["period_return"])
    return out.sort_values("period_return", ascending=False, kind="mergesort").reset_index(drop=True)


def returns_series(df: pd.DataFrame, *, symbol: str, limits: AnalyticsLimits) -> pd.DataFrame:
    _require_symbol(df, symbol)

    sub = df.loc[df["symbol"] == symbol, ["date", "close"]].dropna().copy()
    sub = sub.sort_values("date", kind="mergesort")
    sub["return"] = sub["close"].pct_change()
    sub = _downsample(sub, limits.max_points)
    return sub


def drawdown(df: pd.DataFrame, *, symbol: str, limits: AnalyticsLimits) -> pd.DataFrame:
    _require_symbol(df, symbol)

    sub = df.loc[df["symbol"] == symbol, ["date", "close"]].dropna().copy()
    sub = sub.sort_values("date", kind="mergesort")
    px = sub["close"].astype(float).to_numpy()
    peak = np.maximum.accumulate(px)
    dd = (px / peak) - 1.0
    out = pd.DataFrame({"date": sub["date"].to_numpy(), "drawdown": dd})
    return _downsample(out, limits.max_points)


def correlation_matrix(df: pd.DataFrame, *, symbols: list[str] | None, limits: AnalyticsLimits) -> tuple[list[str], list[list[float | None]]]:
    if not {"date", "symbol", "close"}.issubset(df.columns):
        raise AppError(code="missing_columns", message="Expected columns: date, symbol, close")

    sub = df[["date", "symbol", "close"]].dropna().copy()
    if sub.empty:
        return [], []

    sub = sub.sort_values(["symbol", "date"], kind="mergesort")
    sub["return"] = sub.groupby("symbol", sort=False)["close"].pct_change()
    sub = sub.dropna(subset=["return"])

    if symbols is None or not symbols:
        counts = sub.groupby("symbol")["return"].count().sort_values(ascending=False)
        symbols = list(counts.head(limits.max_symbols).index)
    else:
        symbols = [s for s in symbols if s in set(sub["symbol"].unique())][: limits.max_symbols]

    wide = sub[sub["symbol"].isin(symbols)].pivot_table(index="date", columns="symbol", values="return", aggfunc="mean")
    corr = wide.corr(min_periods=limits.min_corr_periods)

    labels = list(corr.columns.astype(str))
    matrix: list[list[float | None]] = []
    for _, row in corr.iterrows():
        matrix.append([None if pd.isna(v) else float(v) for v in row.to_list()])
    return labels, matrix


def _parse_date(value: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value, errors="raise", utc=False)
    except Exception as e:
        raise AppError(code="invalid_date", message="Invalid date format.", details={"value": value}) from e
    if pd.isna(ts):
        raise AppError(code="invalid_date", message="Invalid date format.", details={"value": value})
    return pd.Timestamp(ts)


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points < 1:
        raise AppError(code="invalid_max_points", message="Invalid max_points.")
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def _require_symbol(df: pd.DataFrame, symbol: str) -> None:
    if not symbol or len(symbol) > 32:
        raise AppError(code="invalid_symbol", message="Invalid symbol.")
    if "symbol" not in df.columns:
        raise AppError(code="missing_symbol", message="Missing `symbol` column.")
    if symbol not in set(df["symbol"].dropna().astype(str).unique()):
        raise AppError(code="unknown_symbol", message="Unknown symbol.", details={"symbol": symbol}, status_code=404)

