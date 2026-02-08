from __future__ import annotations

import numpy as np
import pandas as pd


def with_returns(df: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "symbol", "close"}.issubset(df.columns):
        raise ValueError("Expected columns: date, symbol, close")

    out = df[["date", "symbol", "close"]].copy()
    out = out.dropna(subset=["date", "symbol", "close"])
    out = out.sort_values(["symbol", "date"], kind="mergesort")
    out["return"] = out.groupby("symbol", sort=False)["close"].pct_change()
    return out


def summary_for_symbol(df: pd.DataFrame, symbol: str) -> dict[str, float]:
    r = df.loc[df["symbol"] == symbol, "return"].dropna()
    if r.empty:
        return {"count": 0.0}

    ann_factor = 252.0
    vol = float(r.std(ddof=1) * np.sqrt(ann_factor)) if len(r) > 1 else float("nan")
    mean = float(r.mean() * ann_factor)
    sharpe = float(mean / vol) if vol and np.isfinite(vol) and vol != 0.0 else float("nan")

    return {
        "count": float(len(r)),
        "annualized_return": mean,
        "annualized_volatility": vol,
        "sharpe_like": sharpe,
    }


def drawdown_series(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    px = (
        df.loc[df["symbol"] == symbol, ["date", "close"]]
        .dropna()
        .sort_values("date", kind="mergesort")
        .set_index("date")["close"]
    )
    if px.empty:
        return pd.DataFrame(columns=["date", "drawdown"])

    peak = px.cummax()
    dd = (px / peak) - 1.0
    return dd.rename("drawdown").reset_index()


def returns_pivot(df: pd.DataFrame, *, max_symbols: int = 30) -> pd.DataFrame:
    counts = df.groupby("symbol")["return"].count().sort_values(ascending=False)
    symbols = list(counts.head(max_symbols).index)
    sub = df[df["symbol"].isin(symbols)].copy()
    wide = sub.pivot_table(index="date", columns="symbol", values="return", aggfunc="mean")
    return wide.dropna(axis=0, how="all")


def returns_pivot_for_symbols(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    sub = df[df["symbol"].isin(symbols)].copy()
    wide = sub.pivot_table(index="date", columns="symbol", values="return", aggfunc="mean")
    return wide.sort_index().dropna(axis=0, how="all")


def prices_pivot_for_symbols(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    if not {"date", "symbol", "close"}.issubset(df.columns):
        raise ValueError("Expected columns: date, symbol, close")
    sub = df[df["symbol"].isin(symbols)].copy()
    wide = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="mean")
    return wide.sort_index().dropna(axis=0, how="all")


def period_returns_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "symbol", "close"}.issubset(df.columns):
        raise ValueError("Expected columns: date, symbol, close")

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

