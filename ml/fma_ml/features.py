from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (5, 20, 60)


def build_feature_frame(df: pd.DataFrame, *, symbol: str, cfg: FeatureConfig) -> pd.DataFrame:
    if not symbol:
        raise ValueError("symbol is required.")
    if "symbol" not in df.columns:
        raise ValueError("Missing symbol column.")

    sub = df.loc[df["symbol"] == symbol].copy()
    if sub.empty:
        raise ValueError("Unknown symbol.")

    sub = sub.sort_values("date", kind="mergesort")
    sub["ret_1d"] = sub["close"].pct_change()

    for w in cfg.windows:
        r = sub["ret_1d"]
        sub[f"ret_mean_{w}"] = r.rolling(window=w, min_periods=w).mean()
        sub[f"ret_vol_{w}"] = r.rolling(window=w, min_periods=w).std(ddof=1)

    if {"high", "low", "close"}.issubset(sub.columns):
        sub["range_pct"] = (sub["high"] - sub["low"]) / sub["close"].replace(0.0, np.nan)

    if "volume" in sub.columns:
        v = sub["volume"].astype(float)
        for w in cfg.windows:
            sub[f"vol_mean_{w}"] = v.rolling(window=w, min_periods=w).mean()

    feature_cols = [c for c in sub.columns if c.startswith(("ret_", "range_", "vol_"))]
    out = sub[["date", "close", *feature_cols]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def build_supervised_frame(df: pd.DataFrame, *, symbol: str, cfg: FeatureConfig) -> pd.DataFrame:
    feats = build_feature_frame(df, symbol=symbol, cfg=cfg)
    sub = df.loc[df["symbol"] == symbol, ["date", "close"]].copy().sort_values("date", kind="mergesort")
    next_ret = (sub["close"].shift(-1) / sub["close"]) - 1.0
    sub["target_up"] = (next_ret > 0.0).astype(int)

    out = feats.merge(sub[["date", "target_up"]], on="date", how="left")
    out = out.dropna(subset=["target_up"]).copy()
    out["target_up"] = out["target_up"].astype(int)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["target_up"])
    y = df["target_up"].astype(int)
    return x, y

