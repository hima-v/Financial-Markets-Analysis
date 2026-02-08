from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def price_line(df: pd.DataFrame, symbol: str) -> go.Figure:
    sub = df.loc[df["symbol"] == symbol, ["date", "close"]].dropna().sort_values("date")
    fig = px.line(sub, x="date", y="close", title=f"{symbol} price")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def volume_bar(df: pd.DataFrame, symbol: str) -> go.Figure | None:
    if "volume" not in df.columns:
        return None
    sub = df.loc[df["symbol"] == symbol, ["date", "volume"]].dropna().sort_values("date")
    fig = px.bar(sub, x="date", y="volume", title=f"{symbol} volume")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def returns_hist(df: pd.DataFrame, symbol: str) -> go.Figure:
    r = df.loc[df["symbol"] == symbol, "return"].dropna()
    fig = px.histogram(r, nbins=60, title=f"{symbol} daily returns")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def drawdown_line(dd_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.line(dd_df, x="date", y="drawdown", title=f"{symbol} drawdown")
    fig.update_layout(yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def corr_heatmap(wide_returns: pd.DataFrame) -> go.Figure:
    corr = wide_returns.corr(min_periods=30)
    fig = px.imshow(
        corr,
        text_auto=False,
        zmin=-1.0,
        zmax=1.0,
        title="Return correlation",
        aspect="auto",
        color_continuous_scale="RdBu",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def normalized_price_lines(wide_prices: pd.DataFrame) -> go.Figure:
    if wide_prices.empty:
        return go.Figure()

    def _norm(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        base = float(s.iloc[0])
        if base == 0.0:
            return s
        return (s / base) * 100.0

    normed = wide_prices.apply(_norm, axis=0)
    normed = normed.dropna(axis=1, how="all")
    fig = px.line(normed, title="Normalized price (start = 100)")
    fig.update_layout(legend_title_text="Stock", margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title_text="Index (start = 100)")
    return fig


def movers_bar(movers: pd.DataFrame, *, title: str) -> go.Figure:
    if movers.empty:
        return go.Figure()
    sub = movers[["symbol", "period_return"]].copy()
    sub["period_return_pct"] = sub["period_return"] * 100.0
    fig = px.bar(
        sub.sort_values("period_return_pct"),
        x="period_return_pct",
        y="symbol",
        orientation="h",
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    fig.update_xaxes(title_text="Return (%)")
    fig.update_yaxes(title_text="")
    return fig

