from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import (
    drawdown_series,
    period_returns_by_symbol,
    prices_pivot_for_symbols,
    returns_pivot_for_symbols,
    summary_for_symbol,
    with_returns,
)
from src.charts import (
    corr_heatmap,
    drawdown_line,
    movers_bar,
    normalized_price_lines,
    price_line,
    returns_hist,
    volume_bar,
)
from src.data_loading import coerce_types, load_csv_bytes, load_repo_default, normalize_columns
from src.validation import validate_dataset


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@st.cache_data(show_spinner=False)
def _load_and_prepare_csv_bytes(data: bytes, *, nrows: int | None) -> pd.DataFrame:
    df = load_csv_bytes(data, nrows=nrows)
    df = normalize_columns(df)
    df = coerce_types(df)
    return df


@st.cache_data(show_spinner=False)
def _load_and_prepare_repo_default(*, nrows: int | None) -> pd.DataFrame:
    df = load_repo_default(nrows=nrows)
    df = normalize_columns(df)
    df = coerce_types(df)
    return df


def _sidebar_controls() -> dict[str, object]:
    st.sidebar.header("Data source")
    source = st.sidebar.radio("Choose", options=["Upload a CSV", "Use repo dataset"], index=1, label_visibility="collapsed")
    st.sidebar.caption("Use the repo dataset to explore quickly, or upload your own CSV.")
    nrows = st.sidebar.number_input(
        "Rows to load",
        min_value=500,
        max_value=200_000,
        value=50_000,
        step=500,
        help="This keeps the app fast on large files.",
    )
    return {"source": source, "nrows": int(nrows)}


def _load_dataset(source: str, *, nrows: int) -> pd.DataFrame | None:
    if source == "Use repo dataset":
        try:
            return _load_and_prepare_repo_default(nrows=nrows)
        except Exception as e:
            st.error(str(e))
            return None

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return None
    if getattr(uploaded, "size", None) and uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large. Please upload a file under 50 MB.")
        return None

    try:
        data = uploaded.getvalue()
        return _load_and_prepare_csv_bytes(data, nrows=nrows)
    except Exception:
        st.error("Could not read this file as CSV.")
        return None


def main() -> None:
    st.set_page_config(page_title="Financial Markets Explorer", layout="wide")
    st.title("Financial Markets Explorer")
    st.caption("Upload a dataset and explore prices, returns, risk, and correlations.")

    st.markdown(
        """
        <style>
          :root { --accent: #38bdf8; }
          .block-container { padding-top: 2rem; }
          [data-testid="stSidebar"] { padding-top: 2rem; }
          [data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); padding: 0.75rem; border-radius: 12px; }
          [data-testid="stSidebar"] [data-baseweb="select"] input { caret-color: transparent; cursor: pointer; }
          [data-testid="stSidebar"] [data-baseweb="select"]:focus-within input { caret-color: auto; cursor: text; }
          .filter-pill { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.22); padding: 0.5rem 0.75rem; border-radius: 999px; font-size: 0.9rem; }
          .muted { color: rgba(255,255,255,0.7); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How to use this dashboard", expanded=False):
        st.markdown(
            """
            - Upload a CSV (or use the repo dataset).
            - Pick a stock and date range in the sidebar.
            - Use **Compare** to overlay stocks and view correlations.
            - Use **Data quality** to check consistency and download a cleaned CSV.
            """
        )

        st.markdown("Accepted columns (case-insensitive). Minimum required:")
        st.code("DATE, SYMBOL, CLOSE", language=None)
        st.markdown("Recommended for richer charts:")
        st.code("OPEN, HIGH, LOW, VOLUME", language=None)

        template = pd.DataFrame(
            [
                {
                    "DATE": "2021-01-04",
                    "SYMBOL": "ASIANPAINT",
                    "OPEN": 876.2,
                    "HIGH": 892.45,
                    "LOW": 871.7,
                    "CLOSE": 880.8,
                    "VOLUME": 709103,
                }
            ]
        )
        st.dataframe(template, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="market_data_template.csv",
            mime="text/csv",
        )

    controls = _sidebar_controls()
    df = _load_dataset(str(controls["source"]), nrows=int(controls["nrows"]))
    if df is None:
        st.info("Upload a CSV to start, or use the repo dataset.")
        return

    report = validate_dataset(df)
    if report.errors:
        for msg in report.errors:
            st.error(msg)
        st.stop()
    for msg in report.warnings:
        st.warning(msg)

    returns_df = with_returns(df)
    symbols = sorted(returns_df["symbol"].dropna().unique().tolist())
    if not symbols:
        st.error("No symbols found after cleaning.")
        return

    st.sidebar.divider()
    st.sidebar.header("Filters")
    symbol = st.sidebar.selectbox("Stock name", options=symbols, index=0)
    date_min = returns_df["date"].min()
    date_max = returns_df["date"].max()

    date_range_value: tuple | None = None
    if pd.notna(date_min) and pd.notna(date_max):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(date_min.date(), date_max.date()),
            min_value=date_min.date(),
            max_value=date_max.date(),
            help="Filtering updates all charts.",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_range_value = date_range
            start, end = date_range
            mask = (returns_df["date"] >= pd.Timestamp(start)) & (returns_df["date"] <= pd.Timestamp(end))
            returns_df = returns_df.loc[mask].copy()
            df = df.loc[mask].copy()

    date_label = "All dates"
    if date_range_value is not None:
        date_label = f"{date_range_value[0]} → {date_range_value[1]}"
    st.sidebar.markdown(
        f"<div class='filter-pill'><span class='muted'>Stock</span> <b>{symbol}</b> &nbsp;·&nbsp; <span class='muted'>Range</span> <b>{date_label}</b></div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_symbol, tab_compare, tab_quality = st.tabs(["Overview", "Stock details", "Compare", "Data quality"])

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", value=f"{len(df):,}")
        c2.metric("Stocks", value=f"{df['symbol'].nunique():,}")
        c3.metric("Start", value=str(df["date"].min().date()) if pd.notna(df["date"].min()) else "—")
        c4.metric("End", value=str(df["date"].max().date()) if pd.notna(df["date"].max()) else "—")

        movers = period_returns_by_symbol(df)
        if not movers.empty and movers["symbol"].nunique() >= 2:
            st.subheader("Top movers")
            st.caption("Period return is computed from the first to the last close in the selected range.")
            top_n = 8
            gainers = movers.head(top_n).copy()
            losers = movers.tail(top_n).sort_values("period_return", ascending=True, kind="mergesort").copy()

            g_col, l_col = st.columns(2)
            with g_col:
                st.plotly_chart(movers_bar(gainers, title="Top gainers"), use_container_width=True)
            with l_col:
                st.plotly_chart(movers_bar(losers, title="Top losers"), use_container_width=True)

        with st.expander("Data preview"):
            st.dataframe(df.head(50), use_container_width=True)

    with tab_symbol:
        st.subheader(symbol)
        stats = summary_for_symbol(returns_df, symbol)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Return count", value=f"{int(stats.get('count', 0.0)):,}")
        s2.metric("Ann. return", value=f"{stats.get('annualized_return', float('nan')):.2%}")
        s3.metric("Ann. vol", value=f"{stats.get('annualized_volatility', float('nan')):.2%}")
        s4.metric("Sharpe-like", value=f"{stats.get('sharpe_like', float('nan')):.2f}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(price_line(df, symbol), use_container_width=True)
        with right:
            st.plotly_chart(returns_hist(returns_df, symbol), use_container_width=True)

        v = volume_bar(df, symbol)
        if v is not None:
            st.plotly_chart(v, use_container_width=True)

        dd = drawdown_series(returns_df, symbol)
        if not dd.empty:
            st.plotly_chart(drawdown_line(dd, symbol), use_container_width=True)

    with tab_compare:
        counts = returns_df.groupby("symbol")["return"].count().sort_values(ascending=False)
        default = list(counts.head(8).index)
        selected = st.multiselect("Stocks to compare", options=list(counts.index), default=default, max_selections=12)
        if len(selected) < 2:
            st.info("Select at least two stocks to compare.")
        else:
            price_wide = prices_pivot_for_symbols(df, selected)
            if not price_wide.empty:
                st.plotly_chart(normalized_price_lines(price_wide), use_container_width=True)

            wide = returns_pivot_for_symbols(returns_df, selected)
            if wide.shape[1] >= 2:
                st.plotly_chart(corr_heatmap(wide), use_container_width=True)
                st.caption("Correlation is computed on daily returns.")

    with tab_quality:
        st.subheader("Checks")
        st.write("Required columns: `date`, `symbol`, `close`")
        st.write("Recommended: `open`, `high`, `low`, `volume`")

        dup = df.duplicated(subset=["date", "symbol"], keep=False)
        st.metric("Duplicate (date, symbol)", value=int(dup.sum()))

        missing = df[["date", "symbol", "close"]].isna().sum()
        st.dataframe(missing.rename("missing_rows").to_frame(), use_container_width=True)

        st.subheader("Download cleaned dataset")
        cleaned = df.copy()
        cleaned["date"] = cleaned["date"].dt.date
        st.download_button(
            "Download CSV",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

