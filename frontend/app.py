from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import plotly.express as px

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
from src.data_loading import (
    coerce_types,
    load_csv_bytes,
    load_repo_default,
    normalize_columns,
    read_repo_default_bytes,
)
from src.prediction import list_runs, post_predict
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
    st.sidebar.header("Dataset")
    source = st.sidebar.radio("Source", options=["Upload a CSV", "Use repo dataset"], index=1)
    nrows = st.sidebar.number_input(
        "Rows",
        min_value=500,
        max_value=200_000,
        value=50_000,
        step=500,
        help="Caps the number of rows loaded in the UI.",
    )
    return {"source": source, "nrows": int(nrows)}


def _load_dataset(source: str, *, nrows: int) -> pd.DataFrame | None:
    if source == "Use repo dataset":
        try:
            st.session_state["dataset_name"] = "nse_sensex (1).csv"
            st.session_state["dataset_bytes"] = read_repo_default_bytes()
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
        st.session_state["dataset_name"] = str(uploaded.name) if getattr(uploaded, "name", None) else "uploaded.csv"
        st.session_state["dataset_bytes"] = data
        return _load_and_prepare_csv_bytes(data, nrows=nrows)
    except Exception:
        st.error("Could not read this file as CSV.")
        return None


def main() -> None:
    st.set_page_config(page_title="Financial Markets Explorer", layout="wide")
    st.markdown("<div class='app-header'>Financial Markets Explorer</div>", unsafe_allow_html=True)
    st.caption("Local-first dashboard for market data analytics and baseline ML inference.")

    st.markdown(
        """
        <style>
          :root { --accent: #38bdf8; }
          .block-container { padding-top: 3rem; }
          [data-testid="stSidebar"] { padding-top: 2rem; }
          [data-testid="stMetric"] { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 0.7rem; border-radius: 12px; }
          [data-testid="stSidebar"] [data-baseweb="select"] input { caret-color: transparent; cursor: pointer; }
          [data-testid="stSidebar"] [data-baseweb="select"]:focus-within input { caret-color: auto; cursor: text; }
          .app-header { font-size: 2.25rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; margin: 0.25rem 0 0.25rem 0; word-break: break-word; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("About this project", expanded=False):
        st.markdown(
            """
            - Explore historical stock price data with basic risk/return analytics.
            - Compare instruments over a chosen date range (normalized prices and correlations).
            - Validate dataset format and quality before analysis.
            - Run local baseline inference using a saved model run.
            """
        )

        st.markdown("**Model overview**")
        st.markdown(
            """
            - **Task**: predict whether next-day close is higher than today’s close (binary direction).
            - **Model**: logistic regression with standardization; evaluated with walk-forward splits.
            - **Features**: rolling return mean/volatility, price range, and rolling volume statistics.
            - **Interpretation**: contributions indicate which standardized features push probability up/down.
            """
        )

    controls = _sidebar_controls()
    df = _load_dataset(str(controls["source"]), nrows=int(controls["nrows"]))
    if df is None:
        st.info("Select a dataset source in the sidebar.")
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
    symbol = st.sidebar.selectbox("Stock", options=symbols, index=0)
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

    tab_home, tab_overview, tab_compare, tab_predict, tab_quality = st.tabs(
        ["Home", "Overview", "Compare", "Predict", "Data quality"]
    )

    with tab_home:
        with st.container(border=True):
            st.subheader("Start here")
            st.write("Pick a dataset source in the sidebar, then explore analytics by stock and date range.")

        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(border=True):
                st.subheader("Workflow")
                st.write("1) Select dataset")
                st.write("2) Explore overview + compare")
                st.write("3) Predict (optional)")

        with c2:
            with st.container(border=True):
                st.subheader("Dataset format")
                st.write("Required: DATE, SYMBOL, CLOSE")
                st.write("Recommended: OPEN, HIGH, LOW, VOLUME")

        with st.container(border=True):
            st.subheader("What’s saved locally")
            st.write("- Uploaded datasets: `data/datasets/<dataset_id>/`")
            st.write("- Trained models: `artifacts/<run_id>/`")

    with tab_overview:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", value=f"{len(df):,}")
            c2.metric("Stocks", value=f"{df['symbol'].nunique():,}")
            c3.metric("Start", value=str(df["date"].min().date()) if pd.notna(df["date"].min()) else "—")
            c4.metric("End", value=str(df["date"].max().date()) if pd.notna(df["date"].max()) else "—")

        movers = period_returns_by_symbol(df)
        if not movers.empty and movers["symbol"].nunique() >= 2:
            with st.container(border=True):
                st.subheader("Top movers")
                top_n = 8
                gainers = movers.head(top_n).copy()
                losers = movers.tail(top_n).sort_values("period_return", ascending=True, kind="mergesort").copy()
                g_col, l_col = st.columns(2)
                with g_col:
                    st.plotly_chart(movers_bar(gainers, title="Gainers"), use_container_width=True)
                with l_col:
                    st.plotly_chart(movers_bar(losers, title="Losers"), use_container_width=True)

        with st.container(border=True):
            st.subheader("Selected stock")
            stats = summary_for_symbol(returns_df, symbol)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Days", value=f"{int(stats.get('count', 0.0)):,}")
            s2.metric("Ann. return", value=f"{stats.get('annualized_return', float('nan')):.2%}")
            s3.metric("Ann. vol", value=f"{stats.get('annualized_volatility', float('nan')):.2%}")
            s4.metric("Sharpe-like", value=f"{stats.get('sharpe_like', float('nan')):.2f}")

            left, right = st.columns(2)
            with left:
                st.plotly_chart(price_line(df, symbol), use_container_width=True)
            with right:
                st.plotly_chart(returns_hist(returns_df, symbol), use_container_width=True)
                st.caption("A wider spread suggests higher day-to-day volatility.")

    with tab_compare:
        with st.container(border=True):
            counts = returns_df.groupby("symbol")["return"].count().sort_values(ascending=False)
            default = list(counts.head(8).index)
            selected = st.multiselect("Stocks", options=list(counts.index), default=default, max_selections=12)
            if len(selected) < 2:
                st.info("Select at least two stocks.")
            else:
                price_wide = prices_pivot_for_symbols(df, selected)
                if not price_wide.empty:
                    st.plotly_chart(normalized_price_lines(price_wide), use_container_width=True)

                wide = returns_pivot_for_symbols(returns_df, selected)
                if wide.shape[1] >= 2:
                    st.plotly_chart(corr_heatmap(wide), use_container_width=True)
                    st.caption("Higher correlation means returns tend to move together over the selected period.")

    with tab_predict:
        runs = list_runs(artifacts_dir=Path("artifacts"))
        run_options = {r.label: r.run_id for r in runs}
        if "predict_result" not in st.session_state:
            st.session_state["predict_result"] = None

        run_id = ""

        with st.container(border=True):
            st.subheader("Step 1 — Dataset")
            source_mode = st.radio(" ", options=["Use current dataset", "Use dataset_id"], horizontal=True, label_visibility="collapsed")
            dataset_id = ""
            file_bytes = None
            file_name = None
            if source_mode == "Use dataset_id":
                dataset_id = st.text_input("dataset_id", value="", placeholder="e.g. 46648df5...")
            else:
                file_bytes = st.session_state.get("dataset_bytes")
                file_name = st.session_state.get("dataset_name")
                st.caption(f"Using: {file_name or '—'}")

        with st.container(border=True):
            st.subheader("Step 2 — Model")
            if not run_options:
                st.warning("No saved runs found in `artifacts/`.")
            else:
                selected_label = st.selectbox("Model run", options=list(run_options.keys()), index=0)
                run_id = run_options[selected_label]

        with st.container(border=True):
            with st.expander("Model details", expanded=False):
                if not run_id:
                    st.caption("Select a model run to view details.")
                else:
                    run_path = Path("artifacts") / run_id / "run.json"
                    try:
                        data = json.loads(run_path.read_text(encoding="utf-8"))
                    except Exception:
                        st.caption("Model details are unavailable.")
                    else:
                        model_kind = str(data.get("model_kind", "—"))
                        created_at = str(data.get("created_at", "—"))
                        trained_symbol = str(data.get("symbol", "—"))
                        train_start = data.get("train_start", None)
                        train_end = data.get("train_end", None)
                        train_window = "—"
                        if isinstance(train_start, str) and isinstance(train_end, str):
                            train_window = f"{train_start} → {train_end}"

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Model**: {model_kind}")
                            st.write(f"**Trained on stock**: {trained_symbol}")
                        with c2:
                            st.write(f"**Run**: {run_id}")
                            st.write(f"**Timestamp**: {created_at}")

                        st.write(f"**Training window**: {train_window}")
                        st.write("**Feature groups**: rolling returns and volatility, price range, rolling volume.")

        with st.container(border=True):
            st.subheader("Step 3 — Predict")
            with st.expander("Advanced", expanded=False):
                base_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
            can_predict = (
                run_id != ""
                and ((dataset_id.strip() != "") or (file_bytes is not None and len(file_bytes) > 0))
            )
            submitted = st.button(f"Predict for {symbol}", type="primary", disabled=not can_predict, use_container_width=True)
            if submitted:
                with st.spinner("Running inference..."):
                    try:
                        st.session_state["predict_result"] = post_predict(
                            base_url=base_url,
                            run_id=run_id,
                            symbol=symbol,
                            dataset_id=dataset_id.strip() or None,
                            file_name=file_name,
                            file_bytes=file_bytes,
                        )
                    except Exception as e:
                        st.session_state["predict_result"] = {"error": str(e)}

        with st.container(border=True):
            st.subheader("Result")
            result = st.session_state.get("predict_result")
            if not result:
                st.caption("Run a prediction to see output here.")
            elif "error" in result:
                st.error(result["error"])
            else:
                r1, r2, r3 = st.columns(3)
                prob = float(result["prob_up"])
                lbl = int(result["predicted_label"])
                r1.metric("Prob(up)", value=f"{prob:.3f}")
                r2.metric("Direction", value="Up" if lbl == 1 else "Down")
                r3.metric("As of", value=str(result["as_of"]))
                st.caption("Direction is assigned using a 0.5 probability threshold.")

                expl = result.get("explanation", None)
                if isinstance(expl, dict) and expl.get("method") == "logreg_contributions":
                    pos = expl.get("top_positive", [])
                    neg = expl.get("top_negative", [])
                    items = []
                    for d in neg:
                        items.append({"feature": d["feature"], "contribution": float(d["contribution"])})
                    for d in pos:
                        items.append({"feature": d["feature"], "contribution": float(d["contribution"])})
                    if items:
                        edf = pd.DataFrame(items).dropna().sort_values("contribution")
                        fig = px.bar(edf, x="contribution", y="feature", orientation="h")
                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Positive contributions push the probability toward “Up”; negative contributions push toward “Down”.")

                feats = result.get("features", {})
                if isinstance(feats, dict) and feats:
                    with st.expander("Features", expanded=False):
                        fdf = (
                            pd.DataFrame([{"feature": k, "value": float(v)} for k, v in feats.items()])
                            .sort_values("feature")
                            .reset_index(drop=True)
                        )
                        st.dataframe(fdf, use_container_width=True, hide_index=True)

        st.caption("This is a baseline model for exploration and does not constitute financial advice.")

    with tab_quality:
        with st.container(border=True):
            st.subheader("Checks")
            dup = df.duplicated(subset=["date", "symbol"], keep=False)
            st.metric("Duplicate (date, symbol)", value=int(dup.sum()))

            missing = df[["date", "symbol", "close"]].isna().sum()
            st.dataframe(missing.rename("missing_rows").to_frame(), use_container_width=True)

        with st.container(border=True):
            st.subheader("Export")
            cleaned = df.copy()
            cleaned["date"] = cleaned["date"].dt.date
            st.download_button(
                "Download cleaned CSV",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

