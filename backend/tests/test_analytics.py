from __future__ import annotations

import pandas as pd

from backend.app.services.analytics import AnalyticsLimits, correlation_matrix, drawdown, movers, returns_series


def _df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                ]
            ),
            "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB"],
            "close": [100.0, 110.0, 50.0, 49.0, 60.0],
        }
    )
    return df


def test_movers_returns_top_and_bottom() -> None:
    gainers, losers = movers(_df(), top_n=1)
    assert len(gainers) == 1
    assert len(losers) == 1
    assert set(gainers.columns) >= {"symbol", "period_return"}


def test_returns_series_has_return_column() -> None:
    s = returns_series(_df(), symbol="AAA", limits=AnalyticsLimits(max_points=100))
    assert set(s.columns) >= {"date", "close", "return"}


def test_drawdown_bounds() -> None:
    dd = drawdown(_df(), symbol="AAA", limits=AnalyticsLimits(max_points=100))
    assert dd["drawdown"].max() <= 0.0 + 1e-12
    assert dd["drawdown"].min() >= -1.0 - 1e-12


def test_correlation_matrix_shapes() -> None:
    labels, matrix = correlation_matrix(_df(), symbols=None, limits=AnalyticsLimits(max_symbols=10, min_corr_periods=1))
    assert len(labels) == len(matrix)
    assert all(len(row) == len(labels) for row in matrix)

