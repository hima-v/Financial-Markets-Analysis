from __future__ import annotations

import pandas as pd

from backend.app.services.validate import validate_dataset


def test_validate_ok_minimal() -> None:
    df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02"],
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 101.0],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    report = validate_dataset(df)
    assert report.ok


def test_validate_missing_required_columns() -> None:
    df = pd.DataFrame({"date": ["2021-01-01"], "symbol": ["AAA"]})
    report = validate_dataset(df)
    assert not report.ok
    assert any("Missing required columns" in e for e in report.errors)


def test_validate_duplicate_date_symbol_is_error() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 100.5],
        }
    )
    report = validate_dataset(df)
    assert not report.ok
    assert any("Duplicate (date, symbol)" in e for e in report.errors)


def test_validate_ohlc_inconsistency_is_warning() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "symbol": ["AAA", "AAA"],
            "open": [10.0, 10.0],
            "high": [9.0, 12.0],
            "low": [8.0, 9.0],
            "close": [9.5, 11.0],
        }
    )
    report = validate_dataset(df)
    assert report.ok
    assert any("OHLC consistency issues" in w for w in report.warnings)

