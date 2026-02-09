from __future__ import annotations

import pandas as pd

from backend.app.services.ingest import IngestLimits, load_csv_bytes


def test_ingest_normalizes_columns_and_types() -> None:
    csv = "\n".join(
        [
            "DATE,SYMBOL,CLOSE,VOLUME",
            "2021-01-01,AAA,100,10",
            "2021-01-02,AAA,101,11",
        ]
    ).encode("utf-8")

    df = load_csv_bytes(csv, limits=IngestLimits(max_rows=1000, max_columns=20))
    assert set(["date", "symbol", "close", "volume"]).issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["close"])

