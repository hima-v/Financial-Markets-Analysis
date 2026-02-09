from __future__ import annotations

import pandas as pd

from ml.fma_ml.features import FeatureConfig, build_supervised_frame


def test_build_supervised_frame_has_target_and_features() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06"]),
            "symbol": ["AAA"] * 6,
            "close": [100, 101, 99, 100, 103, 104],
            "high": [101, 102, 100, 101, 104, 105],
            "low": [99, 100, 98, 99, 102, 103],
            "volume": [10, 11, 12, 13, 14, 15],
        }
    )
    out = build_supervised_frame(df, symbol="AAA", cfg=FeatureConfig(windows=(2,)))
    assert "target_up" in out.columns
    assert any(c.startswith("ret_") for c in out.columns)
    assert len(out) > 0

