from __future__ import annotations

import pandas as pd

from ml.fma_ml.models import ModelConfig
from ml.fma_ml.splits import SplitConfig
from ml.fma_ml.train import FeatureConfig, TrainConfig, walk_forward_cv


def test_walk_forward_cv_returns_metrics() -> None:
    dates = pd.date_range("2021-01-01", periods=260, freq="D")
    close = pd.Series(range(100, 100 + len(dates))).astype(float)
    df = pd.DataFrame({"date": dates, "symbol": ["AAA"] * len(dates), "close": close})

    cfg = TrainConfig(
        feature=FeatureConfig(windows=(5, 20)),
        split=SplitConfig(n_splits=5),
        model=ModelConfig(kind="dummy"),
    )
    out = walk_forward_cv(df, symbol="AAA", cfg=cfg)
    assert "aggregate" in out
    assert "accuracy_mean" in out["aggregate"]

