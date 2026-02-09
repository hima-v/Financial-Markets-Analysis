from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier

from ml.fma_ml.features import FeatureConfig
from ml.fma_ml.inference import predict_next_day_direction


def test_predict_next_day_direction_runs() -> None:
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    close = pd.Series(range(100, 100 + len(dates))).astype(float)
    df = pd.DataFrame({"date": dates, "symbol": ["AAA"] * len(dates), "close": close})

    model = DummyClassifier(strategy="most_frequent").fit([[0.0], [1.0]], [0, 0])
    res = predict_next_day_direction(df, symbol="AAA", model=model, feature_cfg=FeatureConfig(windows=(5, )))
    assert res.symbol == "AAA"
    assert 0.0 <= res.prob_up <= 1.0
    assert isinstance(res.features, dict)
    assert "close" in res.features

