from __future__ import annotations

from ml.fma_ml.splits import SplitConfig, time_series_splits


def test_time_series_splits_are_ordered() -> None:
    splits = time_series_splits(200, cfg=SplitConfig(n_splits=5))
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() < te.min()

