from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5


def time_series_splits(n_rows: int, *, cfg: SplitConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < 50:
        raise ValueError("Not enough rows for time-series evaluation.")
    if cfg.n_splits < 2 or cfg.n_splits > 20:
        raise ValueError("n_splits must be between 2 and 20.")

    tss = TimeSeriesSplit(n_splits=cfg.n_splits)
    return [(train_idx, test_idx) for train_idx, test_idx in tss.split(np.arange(n_rows))]

