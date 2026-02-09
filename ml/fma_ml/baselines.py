from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    y_pred: np.ndarray
    y_prob: np.ndarray | None


def baseline_always_up(n: int) -> BaselineResult:
    """Predict class=1 for every example."""
    y_pred = np.ones(shape=(n,), dtype=int)
    y_prob = np.ones(shape=(n,), dtype=float)
    return BaselineResult(y_pred=y_pred, y_prob=y_prob)


def baseline_majority_class(y_train: np.ndarray, n: int) -> BaselineResult:
    """Predict the most frequent label in y_train."""
    if len(y_train) == 0:
        raise ValueError("y_train is empty.")
    # NOTE: y_train is expected to be 0/1.
    counts = np.bincount(y_train.astype(int), minlength=2)
    maj = int(np.argmax(counts))
    y_pred = np.full(shape=(n,), fill_value=maj, dtype=int)
    # Use constant probability as the training positive rate (helps Brier).
    p_up = float(np.mean(y_train.astype(float)))
    y_prob = np.full(shape=(n,), fill_value=p_up, dtype=float)
    return BaselineResult(y_pred=y_pred, y_prob=y_prob)


def baseline_momentum_rule(*, x_test: np.ndarray, feature_names: list[str]) -> BaselineResult:
    """
    Simple momentum heuristic:
    - predict UP if ret_1d > 0 else DOWN

    This is a leakage-safe rule because ret_1d at time t uses close(t)/close(t-1) to predict target_up(t),
    which depends on close(t+1).
    """
    try:
        idx = feature_names.index("ret_1d")
    except ValueError as e:
        raise ValueError("Momentum baseline requires ret_1d feature.") from e

    r = x_test[:, idx].astype(float)
    y_pred = (r > 0.0).astype(int)
    # Deterministic probabilities (educational baseline; not calibrated).
    y_prob = y_pred.astype(float)
    return BaselineResult(y_pred=y_pred, y_prob=y_prob)

