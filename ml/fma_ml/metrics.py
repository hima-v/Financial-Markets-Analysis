from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class FoldMetrics:
    accuracy: float
    f1: float
    roc_auc: float | None


def compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> FoldMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc: float | None = None
    if y_prob is not None:
        uniq = set(int(x) for x in np.unique(y_true))
        if uniq == {0, 1}:
            roc = float(roc_auc_score(y_true, y_prob))

    return FoldMetrics(accuracy=acc, f1=f1, roc_auc=roc)

