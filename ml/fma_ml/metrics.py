from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class FoldMetrics:
    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float | None
    brier: float | None


def compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> FoldMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    uniq = set(int(x) for x in np.unique(y_true))
    # Balanced accuracy is most meaningful when both classes exist; fall back to accuracy otherwise.
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if uniq == {0, 1} else acc
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc: float | None = None
    if y_prob is not None:
        if uniq == {0, 1}:
            roc = float(roc_auc_score(y_true, y_prob))

    brier: float | None = None
    if y_prob is not None:
        if uniq == {0, 1}:
            # Brier score = mean((p - y)^2), lower is better.
            brier = float(brier_score_loss(y_true, y_prob))

    return FoldMetrics(accuracy=acc, balanced_accuracy=bal_acc, f1=f1, roc_auc=roc, brier=brier)

