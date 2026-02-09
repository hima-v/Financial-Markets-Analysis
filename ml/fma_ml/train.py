from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from .features import FeatureConfig, build_supervised_frame
from .metrics import compute_metrics
from .models import ModelConfig, build_model
from .splits import SplitConfig, time_series_splits


@dataclass(frozen=True)
class TrainConfig:
    feature: FeatureConfig = FeatureConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()


def walk_forward_cv(df: pd.DataFrame, *, symbol: str, cfg: TrainConfig) -> dict:
    frame = build_supervised_frame(df, symbol=symbol, cfg=cfg.feature)
    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore")
    y = frame["target_up"].astype(int).to_numpy()

    splits = time_series_splits(len(frame), cfg=cfg.split)
    fold_metrics: list[dict] = []

    for train_idx, test_idx in splits:
        x_train = x.iloc[train_idx].to_numpy()
        y_train = y[train_idx]
        x_test = x.iloc[test_idx].to_numpy()
        y_test = y[test_idx]

        model = _fit_model(x_train, y_train, cfg=cfg.model)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        y_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            if proba.shape[1] == 2:
                y_prob = proba[:, 1]

        m = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
        fold_metrics.append({"accuracy": m.accuracy, "f1": m.f1, "roc_auc": m.roc_auc})

    agg = _aggregate(fold_metrics)
    return {"folds": fold_metrics, "aggregate": agg, "n_rows": int(len(frame)), "n_features": int(x.shape[1])}


def fit_final_model(df: pd.DataFrame, *, symbol: str, cfg: TrainConfig):
    frame = build_supervised_frame(df, symbol=symbol, cfg=cfg.feature)
    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore")
    y = frame["target_up"].astype(int).to_numpy()

    model = _fit_model(x.to_numpy(), y, cfg=cfg.model)
    model.fit(x.to_numpy(), y)
    return model, int(len(frame)), int(x.shape[1])


def _aggregate(folds: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [f[key] for f in folds if f.get(key) is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    return {
        "accuracy_mean": mean("accuracy"),
        "f1_mean": mean("f1"),
        "roc_auc_mean": mean("roc_auc"),
    }


def _fit_model(x: np.ndarray, y: np.ndarray, *, cfg: ModelConfig):
    uniq = set(int(v) for v in np.unique(y))
    if uniq != {0, 1}:
        return DummyClassifier(strategy="most_frequent")
    return build_model(cfg)

