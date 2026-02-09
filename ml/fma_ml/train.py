from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

from .baselines import baseline_always_up, baseline_majority_class, baseline_momentum_rule
from .features import FeatureConfig, build_supervised_frame
from .metrics import compute_metrics
from .models import ModelConfig, build_model
from .splits import SplitConfig, time_series_splits


@dataclass(frozen=True)
class CalibrationConfig:
    # "none" (default), "sigmoid" (Platt scaling), "isotonic"
    method: str = "none"
    calib_fraction: float = 0.2


@dataclass(frozen=True)
class TrainConfig:
    feature: FeatureConfig = FeatureConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    calibration: CalibrationConfig = CalibrationConfig()


def walk_forward_cv(df: pd.DataFrame, *, symbol: str, cfg: TrainConfig) -> dict:
    frame = build_supervised_frame(df, symbol=symbol, cfg=cfg.feature)
    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore")
    y = frame["target_up"].astype(int).to_numpy()

    feature_names = list(x.columns)
    splits = time_series_splits(len(frame), cfg=cfg.split)
    fold_metrics: list[dict] = []
    baseline_folds: dict[str, list[dict]] = {
        "dummy_most_frequent": [],
        "always_up": [],
        "momentum_ret1d": [],
    }

    for train_idx, test_idx in splits:
        x_train_df = x.iloc[train_idx]
        x_test_df = x.iloc[test_idx]
        x_train = x_train_df.to_numpy()
        y_train = y[train_idx]
        x_test = x_test_df.to_numpy()
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
        fold_metrics.append(
            {
                "accuracy": m.accuracy,
                "balanced_accuracy": m.balanced_accuracy,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
                "brier": m.brier,
            }
        )

        # Baselines (educational, time-aware comparisons)
        b = baseline_majority_class(y_train, n=len(test_idx))
        bm = compute_metrics(y_true=y_test, y_pred=b.y_pred, y_prob=b.y_prob)
        baseline_folds["dummy_most_frequent"].append(
            {
                "accuracy": bm.accuracy,
                "balanced_accuracy": bm.balanced_accuracy,
                "f1": bm.f1,
                "roc_auc": bm.roc_auc,
                "brier": bm.brier,
            }
        )

        a = baseline_always_up(n=len(test_idx))
        am = compute_metrics(y_true=y_test, y_pred=a.y_pred, y_prob=a.y_prob)
        baseline_folds["always_up"].append(
            {
                "accuracy": am.accuracy,
                "balanced_accuracy": am.balanced_accuracy,
                "f1": am.f1,
                "roc_auc": am.roc_auc,
                "brier": am.brier,
            }
        )

        mo = baseline_momentum_rule(x_test=x_test, feature_names=feature_names)
        mm = compute_metrics(y_true=y_test, y_pred=mo.y_pred, y_prob=mo.y_prob)
        baseline_folds["momentum_ret1d"].append(
            {
                "accuracy": mm.accuracy,
                "balanced_accuracy": mm.balanced_accuracy,
                "f1": mm.f1,
                "roc_auc": mm.roc_auc,
                "brier": mm.brier,
            }
        )

    agg = _aggregate(fold_metrics)
    baselines = {name: {"folds": folds, "aggregate": _aggregate(folds)} for name, folds in baseline_folds.items()}
    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "baselines": baselines,
        "n_rows": int(len(frame)),
        "n_features": int(x.shape[1]),
    }


def fit_final_model(df: pd.DataFrame, *, symbol: str, cfg: TrainConfig):
    frame = build_supervised_frame(df, symbol=symbol, cfg=cfg.feature)
    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore")
    y = frame["target_up"].astype(int).to_numpy()

    x_np = x.to_numpy()
    model = _fit_model(x_np, y, cfg=cfg.model)

    calib_method = cfg.calibration.method.strip().lower()
    calib_info: dict | None = None
    if calib_method in {"sigmoid", "isotonic"}:
        frac = float(cfg.calibration.calib_fraction)
        if not (0.05 <= frac <= 0.5):
            raise ValueError("calib_fraction must be between 0.05 and 0.5.")

        n = len(frame)
        calib_n = int(max(50, round(n * frac)))
        if n - calib_n < 50:
            raise ValueError("Not enough rows to perform time-respecting calibration split.")

        # Time-respecting split: earliest -> train, most recent -> calibration
        train_end = n - calib_n
        x_train = x_np[:train_end]
        y_train = y[:train_end]
        x_cal = x_np[train_end:]
        y_cal = y[train_end:]

        model.fit(x_train, y_train)
        prob_before = _predict_proba_up_vec(model, x_cal)
        pred_before = (prob_before >= 0.5).astype(int)
        m_before = compute_metrics(y_true=y_cal, y_pred=pred_before, y_prob=prob_before)

        cal = CalibratedClassifierCV(model, method=calib_method, cv="prefit")
        cal.fit(x_cal, y_cal)

        prob_after = _predict_proba_up_vec(cal, x_cal)
        pred_after = (prob_after >= 0.5).astype(int)
        m_after = compute_metrics(y_true=y_cal, y_pred=pred_after, y_prob=prob_after)

        calib_info = {
            "method": calib_method,
            "calib_fraction": frac,
            "calib_rows": int(calib_n),
            "train_rows": int(train_end),
            "metrics_before": {
                "accuracy": m_before.accuracy,
                "balanced_accuracy": m_before.balanced_accuracy,
                "f1": m_before.f1,
                "roc_auc": m_before.roc_auc,
                "brier": m_before.brier,
            },
            "metrics_after": {
                "accuracy": m_after.accuracy,
                "balanced_accuracy": m_after.balanced_accuracy,
                "f1": m_after.f1,
                "roc_auc": m_after.roc_auc,
                "brier": m_after.brier,
            },
        }
        return cal, int(len(frame)), int(x.shape[1]), calib_info

    if calib_method not in {"none", ""}:
        raise ValueError("Unknown calibration method.")

    model.fit(x_np, y)
    return model, int(len(frame)), int(x.shape[1]), calib_info


def _aggregate(folds: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [f[key] for f in folds if f.get(key) is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    return {
        "accuracy_mean": mean("accuracy"),
        "balanced_accuracy_mean": mean("balanced_accuracy"),
        "f1_mean": mean("f1"),
        "roc_auc_mean": mean("roc_auc"),
        "brier_mean": mean("brier"),
    }


def _fit_model(x: np.ndarray, y: np.ndarray, *, cfg: ModelConfig):
    uniq = set(int(v) for v in np.unique(y))
    if uniq != {0, 1}:
        return DummyClassifier(strategy="most_frequent")
    return build_model(cfg)


def _predict_proba_up_vec(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1].astype(float)
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)

