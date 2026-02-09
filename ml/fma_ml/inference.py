from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from .features import FeatureConfig, build_feature_frame, build_supervised_frame
from .metrics import compute_metrics


@dataclass(frozen=True)
class InferenceResult:
    symbol: str
    as_of: str
    prob_up: float
    predicted_label: int
    features: dict[str, float]
    explanation: dict | None


def predict_next_day_direction(
    df: pd.DataFrame,
    *,
    symbol: str,
    model,
    feature_cfg: FeatureConfig,
    threshold: float = 0.5,
) -> InferenceResult:
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    feats = build_feature_frame(df, symbol=symbol, cfg=feature_cfg)
    if feats.empty:
        raise ValueError("Not enough data to compute features.")

    last = feats.sort_values("date", kind="mergesort").iloc[-1]
    x_row = last.drop(labels=["date"])
    x = x_row.to_numpy(dtype=float, copy=True).reshape(1, -1)

    prob_up = _predict_proba_up(model, x)
    label = 1 if prob_up >= float(threshold) else 0
    as_of = pd.to_datetime(last["date"]).strftime("%Y-%m-%d")
    features = {str(k): float(v) for k, v in x_row.to_dict().items()}
    explanation = explain_prediction(model=model, features=features)
    return InferenceResult(
        symbol=symbol,
        as_of=as_of,
        prob_up=float(prob_up),
        predicted_label=label,
        features=features,
        explanation=explanation,
    )


def _predict_proba_up(model, x: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return float(proba[0, 1])
    pred = model.predict(x)
    return float(pred[0])


def _predict_proba_up_vec(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1].astype(float)
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def evaluate_threshold(
    df: pd.DataFrame,
    *,
    symbol: str,
    model,
    feature_cfg: FeatureConfig,
    threshold: float = 0.5,
    window: int = 252,
) -> dict:
    """
    Evaluate model behavior at a chosen probability threshold.

    Uses a recent labeled window (time-respecting): the last `window` rows from the supervised frame.
    """
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be between 0 and 1.")
    if window < 30 or window > 5000:
        raise ValueError("window must be between 30 and 5000.")

    frame = build_supervised_frame(df, symbol=symbol, cfg=feature_cfg)
    if frame.empty:
        raise ValueError("Not enough labeled data for evaluation.")

    # Keep last N labeled examples (chronological).
    frame = frame.sort_values("date", kind="mergesort").tail(int(window)).reset_index(drop=True)
    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore").to_numpy(dtype=float, copy=True)
    y_true = frame["target_up"].astype(int).to_numpy()
    y_prob = _predict_proba_up_vec(model, x)
    y_pred = (y_prob >= float(threshold)).astype(int)

    m = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "window_rows": int(len(frame)),
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "f1": m.f1,
        "roc_auc": m.roc_auc,
        "brier": m.brier,
        "positive_rate_true": float(np.mean(y_true.astype(float))) if len(y_true) else None,
        "positive_rate_pred": float(np.mean(y_pred.astype(float))) if len(y_pred) else None,
    }


def failure_analysis(
    df: pd.DataFrame,
    *,
    symbol: str,
    model,
    feature_cfg: FeatureConfig,
    threshold: float = 0.5,
) -> dict:
    """
    Simple failure analysis for educational value:
    - Metrics by volatility regime (low/med/high based on ret_vol_20 quantiles)
    - Metrics by calendar year
    """
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    frame = build_supervised_frame(df, symbol=symbol, cfg=feature_cfg)
    if frame.empty:
        raise ValueError("Not enough labeled data for analysis.")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)

    x = frame.drop(columns=["target_up"]).drop(columns=["date"], errors="ignore").to_numpy(dtype=float, copy=True)
    y_true = frame["target_up"].astype(int).to_numpy()
    y_prob = _predict_proba_up_vec(model, x)
    y_pred = (y_prob >= float(threshold)).astype(int)

    # Volatility regime: prefer ret_vol_20 if present (FeatureConfig default includes 20).
    vol = None
    if "ret_vol_20" in frame.columns:
        vol = frame["ret_vol_20"].astype(float)
    else:
        # fallback: try to find a 20-day vol feature, else compute from 1d returns
        vol_cols = [c for c in frame.columns if c.startswith("ret_vol_")]
        if vol_cols:
            vol = frame[vol_cols[0]].astype(float)
        else:
            r = frame["close"].astype(float).pct_change()
            vol = r.rolling(window=20, min_periods=20).std(ddof=1)

    vol = pd.to_numeric(vol, errors="coerce")
    q33 = float(vol.quantile(0.33))
    q66 = float(vol.quantile(0.66))

    def _regime(v: float) -> str:
        if not np.isfinite(v):
            return "unknown"
        if v <= q33:
            return "low"
        if v <= q66:
            return "mid"
        return "high"

    regimes = [ _regime(float(v)) for v in vol.to_numpy() ]
    years = pd.to_datetime(frame["date"]).dt.year.astype(int).to_list()

    def _group_metrics(mask: np.ndarray) -> dict | None:
        if mask.sum() < 30:
            return None
        yt = y_true[mask]
        yp = y_pred[mask]
        yb = y_prob[mask]
        m = compute_metrics(y_true=yt, y_pred=yp, y_prob=yb)
        return {
            "n": int(mask.sum()),
            "accuracy": m.accuracy,
            "balanced_accuracy": m.balanced_accuracy,
            "f1": m.f1,
            "roc_auc": m.roc_auc,
            "brier": m.brier,
        }

    by_regime: list[dict] = []
    for rname in ("low", "mid", "high", "unknown"):
        mask = np.array([r == rname for r in regimes], dtype=bool)
        gm = _group_metrics(mask)
        if gm is not None:
            by_regime.append({"regime": rname, **gm})

    by_year: list[dict] = []
    for y in sorted(set(years)):
        mask = np.array([yy == y for yy in years], dtype=bool)
        gm = _group_metrics(mask)
        if gm is not None:
            by_year.append({"year": int(y), **gm})

    return {
        "threshold": float(threshold),
        "by_volatility_regime": by_regime,
        "by_year": by_year,
    }


def explain_prediction(*, model, features: dict[str, float], top_k: int = 6) -> dict | None:
    try:
        pipe = getattr(model, "named_steps", None)
        if not isinstance(pipe, dict):
            return None
        scaler = pipe.get("scale")
        clf = pipe.get("clf")
        if scaler is None or clf is None:
            return None

        coef = getattr(clf, "coef_", None)
        intercept = getattr(clf, "intercept_", None)
        mean_ = getattr(scaler, "mean_", None)
        scale_ = getattr(scaler, "scale_", None)
        if coef is None or intercept is None or mean_ is None or scale_ is None:
            return None

        names = list(features.keys())
        x = np.array([features[n] for n in names], dtype=float)
        z = (x - mean_) / np.where(scale_ == 0.0, 1.0, scale_)
        w = np.array(coef[0], dtype=float)
        contrib = w * z

        items = [{"feature": n, "contribution": float(c)} for n, c in zip(names, contrib, strict=False)]
        items_sorted = sorted(items, key=lambda d: d["contribution"])
        neg = items_sorted[:top_k]
        pos = list(reversed(items_sorted[-top_k:]))
        return {
            "method": "logreg_contributions",
            "top_positive": pos,
            "top_negative": neg,
            "intercept": float(intercept[0]),
        }
    except Exception:
        return None

