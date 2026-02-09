from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureConfig, build_feature_frame


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
) -> InferenceResult:
    feats = build_feature_frame(df, symbol=symbol, cfg=feature_cfg)
    if feats.empty:
        raise ValueError("Not enough data to compute features.")

    last = feats.sort_values("date", kind="mergesort").iloc[-1]
    x_row = last.drop(labels=["date"])
    x = x_row.to_numpy(dtype=float, copy=True).reshape(1, -1)

    prob_up = _predict_proba_up(model, x)
    label = 1 if prob_up >= 0.5 else 0
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

