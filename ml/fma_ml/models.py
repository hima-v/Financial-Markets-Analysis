from __future__ import annotations

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "logreg"
    random_state: int = 42


def build_model(cfg: ModelConfig):
    kind = cfg.kind.strip().lower()
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if kind == "logreg":
        return Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        n_jobs=1,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    raise ValueError("Unknown model kind.")

