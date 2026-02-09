from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .constants import DATASET_ID_RE


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    created_at: str
    symbol: str
    model_kind: str
    n_rows: int
    n_features: int
    metrics: dict


def save_run(*, output_dir: Path, model, info: RunInfo) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_path = output_dir / info.run_id
    run_path.mkdir(parents=True, exist_ok=False)

    joblib.dump(model, run_path / "model.joblib")
    _atomic_write_json(run_path / "run.json", asdict(info))
    return run_path


def new_run_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(*, artifacts_dir: Path, run_id: str) -> Path:
    if not DATASET_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run_id.")
    base = artifacts_dir.resolve()
    path = (base / run_id).resolve()
    if base not in path.parents:
        raise ValueError("Invalid run path.")
    return path


def load_model(*, artifacts_dir: Path, run_id: str):
    p = run_dir(artifacts_dir=artifacts_dir, run_id=run_id)
    model_path = p / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    return joblib.load(model_path)


def load_run_info(*, artifacts_dir: Path, run_id: str) -> dict:
    p = run_dir(artifacts_dir=artifacts_dir, run_id=run_id)
    run_path = p / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(str(run_path))
    return json.loads(run_path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

