from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from .upload import read_upload_bytes
from ..core.config import settings
from ..core.errors import AppError
from ..services.ingest import IngestLimits, load_csv_bytes
from ..services.storage import load_dataset

from ml.fma_ml.artifacts import load_model
from ml.fma_ml.features import FeatureConfig
from ml.fma_ml.inference import predict_next_day_direction


router = APIRouter(prefix="/ml", tags=["ml"])


async def _load_frame(*, dataset_id: str | None, file: UploadFile | None):
    if bool(dataset_id) == bool(file):
        raise AppError(code="invalid_input", message="Provide exactly one of dataset_id or file.")
    if dataset_id:
        return load_dataset(dataset_id)
    assert file is not None
    data = await read_upload_bytes(file)
    return load_csv_bytes(data, limits=IngestLimits())


@router.post("/predict")
async def predict_endpoint(
    run_id: Annotated[str, Form()] = "",
    symbol: Annotated[str, Form()] = "",
    include_features: Annotated[bool, Form()] = False,
    dataset_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(None),
) -> dict:
    if not run_id or len(run_id) != 32:
        raise AppError(code="invalid_run_id", message="Invalid run_id.")
    if not symbol or len(symbol) > 32:
        raise AppError(code="invalid_symbol", message="Invalid symbol.")

    df = await _load_frame(dataset_id=dataset_id, file=file)

    try:
        model = load_model(artifacts_dir=Path(settings.artifacts_dir), run_id=run_id)
    except FileNotFoundError:
        raise AppError(code="model_not_found", message="Model run not found.", status_code=404)
    except ValueError as e:
        raise AppError(code="invalid_run_id", message=str(e))

    try:
        res = predict_next_day_direction(df, symbol=symbol, model=model, feature_cfg=FeatureConfig())
    except ValueError as e:
        raise AppError(code="inference_failed", message=str(e))
    out = {"symbol": res.symbol, "as_of": res.as_of, "prob_up": res.prob_up, "predicted_label": res.predicted_label}
    if include_features:
        out["features"] = res.features
        if res.explanation is not None:
            out["explanation"] = res.explanation
    return out

