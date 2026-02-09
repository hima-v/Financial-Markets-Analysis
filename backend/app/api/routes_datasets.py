from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from .upload import read_upload_bytes
from ..core.errors import AppError
from ..services.ingest import IngestLimits, load_csv_bytes
from ..services.storage import list_dataset_ids, load_meta, save_dataset
from ..services.validate import validate_dataset


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/validate")
async def validate_upload(file: UploadFile = File(...)) -> dict:
    data = await read_upload_bytes(file)
    df = load_csv_bytes(data, limits=IngestLimits())
    report = validate_dataset(df)

    return {
        "ok": report.ok,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "symbol_count": int(df["symbol"].nunique(dropna=True)) if "symbol" in df.columns else 0,
    }


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    data = await read_upload_bytes(file)
    df = load_csv_bytes(data, limits=IngestLimits())
    report = validate_dataset(df)
    if not report.ok:
        raise AppError(
            code="dataset_invalid",
            message="Dataset failed validation.",
            details={"errors": list(report.errors), "warnings": list(report.warnings)},
        )

    meta = save_dataset(df)
    return {"ok": True, "dataset": meta.__dict__, "warnings": list(report.warnings)}


@router.get("")
def list_datasets() -> dict:
    return {"dataset_ids": list_dataset_ids()}


@router.get("/{dataset_id}")
def get_dataset_meta(dataset_id: str) -> dict:
    meta = load_meta(dataset_id)
    return {"dataset": meta.__dict__}

