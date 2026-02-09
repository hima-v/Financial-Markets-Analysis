from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from .upload import read_upload_bytes
from ..core.errors import AppError
from ..services.analytics import AnalyticsLimits, correlation_matrix, drawdown, filter_date_range, movers, returns_series
from ..services.ingest import IngestLimits, load_csv_bytes
from ..services.storage import load_dataset


router = APIRouter(prefix="/analytics", tags=["analytics"])

_MAX_POINTS_MIN = 200
_MAX_POINTS_MAX = 10_000
_MAX_SYMBOLS_MIN = 2
_MAX_SYMBOLS_MAX = 60


def _bounded_int(name: str, value: int, *, min_value: int, max_value: int) -> int:
    if value < min_value or value > max_value:
        raise AppError(
            code="invalid_parameter",
            message="Invalid parameter value.",
            details={"name": name, "min": min_value, "max": max_value, "value": value},
        )
    return value


async def _load_frame(*, dataset_id: str | None, file: UploadFile | None) -> pd.DataFrame:
    if bool(dataset_id) == bool(file):
        raise AppError(
            code="invalid_input",
            message="Provide exactly one of dataset_id or file.",
        )

    if dataset_id:
        return load_dataset(dataset_id)

    assert file is not None
    data = await read_upload_bytes(file)
    return load_csv_bytes(data, limits=IngestLimits())


@router.post("/movers")
async def movers_endpoint(
    dataset_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(None),
    top_n: Annotated[int, Form()] = 8,
    start: Annotated[str | None, Form()] = None,
    end: Annotated[str | None, Form()] = None,
) -> dict:
    df = await _load_frame(dataset_id=dataset_id, file=file)
    df = filter_date_range(df, start=start, end=end)

    gainers, losers = movers(df, top_n=top_n)
    return {
        "top_n": int(top_n),
        "gainers": gainers.to_dict(orient="records"),
        "losers": losers.to_dict(orient="records"),
    }


@router.post("/returns")
async def returns_endpoint(
    dataset_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(None),
    symbol: Annotated[str, Form()] = "",
    start: Annotated[str | None, Form()] = None,
    end: Annotated[str | None, Form()] = None,
    max_points: Annotated[int, Form()] = 1500,
) -> dict:
    df = await _load_frame(dataset_id=dataset_id, file=file)
    df = filter_date_range(df, start=start, end=end)

    mp = _bounded_int("max_points", int(max_points), min_value=_MAX_POINTS_MIN, max_value=_MAX_POINTS_MAX)
    limits = AnalyticsLimits(max_points=mp)
    series = returns_series(df, symbol=symbol, limits=limits)
    series = series.copy()
    series["date"] = series["date"].dt.strftime("%Y-%m-%d")
    return {"symbol": symbol, "points": series.to_dict(orient="records")}


@router.post("/drawdown")
async def drawdown_endpoint(
    dataset_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(None),
    symbol: Annotated[str, Form()] = "",
    start: Annotated[str | None, Form()] = None,
    end: Annotated[str | None, Form()] = None,
    max_points: Annotated[int, Form()] = 1500,
) -> dict:
    df = await _load_frame(dataset_id=dataset_id, file=file)
    df = filter_date_range(df, start=start, end=end)

    mp = _bounded_int("max_points", int(max_points), min_value=_MAX_POINTS_MIN, max_value=_MAX_POINTS_MAX)
    limits = AnalyticsLimits(max_points=mp)
    dd = drawdown(df, symbol=symbol, limits=limits)
    dd = dd.copy()
    dd["date"] = pd.to_datetime(dd["date"]).dt.strftime("%Y-%m-%d")
    return {"symbol": symbol, "points": dd.to_dict(orient="records")}


@router.post("/correlation")
async def correlation_endpoint(
    dataset_id: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(None),
    symbols: Annotated[str | None, Form()] = None,
    start: Annotated[str | None, Form()] = None,
    end: Annotated[str | None, Form()] = None,
    max_symbols: Annotated[int, Form()] = 30,
) -> dict:
    df = await _load_frame(dataset_id=dataset_id, file=file)
    df = filter_date_range(df, start=start, end=end)

    ms = _bounded_int("max_symbols", int(max_symbols), min_value=_MAX_SYMBOLS_MIN, max_value=_MAX_SYMBOLS_MAX)
    parsed_symbols = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    limits = AnalyticsLimits(max_symbols=ms)
    labels, matrix = correlation_matrix(df, symbols=parsed_symbols, limits=limits)
    return {"labels": labels, "matrix": matrix}

