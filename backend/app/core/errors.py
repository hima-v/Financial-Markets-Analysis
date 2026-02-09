from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette import status


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = status.HTTP_400_BAD_REQUEST
    details: dict[str, Any] | None = None


def json_error(*, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    payload = ErrorResponse(code=code, message=message, details=details).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return json_error(status_code=exc.status_code, code=exc.code, message=exc.message, details=exc.details)

