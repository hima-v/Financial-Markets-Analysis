from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

from .api.router import api_router
from .core.config import settings
from .core.errors import app_error_handler, json_error, AppError


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    if settings.enable_dev_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.dev_cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    return app


async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return json_error(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="validation_error",
        message="Request validation failed.",
        details={"errors": exc.errors()},
    )


async def _unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
    return json_error(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message="Unexpected server error.",
    )


app = create_app()

