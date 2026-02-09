from __future__ import annotations

from fastapi import UploadFile
from starlette import status

from ..core.config import settings
from ..core.errors import AppError


ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({"text/csv"})


async def read_upload_bytes(file: UploadFile) -> bytes:
    if not file.filename:
        raise AppError(code="missing_filename", message="Missing uploaded filename.")

    # Basic filename allowlist: only .csv uploads supported
    if not str(file.filename).lower().endswith(".csv"):
        raise AppError(
            code="unsupported_file_extension",
            message="Only .csv uploads are supported.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if not file.content_type or file.content_type not in ALLOWED_CONTENT_TYPES:
        raise AppError(
            code="unsupported_media_type",
            message="Unsupported file type.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            code="file_too_large",
            message="File too large.",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={"max_upload_bytes": settings.max_upload_bytes},
        )

    if not data:
        raise AppError(code="empty_upload", message="Uploaded file is empty.")

    return data

