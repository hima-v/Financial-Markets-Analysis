from __future__ import annotations

from fastapi import APIRouter

from .routes_analytics import router as analytics_router
from .routes_datasets import router as datasets_router
from .routes_health import router as health_router
from .routes_ml import router as ml_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(datasets_router)
api_router.include_router(analytics_router)
api_router.include_router(ml_router)

