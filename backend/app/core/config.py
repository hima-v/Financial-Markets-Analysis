from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FMA_", case_sensitive=False)

    app_name: str = "financial-markets-api"
    environment: str = Field(default="dev", pattern="^(dev|test|prod)$")
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=500 * 1024 * 1024)
    data_dir: str = str(Path("data").resolve())
    max_saved_datasets: int = Field(default=50, ge=1, le=5000)
    artifacts_dir: str = str(Path("artifacts").resolve())
    max_inference_points: int = Field(default=1, ge=1, le=50)

    enable_dev_cors: bool = False
    dev_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


settings = Settings()

