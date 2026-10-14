"""Runtime configuration. Every value can be overridden with an ``FG_``-prefixed environment variable."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FG_", env_file=".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    database_url: str = ""
    models_dir: Path = PROJECT_ROOT / "artifacts" / "fraud_models"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    default_dataset: Path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    reports_dir: Path = PROJECT_ROOT / "reports"

    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_artifact_root: Path = PROJECT_ROOT / "mlruns"

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    max_upload_mb: float = 200.0  # dataset uploads (the full creditcard.csv is ~144 MB)
    max_batch_csv_mb: float = 10.0
    max_batch_records: int = 1000
    max_batch_csv_rows: int = 5000
    drift_window: int = 1000
    min_drift_rows: int = 50
    min_labelled_for_performance: int = 30
    promotion_tolerance: float = 0.01  # validation average-precision tolerance when comparing with the active model
    lab_cache_size: int = 64
    lab_max_concurrency: int = 2

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.project_root / 'backend' / 'fraudguard.db').as_posix()}"

    @property
    def resolved_mlflow_uri(self) -> str:
        return self.mlflow_tracking_uri or f"sqlite:///{(self.project_root / 'mlflow.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
