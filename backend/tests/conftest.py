"""Test fixtures: an isolated database, model registry and a sample of the real transactions dataset.

Environment variables are set before the application is imported so the cached settings point at
temporary locations and the developer's database/registry are never touched.

The sample keeps 150 real fraud rows and 6,000 real legitimate rows (fraud over-represented so the
small validation/test splits still contain positives). No synthetic rows are created.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

DATASET = PROJECT / "data" / "raw" / "creditcard.csv"
if not DATASET.exists():  # pragma: no cover - environment guard
    pytest.exit(f"Tests need the real dataset at {DATASET} (see README: Dataset setup)", returncode=2)

_TMP = Path(tempfile.mkdtemp(prefix="fg-tests-"))
SAMPLE_CSV = _TMP / "creditcard_sample.csv"

_full = pd.read_csv(DATASET)
_sample = pd.concat(
    [_full[_full["Class"] == 1].sample(150, random_state=0), _full[_full["Class"] == 0].sample(6000, random_state=0)]
).sort_values("Time")
_sample.to_csv(SAMPLE_CSV, index=False)
del _full

os.environ.update(
    {
        "FG_DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "FG_MODELS_DIR": str(_TMP / "models"),
        "FG_UPLOADS_DIR": str(_TMP / "uploads"),
        "FG_REPORTS_DIR": str(_TMP / "reports"),
        "FG_DEFAULT_DATASET": str(SAMPLE_CSV),
        "FG_MLFLOW_ENABLED": "false",
        "FG_MIN_DRIFT_ROWS": "20",
        "FG_MIN_LABELLED_FOR_PERFORMANCE": "10",
    }
)

FAST_CONFIG = {
    "lr_c_grid": [0.1, 1.0],
    "cv_folds": 3,
    "if_n_estimators": 100,
    "golden_examples": 10,
    "permutation_repeats": 1,
    "min_validation_average_precision": 0.2,
}


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="session")
def settings():
    from app.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def trained(sample_df):
    """One trained model shared by the ML unit tests."""
    from ml.training import TrainConfig, train_model

    return train_model(sample_df, TrainConfig.from_dict(FAST_CONFIG))


@pytest.fixture(scope="session")
def client(settings):
    """App with a trained, registered and activated v1 model."""
    from fastapi.testclient import TestClient

    from app import db
    from app.main import app, startup
    from app.services.model_service import model_service
    from app.services.training_service import run_job_blocking
    from ml.training import TrainConfig

    startup()
    run_job_blocking(settings, TrainConfig.from_dict(FAST_CONFIG), trigger="cli")
    with db.session_scope() as s:
        model_service.load_active(s)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def to_record(row: pd.Series, **extra) -> dict:
    from ml import contract as C

    return {**{name: float(row[name]) for name in C.FEATURE_NAMES}, **extra}


@pytest.fixture()
def transaction(sample_df) -> dict:
    """A valid request body built from a real (legitimate) dataset row."""
    return to_record(sample_df[sample_df["Class"] == 0].iloc[0])


@pytest.fixture()
def fraud_transaction(sample_df) -> dict:
    return to_record(sample_df[sample_df["Class"] == 1].iloc[0])
