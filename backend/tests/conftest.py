"""Test fixtures: an isolated database, model registry and a stratified sample of the real dataset.

Environment variables are set before the application is imported so the cached settings point at
temporary locations and the developer's database/registry are never touched.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

_TMP = Path(tempfile.mkdtemp(prefix="pp-tests-"))
SAMPLE_CSV = _TMP / "placement_sample.csv"

_full = pd.read_csv(PROJECT / "data" / "raw" / "placement_50k.csv")
_sample, _ = train_test_split(_full, train_size=3000, stratify=_full["PlacementStatus"], random_state=0)
_sample.to_csv(SAMPLE_CSV, index=False)

os.environ.update(
    {
        "PP_DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "PP_MODELS_DIR": str(_TMP / "models"),
        "PP_UPLOADS_DIR": str(_TMP / "uploads"),
        "PP_REPORTS_DIR": str(_TMP / "reports"),
        "PP_DEFAULT_DATASET": str(SAMPLE_CSV),
        "PP_MLFLOW_ENABLED": "false",
        "PP_MIN_DRIFT_ROWS": "20",
        "PP_MIN_LABELLED_FOR_PERFORMANCE": "10",
    }
)

FAST_CONFIG = {
    "candidates": ["logistic_regression"],
    "cv_folds": 3,
    "permutation_repeats": 2,
    "importance_rows": 400,
    "golden_examples": 10,
    "min_validation_roc_auc": 0.6,
}


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="session")
def settings():
    from app.config import get_settings

    return get_settings()


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


@pytest.fixture()
def student(sample_df) -> dict:
    """A valid request body built from a real dataset row."""
    from ml import contract as C

    row = sample_df.iloc[0]
    rec = {}
    for name in C.FEATURE_NAMES:
        f = C.FEATURES_BY_NAME[name]
        v = row[name]
        if pd.isna(v):
            rec[name] = None
        elif isinstance(f, C.NumericFeature):
            rec[name] = int(v) if f.integer else float(v)
        else:
            rec[name] = str(v)
    return rec
