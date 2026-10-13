"""Retraining workflow: promotion, rejection, failure, rollback and concurrency safety."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import select

from .conftest import FAST_CONFIG

# The public API only accepts the documented TrainingConfigIn fields.
API_CONFIG = {"candidates": ["logistic_regression"], "cv_folds": 3, "min_validation_roc_auc": 0.6}


def _active(client) -> str:
    return client.get("/api/health").json()["model_version"]


def _wait(client, job_id: str, timeout: float = 120) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/training/jobs/{job_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_rejected_candidate_preserves_active_model(client):
    before = _active(client)
    r = client.post("/api/training/jobs", json={"config": {"candidates": ["logistic_regression"], "cv_folds": 3, "min_validation_roc_auc": 0.9999}})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "rejected" and job["decision"] == "rejected"
    assert "Quality gate" in job["reason"]
    assert _active(client) == before
    events = client.get("/api/monitoring/events?kind=retrain_rejected").json()["items"]
    assert events and before in events[0]["message"]


def test_failed_dataset_cannot_start_training(client):
    up = client.post("/api/datasets/upload", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")}).json()
    r = client.post("/api/training/jobs", json={"dataset_id": up["id"]})
    assert r.status_code == 422
    assert client.post("/api/training/jobs", json={"dataset_id": 99999}).status_code == 404


def test_failure_during_training_keeps_previous_model(client, settings, sample_df, monkeypatch):
    """A dataset that passed upload validation but breaks during training must not affect serving."""
    from app.services import training_service

    before = _active(client)

    def broken_train(*a, **k):
        raise RuntimeError("simulated crash inside training")

    monkeypatch.setattr(training_service, "train_model", broken_train)
    r = client.post("/api/training/jobs", json={"config": API_CONFIG})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "failed"
    assert "Unexpected RuntimeError" in job["reason"]
    assert _active(client) == before
    assert client.post("/api/predict", json=_student(sample_df)).status_code == 200


def test_promotion_failure_rolls_back(client, settings, sample_df, monkeypatch):
    """If loading the new model fails after it was registered, the registry and serving roll back."""
    from app.services import training_service
    from app.services.model_service import model_service

    before = _active(client)
    versions_before = {m["version"] for m in client.get("/api/models").json()["items"]}
    real_load = model_service.load_active
    calls = {"n": 0}

    def flaky_load(session):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated load failure")
        return real_load(session)

    monkeypatch.setattr(model_service, "load_active", flaky_load)
    from ml.training import TrainConfig

    job_id = training_service.run_job_blocking(settings, TrainConfig.from_dict({**FAST_CONFIG, "min_validation_roc_auc": 0.5}), trigger="cli")
    monkeypatch.setattr(model_service, "load_active", real_load)
    job = client.get(f"/api/training/jobs/{job_id}").json()
    assert job["status"] == "failed" and "rolled back" in job["reason"]
    assert _active(client) == before
    versions_after = {m["version"] for m in client.get("/api/models").json()["items"]}
    assert versions_after == versions_before
    assert not any(p.name not in versions_before and not p.name.startswith(".") for p in Path(settings.models_dir).iterdir())


def test_concurrent_job_is_refused(client):
    from app.services import training_service

    assert training_service._job_lock.acquire(blocking=False)
    try:
        r = client.post("/api/training/jobs", json={"config": API_CONFIG})
        assert r.status_code == 409
    finally:
        training_service._job_lock.release()


def test_successful_retrain_promotes_and_rollback_works(client, sample_df):
    before = _active(client)
    r = client.post("/api/training/jobs", json={"config": {**API_CONFIG, "min_validation_roc_auc": 0.5}})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    new = job["new_version"]
    assert new != before and _active(client) == new
    models = {m["version"]: m["status"] for m in client.get("/api/models").json()["items"]}
    assert models[new] == "active" and models[before] == "archived"
    detail = client.get(f"/api/models/{new}").json()
    assert detail["metadata"]["splits"]["test"]["rows"] > 0 and "golden_examples" not in detail["metadata"]
    # Manual rollback to the previous version.
    assert client.post(f"/api/models/{before}/activate").status_code == 200
    assert _active(client) == before
    assert client.post("/api/predict", json=_student(sample_df)).json()["model_version"] == before
    assert client.post("/api/models/v999/activate").status_code == 404


def _student(sample_df) -> dict:
    import pandas as pd

    from ml import contract as C

    row = sample_df.iloc[1]
    return {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k])) for k in C.FEATURE_NAMES}
