"""Retraining workflow: promotion, rejection, failure, rollback, candidate registration and concurrency safety."""
from __future__ import annotations

import time
from pathlib import Path

from .conftest import FAST_CONFIG, to_record

# The public API accepts the documented TrainingConfigIn fields.
API_CONFIG = {"lr_c_grid": [1.0], "cv_folds": 2, "if_n_estimators": 50, "min_validation_average_precision": 0.2}


def _active(client) -> str:
    return client.get("/api/health").json()["model_version"]


def _wait(client, job_id: str, timeout: float = 180) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/training/jobs/{job_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def _row(sample_df) -> dict:
    return to_record(sample_df.iloc[1])


def test_rejected_candidate_preserves_active_model(client):
    before = _active(client)
    r = client.post("/api/training/jobs", json={"config": {**API_CONFIG, "min_validation_average_precision": 0.9999}})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "rejected" and job["decision"] == "rejected"
    assert "Quality gate" in job["reason"] and "PR-AUC" in job["reason"]
    assert _active(client) == before
    events = client.get("/api/monitoring/events?kind=retrain_rejected").json()["items"]
    assert events and before in events[0]["message"]


def test_invalid_training_config_is_rejected(client):
    assert client.post("/api/training/jobs", json={"config": {"lr_imbalance": "smote"}}).status_code == 422
    assert client.post("/api/training/jobs", json={"config": {"cv_folds": 1}}).status_code == 422
    assert client.post("/api/training/jobs", json={"config": {"nope": 1}}).status_code == 422


def test_failed_dataset_cannot_start_training(client):
    up = client.post("/api/datasets/upload", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")}).json()
    assert client.post("/api/training/jobs", json={"dataset_id": up["id"]}).status_code == 422
    assert client.post("/api/training/jobs", json={"dataset_id": 99999}).status_code == 404


def test_failure_during_training_keeps_previous_model(client, sample_df, monkeypatch):
    from app.services import training_service

    before = _active(client)

    def broken_train(*a, **k):
        raise RuntimeError("simulated crash inside training")

    monkeypatch.setattr(training_service, "train_model", broken_train)
    job = _wait(client, client.post("/api/training/jobs", json={"config": API_CONFIG}).json()["job_id"])
    assert job["status"] == "failed" and "Unexpected RuntimeError" in job["reason"]
    assert _active(client) == before
    assert client.post("/api/predict", json=_row(sample_df)).status_code == 200


def test_promotion_failure_rolls_back(client, settings, monkeypatch):
    """If loading the new model fails after it was registered, the registry and serving roll back."""
    from app.services import training_service
    from app.services.model_service import model_service
    from ml.training import TrainConfig

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
    job_id = training_service.run_job_blocking(settings, TrainConfig.from_dict({**FAST_CONFIG, "min_validation_average_precision": 0.0}), trigger="cli")
    monkeypatch.setattr(model_service, "load_active", real_load)
    job = client.get(f"/api/training/jobs/{job_id}").json()
    assert job["status"] == "failed" and "rolled back" in job["reason"]
    assert _active(client) == before
    assert {m["version"] for m in client.get("/api/models").json()["items"]} == versions_before
    assert not any(p.name not in versions_before and not p.name.startswith(".") for p in Path(settings.models_dir).iterdir())


def test_concurrent_job_is_refused(client):
    from app.services import training_service

    assert training_service._job_lock.acquire(blocking=False)
    try:
        assert client.post("/api/training/jobs", json={"config": API_CONFIG}).status_code == 409
    finally:
        training_service._job_lock.release()


def test_candidate_registration_and_manual_promotion(client, sample_df):
    before = _active(client)
    job = _wait(client, client.post("/api/training/jobs", json={"config": {**API_CONFIG, "min_validation_average_precision": 0.0, "auto_promote": False}}).json()["job_id"])
    assert job["status"] == "succeeded" and job["decision"] == "registered", job
    cand = job["new_version"]
    models = {m["version"]: m["status"] for m in client.get("/api/models").json()["items"]}
    assert models[cand] == "candidate" and _active(client) == before  # registered, not serving
    # Manual promotion verifies integrity + golden examples, then serves the candidate.
    assert client.post(f"/api/models/{cand}/activate").status_code == 200
    assert _active(client) == cand
    assert client.post("/api/predict", json=_row(sample_df)).json()["model_version"] == cand
    # Roll back to the previous version.
    assert client.post(f"/api/models/{before}/activate").status_code == 200
    assert _active(client) == before


def test_tampered_artifact_cannot_be_activated(client, settings):
    models = client.get("/api/models").json()["items"]
    target = next(m for m in models if m["status"] != "active")
    path = Path(settings.models_dir) / target["version"] / "model.joblib"
    original = path.read_bytes()
    before = _active(client)
    try:
        path.write_bytes(original + b"tampered")
        assert client.get(f"/api/models/{target['version']}/integrity").json()["matches"] is False
        r = client.post(f"/api/models/{target['version']}/activate")
        assert r.status_code == 422 and "previous model is still serving" in r.json()["detail"]
        assert _active(client) == before
    finally:
        path.write_bytes(original)


def test_successful_retrain_promotes_and_rollback_works(client, sample_df):
    before = _active(client)
    job = _wait(client, client.post("/api/training/jobs", json={"config": {**API_CONFIG, "min_validation_average_precision": 0.0}}).json()["job_id"])
    assert job["status"] == "succeeded" and job["decision"] == "promoted", job
    new = job["new_version"]
    assert new != before and _active(client) == new
    assert "active" in job["comparison"] and job["comparison"]["candidate_consistency_check"]["passed"]
    models = {m["version"]: m["status"] for m in client.get("/api/models").json()["items"]}
    assert models[new] == "active" and models[before] == "archived"
    detail = client.get(f"/api/models/{new}").json()
    assert detail["metadata"]["splits"]["test"]["rows"] > 0 and "golden_examples" not in detail["metadata"]
    assert detail["policy"]["high_risk_threshold"] == detail["threshold"]
    assert client.post(f"/api/models/{before}/activate").status_code == 200
    assert _active(client) == before
    assert client.post("/api/models/v999/activate").status_code == 404
