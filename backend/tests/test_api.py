"""API tests: health, validation, prediction persistence, dashboard, monitoring, lab and datasets."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, select

from ml import contract as C


def test_health_and_db_init(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"] is True
    assert body["consistency_check"]["passed"] is True
    from app import db

    tables = set(inspect(db.engine()).get_table_names())
    assert {"datasets", "model_versions", "training_runs", "predictions", "request_logs", "monitoring_events", "experiment_runs"} <= tables


def test_schema_matches_contract(client):
    body = client.get("/api/schema").json()
    assert [f["name"] for f in body["request_fields"]] == list(C.FEATURE_NAMES)
    assert set(body["excluded_columns"]) == set(C.EXCLUDED_COLUMNS)
    from app.schemas import StudentFeatures

    assert list(StudentFeatures.model_fields) == list(C.FEATURE_NAMES)


def test_predict_output_schema_and_persistence(client, student):
    r = client.post("/api/predict", json=student)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"request_id", "created_at", "model_version", "prediction", "label", "probability", "threshold", "latency_ms", "timings", "imputed_fields", "disclaimer"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert body["label"] == C.CLASS_LABELS[body["prediction"]]
    assert body["created_at"].endswith("Z")
    trace = client.get(f"/api/predictions/{body['request_id']}").json()
    assert trace["inputs"]["CGPA"] == student["CGPA"]
    assert len(trace["feature_vector"]) == len(trace["feature_names"]) > 0
    assert trace["source"] == "online"


def test_predict_rejects_schema_mismatches(client, student):
    cases = [
        {**student, "City": "Bengaluru"},  # category not in contract
        {**student, "CGPA": 10.5},  # out of range
        {**student, "Internships": 1.5},  # non-integer count
        {**{k: v for k, v in student.items() if k != "Stream"}},  # required field missing
        {**student, "cgpa": 8.0},  # renamed / unknown field (training-serving mismatch)
        {**student, "CGPA": "eight"},  # wrong type
    ]
    for body in cases:
        r = client.post("/api/predict", json=body)
        assert r.status_code == 422, body


def test_optional_nulls_are_imputed(client, student):
    r = client.post("/api/predict", json={**student, "CodingTestScore": None, "Workshops": None})
    assert r.status_code == 200
    assert {"CodingTestScore", "Workshops"} <= set(r.json()["imputed_fields"])


def test_batch_json_and_limits(client, student):
    r = client.post("/api/predict/batch", json={"records": [student] * 5})
    assert r.status_code == 200 and r.json()["count"] == 5
    assert client.post("/api/predict/batch", json={"records": []}).status_code == 422
    assert client.post("/api/predict/batch", json={"records": [student] * 1001}).status_code == 422


def test_batch_csv_reports_invalid_rows(client, sample_df):
    df = sample_df.head(6).copy()
    df.loc[df.index[2], "City"] = "Atlantis"
    df.loc[df.index[4], "AttendancePercent"] = 140
    r = client.post("/api/predict/batch/csv", files={"file": ("batch.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 4
    assert sorted(e["index"] for e in body["errors"]) == [2, 4]
    assert client.post("/api/predict/batch/csv", files={"file": ("x.txt", "a,b", "text/plain")}).status_code == 422


def test_outcome_feedback_and_live_performance(client, sample_df, student):
    ids = []
    for _, row in sample_df.head(12).iterrows():
        rec = {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k])) for k in C.FEATURE_NAMES}
        r = client.post("/api/predict", json=rec, headers={"X-Prediction-Source": "replay"})
        assert r.status_code == 200, r.text
        ids.append((r.json()["request_id"], int(row[C.TARGET])))
    for rid, y in ids:
        assert client.post(f"/api/predictions/{rid}/outcome", json={"actual_outcome": y}).status_code == 200
    assert client.post(f"/api/predictions/{ids[0][0]}/outcome", json={"actual_outcome": 3}).status_code == 422
    assert client.post("/api/predictions/doesnotexist/outcome", json={"actual_outcome": 1}).status_code == 404
    perf = client.get("/api/monitoring/performance").json()
    assert perf["labelled_active_model"] >= 12 and "live" in perf
    hist = client.get("/api/predictions?source=replay").json()
    assert hist["total"] >= 12


def test_dashboard_reflects_real_predictions(client, student):
    before = client.get("/api/dashboard").json()["predictions"]["total"]
    client.post("/api/predict", json=student)
    dash = client.get("/api/dashboard").json()
    assert dash["predictions"]["total"] == before + 1
    assert dash["model"]["version"] == client.get("/api/health").json()["model_version"]
    assert dash["recent_predictions"][0]["probability"] is not None
    assert 0 < dash["model"]["test_metrics"]["roc_auc"] <= 1


def test_monitoring_requests_drift_and_events(client, student):
    summary = client.get("/api/monitoring/summary?hours=1").json()
    assert summary["requests"]["total"] > 0
    assert summary["latency_ms"]["predict_requests"] > 0
    for _ in range(25):
        client.post("/api/predict", json=student)
    drift = client.get("/api/monitoring/drift").json()
    assert drift["status"] in ("stable", "investigate", "significant_shift")
    # 25 identical rows are a heavily shifted population, so drift must be detected.
    assert drift["flagged_features"]
    check = client.post("/api/monitoring/drift/check").json()
    events = client.get("/api/monitoring/events?kind=drift_check").json()["items"]
    assert events and events[0]["details"]["status"] == check["status"]


def test_lifecycle_reports_real_stage_status(client):
    body = client.get("/api/lifecycle").json()
    stages = {s["key"]: s for s in body["stages"]}
    assert list(stages) == ["ingestion", "validation", "features", "training", "evaluation", "registration", "serving", "monitoring", "retraining"]
    assert stages["serving"]["status"] == "complete"
    assert body["tooling"]["mlflow"]["enabled"] is False


def test_dataset_upload_validation_and_limits(client, sample_df, settings, monkeypatch):
    good = client.post("/api/datasets/upload", files={"file": ("ok.csv", sample_df.head(300).to_csv(index=False), "text/csv")})
    assert good.status_code == 201 and good.json()["validation_status"] != "failed"
    bad = client.post("/api/datasets/upload", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 201 and bad.json()["validation_status"] == "failed"
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"\x00\x01\x02", "text/csv")}).status_code == 422
    assert client.post("/api/datasets/upload", files={"file": ("x.json", "{}", "application/json")}).status_code == 422
    monkeypatch.setattr(settings, "max_upload_mb", 0.001)
    big = client.post("/api/datasets/upload", files={"file": ("big.csv", sample_df.head(300).to_csv(index=False), "text/csv")})
    assert big.status_code == 413
    stats = client.get("/api/datasets/1/stats").json()
    cg = next(c for c in stats["columns"] if c["name"] == "CGPA")
    assert cg["kind"] == "numeric" and cg["skewness"] is not None and cg["kurtosis"] is not None
    sample = client.get("/api/datasets/1/sample?n=2").json()
    assert len(sample["items"]) == 2 and sample["split"] == "test"


def test_errors_do_not_leak_internals(client, monkeypatch):
    from app.services import monitoring_service

    def boom(*a, **k):
        raise RuntimeError("secret path C:/internal/file.db")

    monkeypatch.setattr(monitoring_service, "events", boom)
    r = client.get("/api/monitoring/events")
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text and r.json()["detail"] == "Internal server error"


def test_cors_allows_frontend_origin_only(client):
    ok = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


# ----------------------------------------------------------------------------- laboratory over HTTP
@pytest.mark.parametrize(
    "path,params",
    [
        ("m2/linear-regression", {"iterations": 50}),
        ("m2/logistic", {"threshold": 0.4}),
        ("m3/decision-tree", {"max_depth": 3}),
        ("m4/kmeans", {"k": 3, "sample_size": 600}),
        ("m4/dbscan", {"sample_size": 600}),
        ("m4/hierarchical", {"sample_size": 150}),
        ("m5/cross-validation", {"sample_size": 800}),
        ("m5/calibration", {"model": "logistic_regression"}),
        ("m6/skew", {}),
        ("m6/drift-sandbox", {"segment": "tier3"}),
    ],
)
def test_lab_experiments(client, path, params):
    r = client.post(f"/api/lab/{path}", json=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["params"] == {**body["params"], **params}
    res = body["result"]
    assert "split" in res and "not used" in res["split"]["test_split"] if "split" in res else True
    if path == "m4/kmeans":
        assert len(res["clusters"]) == 3 and -1 <= res["silhouette"] <= 1
    if path == "m4/dbscan":
        assert res["n_core"] + res["n_border"] + res["n_noise"] == 600
    if path == "m5/cross-validation":
        assert res["stratified_kfold"]["positive_rate_std"] <= res["kfold"]["positive_rate_std"] + 1e-9
    if path == "m5/calibration":
        assert set(res["methods"]) == {"uncalibrated", "sigmoid", "isotonic"}
    if path == "m6/skew":
        assert res["golden_test"]["passed"] is True
        cat = next(s for s in res["scenarios"] if s["id"] == "category_mismatch")
        assert cat["caught_by_contract_validation"] is True


def test_lab_rejects_bad_params(client):
    assert client.post("/api/lab/m4/kmeans", json={"k": 50}).status_code == 422
    assert client.post("/api/lab/m4/kmeans", json={"unknown": 1}).status_code == 422
    assert client.post("/api/lab/m9/nothing", json={}).status_code == 404
    catalog = client.get("/api/lab/catalog").json()["items"]
    assert {(e["module"]) for e in catalog} == {"m2", "m3", "m4", "m5", "m6"}
