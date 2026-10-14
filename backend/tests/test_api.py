"""API tests: health, validation, single/batch prediction, persistence, history, dashboard, monitoring,
evaluation explorer, lab and datasets."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect

from ml import contract as C

from .conftest import to_record


def test_health_and_db_init(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"] is True
    assert body["consistency_check"]["passed"] is True and body["consistency_check"]["decision_mismatches"] == 0
    assert set(body["policy"]) >= {"fusion_weight_lr", "review_threshold", "high_risk_threshold"}
    from app import db

    tables = set(inspect(db.engine()).get_table_names())
    assert {"app_meta", "datasets", "model_versions", "training_runs", "predictions", "request_logs", "monitoring_events", "experiment_runs"} <= tables


def test_incompatible_placement_database_is_refused(tmp_path):
    """Pointing the app at a database created by the old PlacementPredict schema fails loudly and leaves it untouched."""
    import sqlite3

    from app import db

    old = tmp_path / "placementpredict.db"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, probability FLOAT, prediction INTEGER)")
    con.commit()
    con.close()
    before = old.read_bytes()
    from app.config import get_settings

    try:
        with pytest.raises(db.IncompatibleDatabase):
            db.configure(f"sqlite:///{old.as_posix()}")
    finally:
        db.configure(get_settings().resolved_database_url)
    assert old.read_bytes() == before


def test_schema_matches_contract(client):
    body = client.get("/api/schema").json()
    assert [f["name"] for f in body["request_fields"]] == list(C.FEATURE_NAMES)
    assert body["decisions"] == ["LOW_RISK", "REVIEW", "HIGH_RISK"]
    from app.schemas import TransactionFeatures

    assert [n for n in TransactionFeatures.model_fields if n != C.TRANSACTION_ID] == list(C.FEATURE_NAMES)


def test_single_prediction_schema_and_persistence(client, transaction):
    r = client.post("/api/predict", json={**transaction, "transaction_id": "tx-001"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in C.DECISIONS and body["transaction_id"] == "tx-001"
    s, p = body["scores"], body["policy"]
    assert 0 <= s["lr_probability"] <= 1 and 0 <= s["anomaly_percentile"] <= 1 and 0 <= s["hybrid_score"] <= 1
    assert s["hybrid_score"] == pytest.approx(p["fusion_weight_lr"] * s["lr_rarity"] + p["fusion_weight_if"] * s["if_rarity"], abs=1e-7)
    expected = "HIGH_RISK" if s["hybrid_score"] >= p["high_risk_threshold"] else ("REVIEW" if s["hybrid_score"] >= p["review_threshold"] else "LOW_RISK")
    assert body["decision"] == expected
    assert len(body["explanation"]["lr_top_contributions"]) == 6
    assert body["created_at"].endswith("Z") and body["disclaimer"]
    trace = client.get(f"/api/predictions/{body['request_id']}").json()
    assert trace["inputs"]["Amount"] == transaction["Amount"] and trace["transaction_id"] == "tx-001"
    assert len(trace["feature_vector"]) == len(trace["feature_names"]) == 31
    assert trace["source"] == "online" and trace["policy"] == p


def test_fraud_row_scores_higher_than_legitimate(client, transaction, fraud_transaction):
    legit = client.post("/api/predict", json=transaction).json()
    fraud = client.post("/api/predict", json=fraud_transaction).json()
    assert fraud["scores"]["lr_probability"] > legit["scores"]["lr_probability"]


@pytest.mark.parametrize(
    "change",
    [
        {"Amount": -1.0},  # negative amount
        {"Time": -10.0},  # negative time
        {"V3": "abc"},  # wrong type
        {"cgpa": 8.0},  # unknown field (training-serving mismatch)
        {"transaction_id": "bad id with spaces"},
        {"transaction_id": "x" * 65},
    ],
)
def test_predict_rejects_invalid_input(client, transaction, change):
    assert client.post("/api/predict", json={**transaction, **change}).status_code == 422


def test_predict_rejects_missing_and_non_finite(client, transaction):
    body = {k: v for k, v in transaction.items() if k != "V14"}
    assert client.post("/api/predict", json=body).status_code == 422
    raw = '{"Time": 1, "Amount": 1, ' + ", ".join(f'"V{i}": {"Infinity" if i == 2 else 0.1}' for i in range(1, 29)) + "}"
    assert client.post("/api/predict", content=raw, headers={"Content-Type": "application/json"}).status_code == 422


def test_batch_json_and_limits(client, transaction):
    r = client.post("/api/predict/batch", json={"records": [transaction] * 5})
    assert r.status_code == 200 and r.json()["count"] == 5 and sum(r.json()["decision_counts"].values()) == 5
    assert client.post("/api/predict/batch", json={"records": []}).status_code == 422
    assert client.post("/api/predict/batch", json={"records": [transaction] * 1001}).status_code == 422


def test_batch_csv_reports_invalid_rows_and_ignores_extra_columns(client, sample_df):
    df = sample_df.head(8).copy()
    df["transaction_id"] = [f"csv-{i}" for i in range(8)]
    df.loc[df.index[2], "Amount"] = -3
    df["V5"] = df["V5"].astype(object)
    df.loc[df.index[5], "V5"] = "oops"
    r = client.post("/api/predict/batch/csv", files={"file": ("batch.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 6 and sorted(e["index"] for e in body["errors"]) == [2, 5]
    assert {x["transaction_id"] for x in body["results"]} == {f"csv-{i}" for i in (0, 1, 3, 4, 6, 7)}


@pytest.mark.parametrize(
    "name,content,status",
    [
        ("x.txt", "a,b", 422),  # wrong extension
        ("empty.csv", "", 422),
        ("binary.csv", b"\x00\x01\x02", 422),
        ("cols.csv", "Time,Amount\n1,2\n", 422),  # missing V columns
    ],
)
def test_malformed_csv_is_rejected(client, name, content, status):
    assert client.post("/api/predict/batch/csv", files={"file": (name, content, "text/csv")}).status_code == status


def test_history_filters_search_and_pagination(client, sample_df):
    for i, (_, row) in enumerate(sample_df.head(6).iterrows()):
        client.post("/api/predict", json=to_record(row, transaction_id=f"hist-{i}"), headers={"X-Prediction-Source": "replay"})
    found = client.get("/api/predictions", params={"q": "hist-"}).json()
    assert found["total"] == 6
    page = client.get("/api/predictions", params={"q": "hist-", "limit": 4, "offset": 4}).json()
    assert len(page["items"]) == 2
    by_score = client.get("/api/predictions", params={"q": "hist-", "sort": "score_desc"}).json()["items"]
    assert [x["hybrid_score"] for x in by_score] == sorted((x["hybrid_score"] for x in by_score), reverse=True)
    low = client.get("/api/predictions", params={"q": "hist-", "decision": "LOW_RISK"}).json()
    assert all(x["decision"] == "LOW_RISK" for x in low["items"])
    assert sum(found["decision_counts"].values()) == 6
    amt = client.get("/api/predictions", params={"q": "hist-", "min_amount": 50}).json()
    assert all(x["amount"] >= 50 for x in amt["items"])
    assert client.get("/api/predictions", params={"sort": "bogus"}).status_code == 422
    assert client.get("/api/predictions", params={"source": "replay"}).json()["total"] >= 6


def test_outcome_feedback_and_live_performance(client, sample_df):
    ids = []
    for _, row in sample_df.sample(14, random_state=3).iterrows():
        r = client.post("/api/predict", json=to_record(row), headers={"X-Prediction-Source": "replay"})
        assert r.status_code == 200, r.text
        ids.append((r.json()["request_id"], int(row[C.TARGET])))
    for rid, y in ids:
        res = client.post(f"/api/predictions/{rid}/outcome", json={"actual_outcome": y})
        assert res.status_code == 200 and res.json()["actual_outcome"] == y
    assert client.post(f"/api/predictions/{ids[0][0]}/outcome", json={"actual_outcome": 3}).status_code == 422
    assert client.post("/api/predictions/doesnotexist/outcome", json={"actual_outcome": 1}).status_code == 404
    perf = client.get("/api/monitoring/performance").json()
    assert perf["labelled_active_model"] >= 14 and "live" in perf
    assert client.get("/api/predictions", params={"labelled": True}).json()["total"] >= 14


def test_dashboard_reflects_real_predictions(client, transaction):
    before = client.get("/api/dashboard").json()["predictions"]["total"]
    client.post("/api/predict", json=transaction)
    dash = client.get("/api/dashboard").json()
    assert dash["predictions"]["total"] == before + 1
    assert sum(dash["predictions"]["by_decision"].values()) == before + 1
    assert dash["model"]["version"] == client.get("/api/health").json()["model_version"]
    assert set(dash["model"]["test_metrics"]) == {"logistic_regression", "isolation_forest", "hybrid_high_risk", "hybrid_review_or_high"}


def test_monitoring_requests_drift_and_events(client, transaction):
    summary = client.get("/api/monitoring/summary?hours=1").json()
    assert summary["requests"]["total"] > 0 and summary["latency_ms"]["predict_requests"] > 0
    for _ in range(25):
        client.post("/api/predict", json=transaction)
    drift = client.get("/api/monitoring/drift").json()
    assert drift["status"] in ("stable", "investigate", "significant_shift")
    # 25 identical rows are a heavily shifted population, so drift must be detected.
    assert drift["flagged_features"]
    assert set(drift["scores"]) == {"lr_probability", "anomaly_percentile", "hybrid_score"}
    check = client.post("/api/monitoring/drift/check").json()
    events = client.get("/api/monitoring/events?kind=drift_check").json()["items"]
    assert events and events[0]["details"]["status"] == check["status"]


def test_evaluation_explorer_uses_validation_only(client):
    summary = client.get("/api/evaluation/summary").json()
    p = summary["policy"]
    body = {
        "weight": p["fusion_weight_lr"],
        "review_threshold": p["review_threshold"],
        "high_risk_threshold": p["high_risk_threshold"],
        "lr_threshold": p["lr_threshold"],
        "if_flag_rate": round(1 - p["if_flag_percentile"], 10),
    }
    res = client.post("/api/evaluation/explore", json=body).json()
    assert res["split"] == "validation"
    stored = summary["metrics"]["validation"]
    # Replaying the active policy reproduces the stored validation evaluation exactly.
    for k in ("logistic_regression", "hybrid_high_risk", "hybrid_review_or_high"):
        assert res["metrics"][k]["confusion_matrix"] == stored[k]["confusion_matrix"], k
    assert res["metrics"]["decisions"] == stored["decisions"]
    assert client.post("/api/evaluation/explore", json={**body, "review_threshold": 0.9, "high_risk_threshold": 0.1}).status_code == 422
    other = client.post("/api/evaluation/explore", json={**body, "weight": 0.5}).json()
    assert other["metrics"]["hybrid_high_risk"]["average_precision"] != res["metrics"]["hybrid_high_risk"]["average_precision"]


def test_lifecycle_reports_real_stage_status(client):
    body = client.get("/api/lifecycle").json()
    stages = {s["key"]: s for s in body["stages"]}
    assert list(stages) == ["ingestion", "validation", "features", "training", "evaluation", "registration", "serving", "monitoring", "retraining"]
    assert stages["serving"]["status"] == "complete"
    assert body["tooling"]["mlflow"]["enabled"] is False and body["tooling"]["mlflow"]["experiment"] == "fraud-detection"


def test_dataset_upload_validation_sample_and_limits(client, sample_df, settings, monkeypatch):
    good = client.post("/api/datasets/upload", files={"file": ("ok.csv", sample_df.to_csv(index=False), "text/csv")})
    assert good.status_code == 201 and good.json()["validation_status"] != "failed"
    bad = client.post("/api/datasets/upload", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 201 and bad.json()["validation_status"] == "failed"
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"\x00\x01\x02", "text/csv")}).status_code == 422
    assert client.post("/api/datasets/upload", files={"file": ("x.json", "{}", "application/json")}).status_code == 422
    monkeypatch.setattr(settings, "max_upload_mb", 0.001)
    big = client.post("/api/datasets/upload", files={"file": ("big.csv", sample_df.head(300).to_csv(index=False), "text/csv")})
    assert big.status_code == 413
    stats = client.get("/api/datasets/1/stats").json()
    amount = next(c for c in stats["columns"] if c["name"] == "Amount")
    assert amount["kind"] == "numeric" and amount["skewness"] > 0 and amount["mean_fraud"] is not None
    fraud = client.get("/api/datasets/1/sample", params={"n": 3, "label": 1}).json()
    assert fraud["split"] == "test" and len(fraud["items"]) == 3 and all(i["actual_class"] == 1 for i in fraud["items"])


def test_sampled_rows_really_come_from_the_test_split(client, trained):
    """The loader replays held-out rows of the active model's split, never training rows."""
    items = client.get("/api/datasets/1/sample", params={"n": 20, "seed": 1}).json()["items"]
    train_keys = set(map(tuple, trained.splits.train[list(C.FEATURE_NAMES)].round(8).to_numpy()))
    for it in items:
        assert tuple(round(it["features"][n], 8) for n in C.FEATURE_NAMES) not in train_keys


def test_model_integrity_endpoint(client):
    v = client.get("/api/health").json()["model_version"]
    res = client.get(f"/api/models/{v}/integrity").json()
    assert res["matches"] is True and res["actual_sha256"] == res["expected_sha256"]
    assert client.get("/api/models/v999/integrity").status_code == 404


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
        ("m4/anomaly", {}),
        ("m5/cross-validation", {"sample_size": 2000}),
        ("m5/metrics", {"model": "logistic_regression"}),
        ("m6/skew", {}),
        ("m6/drift-sandbox", {"segment": "high_amount"}),
    ],
)
def test_lab_experiments(client, path, params):
    r = client.post(f"/api/lab/{path}", json=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["params"] == {**body["params"], **params}
    res = body["result"]
    if "split" in res:
        assert "not used" in res["split"]["test_split"]
    if path == "m4/kmeans":
        assert len(res["clusters"]) == 3 and -1 <= res["silhouette"] <= 1
    if path == "m4/dbscan":
        assert res["n_core"] + res["n_border"] + res["n_noise"] == 600
    if path == "m5/cross-validation":
        assert res["stratified_kfold"]["fraud_count_std"] <= res["kfold"]["fraud_count_std"] + 1e-9
    if path == "m5/metrics":
        assert res["accuracy_baseline"] > 0.9  # the accuracy paradox is reported
    if path == "m6/skew":
        assert res["golden_test"]["passed"] is True
        assert {s["id"] for s in res["scenarios"]} == {"scaler_refit", "amount_cents", "time_hours", "column_swap", "double_log"}


def test_lab_rejects_bad_params(client):
    assert client.post("/api/lab/m4/kmeans", json={"k": 50}).status_code == 422
    assert client.post("/api/lab/m4/kmeans", json={"unknown": 1}).status_code == 422
    assert client.post("/api/lab/m9/nothing", json={}).status_code == 404
    catalog = client.get("/api/lab/catalog").json()["items"]
    assert {e["module"] for e in catalog} == {"m2", "m3", "m4", "m5", "m6"} and len(catalog) == 27


def test_sample_frame_is_real(sample_df):
    assert isinstance(sample_df, pd.DataFrame) and int(sample_df[C.TARGET].sum()) == 150
