"""Health, feature schema, dashboard and lifecycle endpoints."""
from __future__ import annotations

import importlib.metadata
import time
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ml import contract as C
from ml.estimators import available_classifiers
from ml.features import ENGINEERED_FEATURES

from .. import API_VERSION
from ..config import Settings, get_settings
from ..db import get_session, utcnow
from ..orm import Dataset, ExperimentRun, ModelVersion, MonitoringEvent, Prediction, RequestLog, TrainingRun
from ..schemas import HealthResponse
from ..services.model_service import model_service
from ..services.training_service import job_running

router = APIRouter(prefix="/api", tags=["system"])
STARTED = time.time()


@router.get("/health", response_model=HealthResponse, summary="Liveness, database and model status")
def health(session: Session = Depends(get_session, scope="function")) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        db = "ok"
    except Exception:
        db = "error"
    if db == "ok":
        model_service.sync(session)
    am = model_service.active
    return HealthResponse(
        status="ok" if (am is not None and db == "ok") else "degraded",
        api_version=API_VERSION,
        time=utcnow(),
        uptime_s=round(time.time() - STARTED, 1),
        database=db,
        model_loaded=am is not None,
        model_version=am.version if am else None,
        model_type=am.model_type if am else None,
        consistency_check=am.consistency if am else None,
        message=None if am else model_service.load_error,
    )


@router.get("/schema", summary="Feature contract and the active model's input schema")
def schema() -> dict[str, Any]:
    am = model_service.active
    feature_set = am.metadata["feature_set"] if am else "full"
    return {
        "request_fields": C.feature_schema("full"),
        "model_features": list(C.FEATURE_SETS[feature_set]),
        "feature_set": feature_set,
        "engineered_features": list(ENGINEERED_FEATURES),
        "excluded_columns": C.EXCLUDED_COLUMNS,
        "target": C.TARGET,
        "class_labels": C.CLASS_LABELS,
        "available_classifiers": available_classifiers(),
    }


@router.get("/dashboard", summary="Aggregated statistics for the dashboard")
def dashboard(session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    now = utcnow()
    am = model_service.active
    total = session.scalar(select(func.count(Prediction.id))) or 0
    since_24 = now - timedelta(hours=24)
    last24 = session.scalar(select(func.count(Prediction.id)).where(Prediction.created_at >= since_24)) or 0
    placed = session.scalar(select(func.count(Prediction.id)).where(Prediction.prediction == 1)) or 0
    labelled = session.scalar(select(func.count(Prediction.id)).where(Prediction.actual_outcome.is_not(None))) or 0
    correct = session.scalar(
        select(func.count(Prediction.id)).where(Prediction.actual_outcome.is_not(None), Prediction.actual_outcome == Prediction.prediction)
    ) or 0
    avg_prob = session.scalar(select(func.avg(Prediction.probability)))

    days = 14
    start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = session.execute(select(Prediction.created_at, Prediction.prediction).where(Prediction.created_at >= start)).all()
    trend = {(start + timedelta(days=i)).date().isoformat(): {"date": (start + timedelta(days=i)).date().isoformat(), "predictions": 0, "placed": 0} for i in range(days)}
    for ts, pred in rows:
        k = ts.date().isoformat()
        if k in trend:
            trend[k]["predictions"] += 1
            trend[k]["placed"] += int(pred)

    probs = [p for (p,) in session.execute(select(Prediction.probability)).all()]
    bins = [0] * 10
    for p in probs:
        bins[min(int(p * 10), 9)] += 1

    recent = session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(8)).all()
    last_run = session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc())).first()
    last_drift = session.scalars(select(MonitoringEvent).where(MonitoringEvent.kind == "drift_check").order_by(MonitoringEvent.id.desc())).first()
    events = session.scalars(select(MonitoringEvent).order_by(MonitoringEvent.id.desc()).limit(6)).all()
    lat = [l for (l,) in session.execute(select(RequestLog.latency_ms).where(RequestLog.ts >= since_24, RequestLog.path == "/api/predict")).all()]
    reqs24 = session.scalar(select(func.count(RequestLog.id)).where(RequestLog.ts >= since_24)) or 0
    err24 = session.scalar(select(func.count(RequestLog.id)).where(RequestLog.ts >= since_24, RequestLog.status_code >= 500)) or 0

    return {
        "generated_at": now,
        "predictions": {
            "total": total,
            "last_24h": last24,
            "placed": placed,
            "placed_rate": round(placed / total, 4) if total else None,
            "mean_probability": round(float(avg_prob), 4) if avg_prob is not None else None,
            "labelled": labelled,
            "live_accuracy": round(correct / labelled, 4) if labelled else None,
            "trend": list(trend.values()),
            "probability_histogram": [{"bin": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "count": c} for i, c in enumerate(bins)],
        },
        "operations": {
            "requests_24h": reqs24,
            "server_errors_24h": err24,
            "error_rate_24h": round(err24 / reqs24, 4) if reqs24 else None,
            "predict_latency_p50_ms": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
        },
        "model": None
        if am is None
        else {
            "version": am.version,
            "type": am.model_type,
            "label": am.metadata.get("model_label"),
            "threshold": am.threshold,
            "feature_set": am.metadata.get("feature_set"),
            "loaded_at": am.loaded_at,
            "trained_at": (session.scalars(select(ModelVersion.created_at).where(ModelVersion.version == am.version)).first()),
            "test_metrics": {k: am.metadata["metrics"]["test"][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")},
            "test_support": am.metadata["metrics"]["test"]["support"],
            "consistency": am.consistency,
            "cv_results": [
                {"model": r["model"], "label": r["label"], "roc_auc": r["roc_auc"]["mean"], "roc_auc_std": r["roc_auc"]["std"], "f1": r["f1"]["mean"]}
                for r in am.metadata["model_selection"]["results"]
            ],
        },
        "models_registered": session.scalar(select(func.count(ModelVersion.id))) or 0,
        "training": {
            "runs": session.scalar(select(func.count(TrainingRun.id))) or 0,
            "job_running": job_running(),
            "last": None
            if last_run is None
            else {"job_id": last_run.job_id, "status": last_run.status, "decision": last_run.decision, "reason": last_run.reason, "finished_at": last_run.finished_at, "new_version": last_run.new_version},
        },
        "monitoring": {
            "last_drift_check": None if last_drift is None else {"ts": last_drift.ts, "severity": last_drift.severity, "message": last_drift.message},
            "recent_events": [{"ts": e.ts, "kind": e.kind, "severity": e.severity, "message": e.message} for e in events],
        },
        "recent_predictions": [
            {
                "request_id": p.request_id,
                "created_at": p.created_at,
                "source": p.source,
                "label": C.CLASS_LABELS[p.prediction],
                "prediction": p.prediction,
                "probability": round(p.probability, 4),
                "model_version": p.model_version,
                "cgpa": p.inputs.get("CGPA"),
                "stream": p.inputs.get("Stream"),
                "tier": p.inputs.get("CollegeTier"),
            }
            for p in recent
        ],
    }


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@router.get("/lifecycle", summary="Real status of every ML lifecycle stage")
def lifecycle(session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    am = model_service.active
    datasets = session.scalars(select(Dataset).order_by(Dataset.id.desc())).all()
    ds = next((d for d in datasets if d.source == "default"), datasets[0] if datasets else None)
    runs = session.scalar(select(func.count(TrainingRun.id))) or 0
    last_run = session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc())).first()
    last_ok = session.scalars(select(TrainingRun).where(TrainingRun.status == "succeeded").order_by(TrainingRun.id.desc())).first()
    versions = session.scalar(select(func.count(ModelVersion.id))) or 0
    preds = session.scalar(select(func.count(Prediction.id))) or 0
    last_pred = session.scalars(select(Prediction).order_by(Prediction.id.desc())).first()
    drift_checks = session.scalar(select(func.count(MonitoringEvent.id)).where(MonitoringEvent.kind == "drift_check")) or 0
    lab_runs = session.scalar(select(func.count(ExperimentRun.id))) or 0
    root = settings.project_root
    dvc_files = {
        "repo_initialised": (root / ".dvc").is_dir(),
        "dataset_tracked": (root / "data" / "raw" / "placement_50k.csv.dvc").exists(),
        "pipeline_defined": (root / "dvc.yaml").exists(),
        "lock_file": (root / "dvc.lock").exists(),
    }
    mlflow_runs = session.scalar(select(func.count(TrainingRun.id)).where(TrainingRun.mlflow_run_id.is_not(None))) or 0
    mlflow_errors = session.scalars(select(TrainingRun.mlflow_error).where(TrainingRun.mlflow_error.is_not(None)).order_by(TrainingRun.id.desc())).first()

    def st(ok: bool, warn: bool = False) -> str:
        return "complete" if ok else ("attention" if warn else "pending")

    md = am.metadata if am else {}
    stages = [
        {
            "key": "ingestion",
            "title": "Dataset ingestion",
            "status": st(ds is not None),
            "detail": f"{ds.name}: {ds.n_rows:,} rows x {ds.n_columns} columns (sha256 {ds.sha256[:12]}...)" if ds else "No dataset registered",
            "facts": {"datasets_registered": len(datasets)},
        },
        {
            "key": "validation",
            "title": "Data validation",
            "status": st(ds is not None and ds.validation_status == "passed", ds is not None and ds.validation_status != "failed"),
            "detail": f"Contract validation {ds.validation_status.replace('_', ' ')}" if ds else "Waiting for data",
            "facts": {
                "warnings": [c["name"] for c in (ds.validation_report["checks"] if ds else []) if c["status"] == "warn"],
                "failures": [c["name"] for c in (ds.validation_report["checks"] if ds else []) if c["status"] == "fail"],
            },
        },
        {
            "key": "features",
            "title": "Feature processing",
            "status": st(am is not None),
            "detail": f"{len(md.get('features', []))} contract features + {len(md.get('engineered_features', []))} engineered -> {len(md.get('transformed_features', []))} model inputs" if am else "Defined by ml/contract.py and ml/features.py",
            "facts": {"feature_set": md.get("feature_set"), "excluded": list(C.EXCLUDED_COLUMNS)},
        },
        {
            "key": "training",
            "title": "Model training",
            "status": st(last_ok is not None, runs > 0),
            "detail": f"{runs} training run(s); last: {last_run.status}" if last_run else "No training runs yet",
            "facts": {"last_finished_at": last_run.finished_at if last_run else None, "candidates": md.get("config", {}).get("candidates")},
        },
        {
            "key": "evaluation",
            "title": "Model evaluation",
            "status": st(am is not None),
            "detail": f"Held-out test ROC-AUC {md['metrics']['test']['roc_auc']}, F1 {md['metrics']['test']['f1']} on {md['metrics']['test']['support']:,} rows" if am else "Pending a trained model",
            "facts": {"protocol": md.get("model_selection", {}).get("protocol"), "splits": md.get("splits")},
        },
        {
            "key": "registration",
            "title": "Model registration",
            "status": st(versions > 0),
            "detail": f"{versions} version(s) registered; active {am.version}" if am else f"{versions} version(s) registered",
            "facts": {"artifact_sha256": md.get("artifact_sha256"), "mlflow_runs": mlflow_runs},
        },
        {
            "key": "serving",
            "title": "Prediction serving",
            "status": st(am is not None and bool(am.consistency.get("passed")), am is not None),
            "detail": (f"Model {am.version} loaded; golden consistency check {'passed' if am.consistency.get('passed') else 'failed'}" if am else (model_service.load_error or "No model")),
            "facts": {"predictions_served": preds, "last_prediction_at": last_pred.created_at if last_pred else None},
        },
        {
            "key": "monitoring",
            "title": "Monitoring",
            "status": st(drift_checks > 0, preds > 0),
            "detail": f"{preds} predictions logged, {drift_checks} drift check(s) recorded",
            "facts": {"min_rows_for_drift": settings.min_drift_rows},
        },
        {
            "key": "retraining",
            "title": "Retraining",
            "status": st(runs > 1 and last_run is not None and last_run.status in ("succeeded", "rejected"), runs > 0),
            "detail": f"Last job {last_run.status}: {last_run.reason}" if last_run and last_run.reason else "Use the Models page or POST /api/training/jobs",
            "facts": {"job_running": job_running()},
        },
    ]
    return {
        "stages": stages,
        "last_prediction_request_id": last_pred.request_id if last_pred else None,
        "tooling": {
            "mlflow": {"installed_version": _pkg("mlflow"), "enabled": settings.mlflow_enabled, "tracking_uri": "sqlite:///mlflow.db (project root)", "runs_logged": mlflow_runs, "last_error": mlflow_errors},
            "dvc": {"installed_version": _pkg("dvc"), **dvc_files},
            "scikit_learn": _pkg("scikit-learn"),
            "optional": {k: _pkg(k) for k in ("xgboost", "lightgbm", "shap", "optuna", "umap-learn", "skl2onnx", "onnxruntime")},
        },
        "lab_runs": lab_runs,
    }
