"""Operational and model monitoring computed from persisted requests and predictions (never simulated)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import metrics as M
from ml.drift import drift_report

from ..config import Settings
from ..db import utcnow
from ..orm import MonitoringEvent, Prediction, RequestLog
from .events import record_event
from .model_service import model_service

PREDICT_PATHS = ("/api/predict", "/api/predict/batch", "/api/predict/batch/csv")


def _bucket(hours: int) -> timedelta:
    if hours <= 1:
        return timedelta(minutes=5)
    if hours <= 24:
        return timedelta(hours=1)
    return timedelta(days=1)


def _floor(ts: datetime, step: timedelta) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=ts.tzinfo)
    return epoch + ((ts - epoch) // step) * step


def summary(session: Session, hours: int = 24) -> dict[str, Any]:
    now = utcnow()
    since = now - timedelta(hours=hours)
    logs = session.execute(
        select(RequestLog.ts, RequestLog.path, RequestLog.status_code, RequestLog.latency_ms).where(RequestLog.ts >= since)
    ).all()
    step = _bucket(hours)
    buckets: dict[datetime, dict[str, Any]] = {}
    t = _floor(since, step)
    while t <= now:
        buckets[t] = {"t": t, "requests": 0, "errors": 0, "client_errors": 0, "latency_sum": 0.0, "predict_requests": 0}
        t += step
    lat_predict = []
    for ts, path, code, lat in logs:
        b = buckets.setdefault(_floor(ts, step), {"t": _floor(ts, step), "requests": 0, "errors": 0, "client_errors": 0, "latency_sum": 0.0, "predict_requests": 0})
        b["requests"] += 1
        b["latency_sum"] += lat
        if code >= 500:
            b["errors"] += 1
        elif code >= 400:
            b["client_errors"] += 1
        if path in PREDICT_PATHS:
            b["predict_requests"] += 1
            lat_predict.append(lat)
    series = [
        {
            "t": b["t"],
            "requests": b["requests"],
            "predict_requests": b["predict_requests"],
            "errors": b["errors"],
            "client_errors": b["client_errors"],
            "mean_latency_ms": round(b["latency_sum"] / b["requests"], 2) if b["requests"] else None,
        }
        for b in sorted(buckets.values(), key=lambda d: d["t"])
    ]
    total = len(logs)
    errors = sum(1 for _, _, c, _ in logs if c >= 500)
    client_errors = sum(1 for _, _, c, _ in logs if 400 <= c < 500)
    preds = session.execute(
        select(Prediction.probability, Prediction.prediction, Prediction.source, Prediction.latency_ms).where(Prediction.created_at >= since)
    ).all()
    probs = np.array([p[0] for p in preds], float)
    hist = np.histogram(probs, bins=10, range=(0, 1))[0].tolist() if len(probs) else [0] * 10
    by_source: dict[str, int] = {}
    for p in preds:
        by_source[p[2]] = by_source.get(p[2], 0) + 1
    minutes = max(hours * 60, 1)
    return {
        "window_hours": hours,
        "generated_at": now,
        "requests": {
            "total": total,
            "server_errors": errors,
            "client_errors": client_errors,
            "error_rate": round(errors / total, 4) if total else None,
            "client_error_rate": round(client_errors / total, 4) if total else None,
            "throughput_per_min": round(total / minutes, 4),
        },
        "latency_ms": {
            "predict_p50": round(float(np.percentile(lat_predict, 50)), 2) if lat_predict else None,
            "predict_p95": round(float(np.percentile(lat_predict, 95)), 2) if lat_predict else None,
            "predict_p99": round(float(np.percentile(lat_predict, 99)), 2) if lat_predict else None,
            "predict_requests": len(lat_predict),
        },
        "predictions": {
            "total": int(len(preds)),
            "placed_rate": round(float(np.mean([p[1] for p in preds])), 4) if preds else None,
            "mean_probability": round(float(probs.mean()), 4) if len(probs) else None,
            "by_source": by_source,
            "probability_histogram": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": int(c)} for i, c in enumerate(hist)],
        },
        "series": series,
    }


def drift(session: Session, settings: Settings, window: int | None = None) -> dict[str, Any]:
    am = model_service.active
    if am is None:
        return {"status": "no_model", "message": model_service.load_error}
    window = window or settings.drift_window
    rows = session.execute(
        select(Prediction.inputs, Prediction.probability, Prediction.created_at)
        .where(Prediction.model_version == am.version)
        .order_by(Prediction.id.desc())
        .limit(window)
    ).all()
    n = len(rows)
    base = {"model_version": am.version, "window": window, "rows": n, "min_rows": settings.min_drift_rows}
    if n < settings.min_drift_rows:
        return {**base, "status": "insufficient_data", "message": f"Drift needs at least {settings.min_drift_rows} predictions from model {am.version}; {n} recorded so far."}
    frame = pd.DataFrame([r[0] for r in rows])
    for f in am.features:
        if f not in frame.columns:
            frame[f] = np.nan
    frame = frame[am.features]
    proba = np.array([r[1] for r in rows], float)
    rep = drift_report(am.metadata["reference_profile"], frame, proba)
    ranked = sorted(
        ({"feature": k, "psi": v["psi"], "kl_divergence": v["kl_divergence"], "status": v["status"], "kind": v["kind"]} for k, v in rep["features"].items()),
        key=lambda d: -(d["psi"] or 0),
    )
    worst = max((d["psi"] or 0) for d in ranked) if ranked else 0
    status = "significant_shift" if worst >= 0.25 else ("investigate" if worst >= 0.10 else "stable")
    return {
        **base,
        "status": status,
        "from": rows[-1][2],
        "to": rows[0][2],
        "features": ranked,
        "prediction": {k: rep["prediction"][k] for k in ("psi", "kl_divergence", "status", "bins")} if "prediction" in rep else None,
        "detail": rep["features"],
        "flagged_features": rep["flagged_features"],
        "thresholds": rep["thresholds"],
        "interpretation": "PSI compares serving inputs with the training distribution. Drift is a signal to investigate, not proof that accuracy dropped; see live performance once outcomes are recorded.",
    }


def check_drift(session: Session, settings: Settings) -> dict[str, Any]:
    rep = drift(session, settings)
    if rep.get("status") in ("stable", "investigate", "significant_shift"):
        severity = {"stable": "info", "investigate": "warning", "significant_shift": "critical"}[rep["status"]]
        record_event(
            session,
            "drift_check",
            severity,
            f"Drift check on {rep['rows']} predictions: {rep['status'].replace('_', ' ')}"
            + (f" ({len(rep['flagged_features'])} flagged: {', '.join(rep['flagged_features'][:5])})" if rep["flagged_features"] else ""),
            {"status": rep["status"], "flagged": rep["flagged_features"], "prediction_psi": (rep.get("prediction") or {}).get("psi"), "model_version": rep["model_version"]},
        )
    return rep


def performance(session: Session, settings: Settings) -> dict[str, Any]:
    am = model_service.active
    q = select(Prediction.probability, Prediction.prediction, Prediction.actual_outcome, Prediction.model_version, Prediction.outcome_recorded_at).where(
        Prediction.actual_outcome.is_not(None)
    )
    rows = session.execute(q.order_by(Prediction.id)).all()
    total_preds = session.scalar(select(func.count(Prediction.id))) or 0
    out: dict[str, Any] = {"labelled": len(rows), "total_predictions": int(total_preds), "min_labelled": settings.min_labelled_for_performance}
    if am is not None:
        out["model_version"] = am.version
        out["offline_test"] = {k: am.metadata["metrics"]["test"][k] for k in ("accuracy", "f1", "roc_auc", "precision", "recall")}
    rows_active = [r for r in rows if am is not None and r[3] == am.version]
    out["labelled_active_model"] = len(rows_active)
    if len(rows_active) < settings.min_labelled_for_performance:
        out["status"] = "insufficient_labels"
        out["message"] = f"Live performance needs at least {settings.min_labelled_for_performance} predictions with recorded outcomes for the active model."
        return out
    y = np.array([r[2] for r in rows_active], int)
    p = np.array([r[0] for r in rows_active], float)
    m = M.classification_metrics(y, p, am.threshold)  # type: ignore[union-attr]
    block = 50
    blocks = []
    for i in range(0, len(y), block):
        yb, pb = y[i : i + block], p[i : i + block]
        if len(yb) >= 10:
            blocks.append({"block": i // block + 1, "rows": int(len(yb)), "accuracy": round(float(np.mean((pb >= am.threshold) == yb)), 4)})  # type: ignore[union-attr]
    drop = (out["offline_test"]["accuracy"] - m["accuracy"]) if am is not None else None
    out.update(
        {
            "status": "degraded" if drop is not None and drop > 0.05 else "healthy",
            "live": m,
            "accuracy_drop_vs_test": round(drop, 4) if drop is not None else None,
            "blocks": blocks,
            "interpretation": "Concept drift shows up as live accuracy falling below the offline test estimate on recent labelled predictions.",
        }
    )
    return out


def events(session: Session, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
    q = select(MonitoringEvent).order_by(MonitoringEvent.id.desc()).limit(limit)
    if kind:
        q = q.where(MonitoringEvent.kind == kind)
    return [
        {"id": e.id, "ts": e.ts, "kind": e.kind, "severity": e.severity, "message": e.message, "details": e.details}
        for e in session.scalars(q).all()
    ]
