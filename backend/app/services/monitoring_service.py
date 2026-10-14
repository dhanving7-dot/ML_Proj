"""Operational and model monitoring computed from persisted requests and predictions (never simulated)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import contract as C
from ml import metrics as M
from ml.drift import PSI_SIGNIFICANT, PSI_STABLE, drift_report

from ..config import Settings
from ..db import utcnow
from ..orm import MonitoringEvent, Prediction, RequestLog
from .events import record_event
from .model_service import model_service

PREDICT_PATHS = ("/api/predict", "/api/predict/batch", "/api/predict/batch/csv")
SCORE_NAMES = ("lr_probability", "anomaly_percentile", "hybrid_score")


def _bucket(hours: int) -> timedelta:
    if hours <= 1:
        return timedelta(minutes=5)
    if hours <= 24:
        return timedelta(hours=1)
    return timedelta(days=1)


def _floor(ts: datetime, step: timedelta) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=ts.tzinfo)
    return epoch + ((ts - epoch) // step) * step


def _histogram(values: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    counts = np.histogram(values, bins=bins, range=(0, 1))[0] if len(values) else np.zeros(bins, int)
    return [{"bin": f"{i / bins:.1f}–{(i + 1) / bins:.1f}", "count": int(c)} for i, c in enumerate(counts)]


def summary(session: Session, hours: int = 24) -> dict[str, Any]:
    now = utcnow()
    since = now - timedelta(hours=hours)
    logs = session.execute(select(RequestLog.ts, RequestLog.path, RequestLog.status_code, RequestLog.latency_ms).where(RequestLog.ts >= since)).all()
    step = _bucket(hours)
    blank = lambda t: {"t": t, "requests": 0, "errors": 0, "client_errors": 0, "latency_sum": 0.0, "predict_requests": 0, "LOW_RISK": 0, "REVIEW": 0, "HIGH_RISK": 0}  # noqa: E731
    buckets: dict[datetime, dict[str, Any]] = {}
    t = _floor(since, step)
    while t <= now:
        buckets[t] = blank(t)
        t += step
    lat_predict = []
    for ts, path, code, lat in logs:
        b = buckets.setdefault(_floor(ts, step), blank(_floor(ts, step)))
        b["requests"] += 1
        b["latency_sum"] += lat
        if code >= 500:
            b["errors"] += 1
        elif code >= 400:
            b["client_errors"] += 1
        if path in PREDICT_PATHS:
            b["predict_requests"] += 1
            lat_predict.append(lat)
    preds = session.execute(
        select(Prediction.created_at, Prediction.decision, Prediction.source, Prediction.hybrid_score, Prediction.lr_probability, Prediction.anomaly_percentile).where(
            Prediction.created_at >= since
        )
    ).all()
    for created, decision, *_ in preds:
        b = buckets.setdefault(_floor(created, step), blank(_floor(created, step)))
        b[decision] += 1
    series = [
        {
            "t": b["t"],
            "requests": b["requests"],
            "predict_requests": b["predict_requests"],
            "errors": b["errors"],
            "client_errors": b["client_errors"],
            "mean_latency_ms": round(b["latency_sum"] / b["requests"], 2) if b["requests"] else None,
            "low_risk": b["LOW_RISK"],
            "review": b["REVIEW"],
            "high_risk": b["HIGH_RISK"],
        }
        for b in sorted(buckets.values(), key=lambda d: d["t"])
    ]
    total = len(logs)
    errors = sum(1 for _, _, c, _ in logs if c >= 500)
    client_errors = sum(1 for _, _, c, _ in logs if 400 <= c < 500)
    by_source: dict[str, int] = {}
    by_decision = {d: 0 for d in C.DECISIONS}
    for p in preds:
        by_source[p[2]] = by_source.get(p[2], 0) + 1
        by_decision[p[1]] += 1
    hybrid = np.array([p[3] for p in preds], float)
    lr = np.array([p[4] for p in preds], float)
    anomaly = np.array([p[5] for p in preds], float)
    minutes = max(hours * 60, 1)
    n = len(preds)
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
            "total": n,
            "by_decision": by_decision,
            "flagged_rate": round((by_decision["REVIEW"] + by_decision["HIGH_RISK"]) / n, 6) if n else None,
            "high_risk_rate": round(by_decision["HIGH_RISK"] / n, 6) if n else None,
            "mean_hybrid_score": round(float(hybrid.mean()), 6) if n else None,
            "by_source": by_source,
            "histograms": {"hybrid_score": _histogram(hybrid), "lr_probability": _histogram(lr), "anomaly_percentile": _histogram(anomaly)},
        },
        "series": series,
    }


def drift(session: Session, settings: Settings, window: int | None = None) -> dict[str, Any]:
    am = model_service.active
    if am is None:
        return {"status": "no_model", "message": model_service.load_error}
    window = window or settings.drift_window
    rows = session.execute(
        select(Prediction.inputs, Prediction.lr_probability, Prediction.anomaly_percentile, Prediction.hybrid_score, Prediction.created_at)
        .where(Prediction.model_version == am.version)
        .order_by(Prediction.id.desc())
        .limit(window)
    ).all()
    n = len(rows)
    base = {"model_version": am.version, "window": window, "rows": n, "min_rows": settings.min_drift_rows}
    if n < settings.min_drift_rows:
        return {**base, "status": "insufficient_data", "message": f"Drift needs at least {settings.min_drift_rows} predictions from model {am.version}; {n} recorded so far."}
    frame = pd.DataFrame([r[0] for r in rows])[am.features]
    scores = {name: np.array([r[i + 1] for r in rows], float) for i, name in enumerate(SCORE_NAMES)}
    rep = drift_report(am.metadata["reference_profile"], frame, scores)
    ranked = sorted(
        ({"feature": k, "psi": v["psi"], "kl_divergence": v["kl_divergence"], "status": v["status"], "kind": v["kind"]} for k, v in rep["features"].items()),
        key=lambda d: -(d["psi"] or 0),
    )
    worst = max((d["psi"] or 0) for d in ranked) if ranked else 0
    status = "significant_shift" if worst >= PSI_SIGNIFICANT else ("investigate" if worst >= PSI_STABLE else "stable")
    return {
        **base,
        "status": status,
        "from": rows[-1][4],
        "to": rows[0][4],
        "features": ranked,
        "scores": {k: {m: v[m] for m in ("psi", "kl_divergence", "status", "bins", "summary")} for k, v in rep.get("scores", {}).items()},
        "detail": rep["features"],
        "flagged_features": rep["flagged_features"],
        "thresholds": rep["thresholds"],
        "interpretation": (
            "PSI compares served inputs with the training split and served scores with the validation split. "
            "Drift is a signal to investigate, not proof that detection got worse; use recorded outcomes for that. "
            "Time is raw seconds since the dataset's first transaction, so its PSI also moves when traffic comes from a different period."
        ),
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
            {
                "status": rep["status"],
                "flagged": rep["flagged_features"],
                "score_psi": {k: v["psi"] for k, v in (rep.get("scores") or {}).items()},
                "model_version": rep["model_version"],
            },
        )
    return rep


def performance(session: Session, settings: Settings) -> dict[str, Any]:
    """Live detection quality from recorded outcomes (feedback). Needs both classes to compute ranking metrics."""
    am = model_service.active
    rows = session.execute(
        select(Prediction.hybrid_score, Prediction.decision, Prediction.actual_outcome, Prediction.model_version)
        .where(Prediction.actual_outcome.is_not(None))
        .order_by(Prediction.id)
    ).all()
    total_preds = session.scalar(select(func.count(Prediction.id))) or 0
    out: dict[str, Any] = {"labelled": len(rows), "total_predictions": int(total_preds), "min_labelled": settings.min_labelled_for_performance}
    if am is not None:
        out["model_version"] = am.version
        t = am.metadata["metrics"]["test"]
        out["offline_test"] = {
            "high_risk_precision": t["hybrid_high_risk"]["precision"],
            "high_risk_recall": t["hybrid_high_risk"]["recall"],
            "review_or_high_recall": t["hybrid_review_or_high"]["recall"],
            "average_precision": t["hybrid_high_risk"]["average_precision"],
        }
    rows_active = [r for r in rows if am is not None and r[3] == am.version]
    out["labelled_active_model"] = len(rows_active)
    if len(rows_active) < settings.min_labelled_for_performance:
        out["status"] = "insufficient_labels"
        out["message"] = f"Live performance needs at least {settings.min_labelled_for_performance} predictions of the active model with a recorded outcome."
        return out
    y = np.array([r[2] for r in rows_active], int)
    h = np.array([r[0] for r in rows_active], float)
    dec = np.array([C.DECISIONS.index(r[1]) for r in rows_active], int)
    high = M.classification_metrics(y, (dec == 2).astype(float), 0.5, probabilistic=False)
    review = M.classification_metrics(y, (dec >= 1).astype(float), 0.5, probabilistic=False)
    ranking = {
        "average_precision": round(float(M.average_precision_score(y, h)), 4) if len(np.unique(y)) == 2 else None,
        "roc_auc": round(float(M.roc_auc_score(y, h)), 4) if len(np.unique(y)) == 2 else None,
    }
    block = 50
    blocks = []
    for i in range(0, len(y), block):
        yb, db = y[i : i + block], dec[i : i + block]
        if len(yb) >= 10:
            flagged = db >= 1
            blocks.append(
                {
                    "block": i // block + 1,
                    "rows": int(len(yb)),
                    "frauds": int(yb.sum()),
                    "recall_review_or_high": round(float(np.sum(flagged & (yb == 1)) / yb.sum()), 4) if yb.sum() else None,
                    "precision_review_or_high": round(float(np.sum(flagged & (yb == 1)) / flagged.sum()), 4) if flagged.any() else None,
                }
            )
    drop = None
    if am is not None and high["positives"]:
        drop = round(out["offline_test"]["review_or_high_recall"] - review["recall"], 4)
    out.update(
        {
            "status": "degraded" if drop is not None and drop > 0.15 else "healthy",
            "live": {"high_risk": high, "review_or_high": review, "ranking": ranking},
            "recall_drop_vs_test": drop,
            "blocks": blocks,
            "interpretation": (
                "Concept drift shows up as recall at REVIEW-or-above falling below the offline test estimate on recent labelled predictions. "
                "Labelled outcomes are a biased sample when investigators mostly confirm flagged transactions."
            ),
        }
    )
    return out


def events(session: Session, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
    q = select(MonitoringEvent).order_by(MonitoringEvent.id.desc()).limit(limit)
    if kind:
        q = q.where(MonitoringEvent.kind == kind)
    return [{"id": e.id, "ts": e.ts, "kind": e.kind, "severity": e.severity, "message": e.message, "details": e.details} for e in session.scalars(q).all()]
