"""Online prediction, batch prediction, prediction history (search + filters) and outcome feedback."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ml import contract as C
from ml.features import describe_feature

from ..config import Settings, get_settings
from ..db import get_session, utcnow
from ..errors import InvalidInput, NotFound
from ..orm import ModelVersion, Prediction
from ..schemas import BatchError, BatchItem, BatchRequest, BatchResponse, OutcomeRequest, PredictionResponse, TransactionFeatures
from ..services import prediction_service
from ..services.dataset_service import parse_csv_bytes, read_limited
from ..services.events import record_event

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse, summary="Score one transaction")
def predict_one(
    transaction: TransactionFeatures,  # type: ignore[valid-type]
    session: Session = Depends(get_session, scope="function"),
    source: str = Header("online", alias="X-Prediction-Source", pattern="^(online|replay)$", description="Label replayed dataset rows so they are reported separately."),
) -> PredictionResponse:
    rows, timings, am = prediction_service.predict(session, [transaction.model_dump()], source=source)
    body = prediction_service.response_dict(rows[0], am.model_type)
    body["latency_ms"] = timings["total_ms"]
    body["timings"] = timings
    return PredictionResponse(**body)


def _batch_response(rows: list[Prediction], timings: dict[str, float], batch_id: str, version: str, index_map: list[int], errors: list[BatchError]) -> BatchResponse:
    counts = {d: 0 for d in C.DECISIONS}
    for r in rows:
        counts[r.decision] += 1
    return BatchResponse(
        batch_id=batch_id,
        model_version=version,
        count=len(rows),
        decision_counts=counts,
        latency_ms=timings["total_ms"],
        results=[
            BatchItem(
                index=i,
                request_id=r.request_id,
                transaction_id=r.transaction_id,
                decision=r.decision,  # type: ignore[arg-type]
                hybrid_score=r.hybrid_score,
                lr_probability=r.lr_probability,
                anomaly_percentile=r.anomaly_percentile,
            )
            for i, r in zip(index_map, rows)
        ],
        errors=errors,
    )


@router.post("/predict/batch", response_model=BatchResponse, summary="Batch scoring (JSON, up to 1,000 transactions)")
def predict_batch(
    body: BatchRequest,
    session: Session = Depends(get_session, scope="function"),
    settings: Settings = Depends(get_settings),
    source: str = Header("batch", alias="X-Prediction-Source", pattern="^(batch|replay)$"),
) -> BatchResponse:
    if len(body.records) > settings.max_batch_records:
        raise InvalidInput(f"At most {settings.max_batch_records} records per batch")
    batch_id = uuid.uuid4().hex
    records = [r.model_dump() for r in body.records]
    rows, timings, am = prediction_service.predict(session, records, source=source, batch_id=batch_id)
    return _batch_response(rows, timings, batch_id, am.version, list(range(len(rows))), [])


def _csv_records(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[int], list[BatchError]]:
    missing = [c for c in C.FEATURE_NAMES if c not in df.columns]
    if missing:
        raise InvalidInput(f"CSV is missing required columns: {missing}")
    valid, index_map, errors = [], [], []
    wanted = [*C.FEATURE_NAMES, *([C.TRANSACTION_ID] if C.TRANSACTION_ID in df.columns else [])]
    for i, row in enumerate(df[wanted].to_dict(orient="records")):
        rec: dict[str, Any] = {}
        for name, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[name] = None
            elif name == C.TRANSACTION_ID:
                rec[name] = str(v)
            else:
                rec[name] = v.item() if hasattr(v, "item") else v
        try:
            valid.append(TransactionFeatures.model_validate(rec).model_dump())
            index_map.append(i)
        except ValidationError as exc:
            errors.append(BatchError(index=i, errors=[{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()[:10]]))
    return valid, index_map, errors


@router.post("/predict/batch/csv", response_model=BatchResponse, summary="Batch scoring from a CSV upload")
async def predict_batch_csv(file: UploadFile = File(...), session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> BatchResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise InvalidInput("Only .csv files are accepted")
    content = await read_limited(file, int(settings.max_batch_csv_mb * 1024 * 1024))
    df = parse_csv_bytes(content)
    if len(df) > settings.max_batch_csv_rows:
        raise InvalidInput(f"At most {settings.max_batch_csv_rows} rows per CSV batch")
    valid, index_map, errors = _csv_records(df)
    if not valid:
        raise InvalidInput(f"No valid rows: {len(errors)} rows failed validation")
    batch_id = uuid.uuid4().hex
    rows, timings, am = prediction_service.predict(session, valid, source="batch", batch_id=batch_id)
    return _batch_response(rows, timings, batch_id, am.version, index_map, errors[:200])


def _row_dict(p: Prediction, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "request_id": p.request_id,
        "batch_id": p.batch_id,
        "transaction_id": p.transaction_id,
        "source": p.source,
        "created_at": p.created_at,
        "model_version": p.model_version,
        "decision": p.decision,
        "hybrid_score": p.hybrid_score,
        "lr_probability": p.lr_probability,
        "anomaly_score": p.anomaly_score,
        "anomaly_percentile": p.anomaly_percentile,
        "lr_flag": bool(p.lr_flag),
        "if_flag": bool(p.if_flag),
        "amount": p.amount,
        "time_s": p.time_s,
        "out_of_range": p.out_of_range or [],
        "latency_ms": p.latency_ms,
        "actual_outcome": p.actual_outcome,
        "outcome_recorded_at": p.outcome_recorded_at,
    }
    if full:
        d.update({"inputs": p.inputs, "feature_vector": p.feature_vector, "policy": p.policy, "explanation": p.explanation, "timings": p.timings})
    return d


SortKey = Literal["newest", "oldest", "score_desc", "score_asc", "amount_desc"]


@router.get("/predictions", summary="Prediction history with search, filters and pagination")
def list_predictions(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source: str | None = Query(None, pattern="^(online|batch|replay)$"),
    decision: list[Literal["LOW_RISK", "REVIEW", "HIGH_RISK"]] | None = Query(None),
    model_version: str | None = Query(None, max_length=20),
    q: str | None = Query(None, max_length=64, description="Matches the start of the request ID or any part of the transaction ID"),
    min_score: float | None = Query(None, ge=0, le=1),
    max_score: float | None = Query(None, ge=0, le=1),
    min_amount: float | None = Query(None, ge=0),
    max_amount: float | None = Query(None, ge=0),
    labelled: bool | None = Query(None),
    outcome: int | None = Query(None, ge=0, le=1),
    since: datetime | None = None,
    until: datetime | None = None,
    sort: SortKey = "newest",
    session: Session = Depends(get_session, scope="function"),
) -> dict[str, Any]:
    conds = []
    if source:
        conds.append(Prediction.source == source)
    if decision:
        conds.append(Prediction.decision.in_(decision))
    if model_version:
        conds.append(Prediction.model_version == model_version)
    if q:
        like = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conds.append(or_(Prediction.request_id.like(f"{like}%", escape="\\"), Prediction.transaction_id.like(f"%{like}%", escape="\\")))
    if min_score is not None:
        conds.append(Prediction.hybrid_score >= min_score)
    if max_score is not None:
        conds.append(Prediction.hybrid_score <= max_score)
    if min_amount is not None:
        conds.append(Prediction.amount >= min_amount)
    if max_amount is not None:
        conds.append(Prediction.amount <= max_amount)
    if labelled is not None:
        conds.append(Prediction.actual_outcome.is_not(None) if labelled else Prediction.actual_outcome.is_(None))
    if outcome is not None:
        conds.append(Prediction.actual_outcome == outcome)
    if since is not None:
        conds.append(Prediction.created_at >= since)
    if until is not None:
        conds.append(Prediction.created_at <= until)
    order = {
        "newest": Prediction.id.desc(),
        "oldest": Prediction.id.asc(),
        "score_desc": Prediction.hybrid_score.desc(),
        "score_asc": Prediction.hybrid_score.asc(),
        "amount_desc": Prediction.amount.desc(),
    }[sort]
    query = select(Prediction).where(*conds).order_by(order, Prediction.id.desc())
    total = session.scalar(select(func.count(Prediction.id)).where(*conds)) or 0
    facet_rows = session.execute(select(Prediction.decision, func.count(Prediction.id)).where(*conds).group_by(Prediction.decision)).all()
    items = session.scalars(query.offset(offset).limit(limit)).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "decision_counts": {d: next((int(c) for k, c in facet_rows if k == d), 0) for d in C.DECISIONS},
        "items": [_row_dict(p) for p in items],
    }


@router.get("/predictions/{request_id}", summary="Full trace of one prediction")
def get_prediction(request_id: str, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    p = session.scalars(select(Prediction).where(Prediction.request_id == request_id)).first()
    if p is None:
        raise NotFound("Prediction not found")
    out = _row_dict(p, full=True)
    mv = session.scalars(select(ModelVersion).where(ModelVersion.version == p.model_version)).first()
    names: list[str] = []
    if mv is not None:
        try:
            from pathlib import Path

            from ml.registry import read_metadata

            names = read_metadata(Path(mv.artifact_path).parent).get("transformed_features", [])
        except Exception:
            names = []
    out["feature_names"] = names
    out["feature_labels"] = [describe_feature(n) for n in names]
    t = p.timings or {}
    pol = p.policy or {}
    out["trace"] = [
        {"step": "Request received", "detail": f"{p.source} request validated against the 30-feature contract (Pydantic, finite values, non-negative Time and Amount)"},
        {"step": "Preprocessing", "detail": "Cyclical hour-of-cycle from Time, log(1 + Amount), standardisation with training-split statistics", "ms": t.get("preprocessing_ms")},
        {
            "step": "Scoring",
            "detail": f"LR probability {p.lr_probability:.6f}; Isolation Forest anomaly percentile {p.anomaly_percentile:.4f}; hybrid score {p.hybrid_score:.4f} "
            f"(w = {pol.get('fusion_weight_lr')})",
            "ms": t.get("inference_ms"),
        },
        {
            "step": "Decision",
            "detail": f"{p.decision} (REVIEW at >= {pol.get('review_threshold')}, HIGH_RISK at >= {pol.get('high_risk_threshold')})",
        },
        {"step": "Explanation", "detail": "Exact LR log-odds contributions and score rarities computed from the fitted models", "ms": t.get("explanation_ms")},
        {"step": "Persistence", "detail": "Inputs, feature vector, scores, decision and policy stored in the predictions table", "ms": t.get("persistence_ms")},
        {"step": "Monitoring", "detail": "Request latency logged; inputs and scores join the drift window for this model version"},
    ]
    return out


@router.post("/predictions/{request_id}/outcome", summary="Record the confirmed outcome (feedback) for a prediction")
def record_outcome(request_id: str, body: OutcomeRequest, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    p = session.scalars(select(Prediction).where(Prediction.request_id == request_id)).first()
    if p is None:
        raise NotFound("Prediction not found")
    p.actual_outcome = int(body.actual_outcome)
    p.outcome_recorded_at = utcnow()
    labelled = session.scalar(select(func.count(Prediction.id)).where(Prediction.actual_outcome.is_not(None))) or 0
    if labelled and labelled % 100 == 0:
        record_event(session, "outcomes_recorded", "info", f"{labelled} predictions now have a confirmed outcome")
    flagged = p.decision in ("REVIEW", "HIGH_RISK")
    return {
        "request_id": request_id,
        "actual_outcome": p.actual_outcome,
        "decision": p.decision,
        "flagged": flagged,
        "correct": flagged == bool(p.actual_outcome),
        "recorded_at": p.outcome_recorded_at,
    }
