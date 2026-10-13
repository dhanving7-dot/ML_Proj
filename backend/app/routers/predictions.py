"""Online prediction, batch prediction, prediction history and ground-truth feedback."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import contract as C

from ..config import Settings, get_settings
from ..db import get_session, utcnow
from ..errors import InvalidInput, NotFound
from ..orm import ModelVersion, Prediction
from ..schemas import BatchError, BatchItem, BatchRequest, BatchResponse, OutcomeRequest, PredictionResponse, StudentFeatures
from ..services import prediction_service
from ..services.dataset_service import parse_csv_bytes, read_limited
from ..services.events import record_event

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse, summary="Online prediction for one student")
def predict_one(
    student: StudentFeatures,  # type: ignore[valid-type]
    session: Session = Depends(get_session, scope="function"),
    source: str = Header("online", alias="X-Prediction-Source", pattern="^(online|replay)$", description="Label replayed traffic so it is reported separately."),
) -> PredictionResponse:
    rows, timings, am = prediction_service.predict(session, [student.model_dump()], source=source)
    row = rows[0]
    return PredictionResponse(
        request_id=row.request_id,
        created_at=row.created_at,
        model_version=row.model_version,
        model_type=am.model_type,
        prediction=row.prediction,
        label=C.CLASS_LABELS[row.prediction],
        probability=round(row.probability, 6),
        threshold=row.threshold,
        latency_ms=timings["total_ms"],
        timings=timings,
        imputed_fields=row.imputed_fields,
    )


def _batch_response(rows: list[Prediction], timings: dict[str, float], batch_id: str, version: str, index_map: list[int], errors: list[BatchError]) -> BatchResponse:
    return BatchResponse(
        batch_id=batch_id,
        model_version=version,
        count=len(rows),
        placed_count=sum(r.prediction for r in rows),
        latency_ms=timings["total_ms"],
        results=[
            BatchItem(index=i, request_id=r.request_id, prediction=r.prediction, label=C.CLASS_LABELS[r.prediction], probability=round(r.probability, 6))
            for i, r in zip(index_map, rows)
        ],
        errors=errors,
    )


@router.post("/predict/batch", response_model=BatchResponse, summary="Batch prediction (JSON, up to 1,000 records)")
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
    missing = [c for c in C.FEATURE_NAMES if c not in df.columns and not C.FEATURES_BY_NAME[c].nullable]  # type: ignore[union-attr]
    if missing:
        raise InvalidInput(f"CSV is missing required columns: {missing}")
    valid, index_map, errors = [], [], []
    for i, row in enumerate(df.to_dict(orient="records")):
        rec = {}
        for name in C.FEATURE_NAMES:
            v = row.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[name] = None
            else:
                rec[name] = v.item() if hasattr(v, "item") else v
        try:
            valid.append(StudentFeatures.model_validate(rec).model_dump())
            index_map.append(i)
        except ValidationError as exc:
            errors.append(
                BatchError(index=i, errors=[{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()[:10]])
            )
    return valid, index_map, errors


@router.post("/predict/batch/csv", response_model=BatchResponse, summary="Batch prediction from a CSV upload")
async def predict_batch_csv(file: UploadFile = File(...), session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> BatchResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise InvalidInput("Only .csv files are accepted")
    content = await read_limited(file, 5 * 1024 * 1024)
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
        "source": p.source,
        "created_at": p.created_at,
        "model_version": p.model_version,
        "prediction": p.prediction,
        "label": C.CLASS_LABELS[p.prediction],
        "probability": round(p.probability, 6),
        "threshold": p.threshold,
        "latency_ms": p.latency_ms,
        "actual_outcome": p.actual_outcome,
        "outcome_recorded_at": p.outcome_recorded_at,
        "summary": {k: p.inputs.get(k) for k in ("CGPA", "CollegeTier", "Stream", "Internships", "CodingTestScore")},
    }
    if full:
        d.update({"inputs": p.inputs, "feature_vector": p.feature_vector, "imputed_fields": p.imputed_fields, "timings": p.timings})
    return d


@router.get("/predictions", summary="Prediction history (newest first)")
def list_predictions(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source: str | None = Query(None, pattern="^(online|batch|replay)$"),
    session: Session = Depends(get_session, scope="function"),
) -> dict[str, Any]:
    q = select(Prediction).order_by(Prediction.id.desc())
    cq = select(func.count(Prediction.id))
    if source:
        q = q.where(Prediction.source == source)
        cq = cq.where(Prediction.source == source)
    items = session.scalars(q.offset(offset).limit(limit)).all()
    return {"total": session.scalar(cq) or 0, "limit": limit, "offset": offset, "items": [_row_dict(p) for p in items]}


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
            from ml.registry import read_metadata

            names = read_metadata(Path(mv.artifact_path).parent).get("transformed_features", [])
        except Exception:
            names = []
    out["feature_names"] = names
    t = p.timings or {}
    out["trace"] = [
        {"step": "Request received", "detail": f"{p.source} request validated against the feature contract (Pydantic)"},
        {"step": "Preprocessing", "detail": "Engineered SGPA features, imputation, scaling and encoding with training-fitted parameters", "ms": t.get("preprocessing_ms")},
        {"step": "Inference", "detail": f"Model {p.model_version} produced probability {p.probability:.4f} (threshold {p.threshold:.2f})", "ms": t.get("inference_ms")},
        {"step": "Persistence", "detail": "Inputs, feature vector and result stored in the predictions table", "ms": t.get("persistence_ms")},
        {"step": "Monitoring", "detail": "Request latency logged; the inputs join the drift window for this model version"},
    ]
    return out


@router.post("/predictions/{request_id}/outcome", summary="Record the ground-truth outcome for a prediction")
def record_outcome(request_id: str, body: OutcomeRequest, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    p = session.scalars(select(Prediction).where(Prediction.request_id == request_id)).first()
    if p is None:
        raise NotFound("Prediction not found")
    p.actual_outcome = int(body.actual_outcome)
    p.outcome_recorded_at = utcnow()
    labelled = session.scalar(select(func.count(Prediction.id)).where(Prediction.actual_outcome.is_not(None))) or 0
    if labelled and labelled % 100 == 0:
        record_event(session, "outcomes_recorded", "info", f"{labelled} predictions now have ground-truth outcomes")
    return {"request_id": request_id, "actual_outcome": p.actual_outcome, "correct": p.actual_outcome == p.prediction, "recorded_at": p.outcome_recorded_at}
