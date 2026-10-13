"""Model registry, training jobs and datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.registry import read_metadata
from ml.training import TrainConfig

from ..config import Settings, get_settings
from ..db import get_session
from ..errors import InvalidInput, NotFound
from ..orm import Dataset, ModelVersion, TrainingRun
from ..schemas import TrainingJobRequest
from ..services import dataset_service, training_service
from ..services.model_service import model_service

router = APIRouter(prefix="/api", tags=["models", "training", "datasets"])


def _model_dict(mv: ModelVersion) -> dict[str, Any]:
    return {
        "version": mv.version,
        "model_type": mv.model_type,
        "status": mv.status,
        "created_at": mv.created_at,
        "activated_at": mv.activated_at,
        "feature_set": mv.feature_set,
        "threshold": mv.threshold,
        "sha256": mv.sha256,
        "size_bytes": mv.size_bytes,
        "dataset_id": mv.dataset_id,
        "mlflow_run_id": mv.mlflow_run_id,
        "metrics": mv.metrics,
    }


@router.get("/models", summary="List registered model versions")
def list_models(session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    items = session.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())).all()
    am = model_service.active
    return {"active_version": am.version if am else None, "items": [_model_dict(m) for m in items]}


@router.get("/models/{version}", summary="Model details: metadata, metrics, curves, importances")
def model_detail(version: str, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    mv = session.scalars(select(ModelVersion).where(ModelVersion.version == version)).first()
    if mv is None:
        raise NotFound(f"Model version {version} not found")
    md = read_metadata(Path(mv.artifact_path).parent)
    md.pop("golden_examples", None)
    ref = md.pop("reference_profile", None)
    am = model_service.active
    return {
        **_model_dict(mv),
        "metadata": md,
        "reference_features": sorted((ref or {}).get("features", {}).keys()),
        "consistency": am.consistency if am and am.version == version else None,
    }


@router.post("/models/{version}/activate", summary="Activate (roll back to) a registered version")
def activate(version: str, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    mv = training_service.activate_version(session, version)
    return _model_dict(mv)


@router.post("/training/jobs", status_code=202, summary="Start a controlled retraining job")
def start_training(body: TrainingJobRequest, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    cfg_in = body.config.model_dump(exclude_none=True)
    try:
        config = TrainConfig.from_dict(cfg_in)
    except ValueError as exc:
        raise InvalidInput(str(exc)) from None
    job_id = training_service.start_background_job(settings, config, body.dataset_id, trigger="api")
    return {"job_id": job_id, "status": "queued"}


@router.get("/training/jobs", summary="Training / retraining history")
def list_jobs(limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    runs = session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)).all()
    return {"job_running": training_service.job_running(), "items": [training_service.run_dict(r) for r in runs]}


@router.get("/training/jobs/{job_id}", summary="Status of one training job")
def job_status(job_id: str, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    run = session.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).first()
    if run is None:
        raise NotFound("Training job not found")
    return training_service.run_dict(run)


@router.get("/datasets", summary="Registered datasets")
def list_datasets(session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    items = session.scalars(select(Dataset).order_by(Dataset.id.desc())).all()
    return {"items": [dataset_service.dataset_dict(d) for d in items]}


@router.post("/datasets/upload", status_code=201, summary="Upload and validate a CSV dataset")
async def upload_dataset(file: UploadFile = File(...), session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise InvalidInput("Only .csv files are accepted")
    content = await dataset_service.read_limited(file, int(settings.max_upload_mb * 1024 * 1024))
    ds = dataset_service.save_upload(session, settings, file.filename or "upload.csv", content)
    return {**dataset_service.dataset_dict(ds), "validation_report": ds.validation_report}


@router.get("/datasets/{dataset_id}", summary="Dataset metadata and validation report")
def dataset_detail(dataset_id: int, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    ds = dataset_service.get_dataset(session, dataset_id)
    return {**dataset_service.dataset_dict(ds), "validation_report": ds.validation_report}


@router.get("/datasets/{dataset_id}/stats", summary="Column statistics incl. skewness and kurtosis")
def dataset_stats(dataset_id: int, session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    ds = dataset_service.get_dataset(session, dataset_id)
    return dataset_service.dataset_stats(ds)


@router.get("/datasets/{dataset_id}/sample", summary="Real rows from the held-out test split")
def dataset_sample(dataset_id: int, n: int = Query(1, ge=1, le=50), seed: int | None = Query(None, ge=0), session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    ds = dataset_service.get_dataset(session, dataset_id)
    if ds.validation_status == "failed":
        raise InvalidInput("This dataset failed validation")
    return {"dataset_id": ds.id, "split": "test", "items": dataset_service.sample_rows(ds, n, seed)}
