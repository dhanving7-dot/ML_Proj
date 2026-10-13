"""Controlled (re)training workflow with safe promotion.

1. Validate the dataset.            4. Compare with the active model (same validation rows).
2. Train a candidate.               5. Register + activate only if every gate passes.
3. Evaluate it (leakage-safe).      6. On any failure the previous model stays active.

A process-wide lock allows one job at a time, artifacts are written atomically and promotion is a
single database transaction, so concurrent requests cannot corrupt the registry.
"""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import contract as C
from ml.data import load_csv
from ml.registry import save_artifact
from ml.tracking import log_training_run
from ml.training import DataValidationError, TrainConfig, train_model

from ..config import Settings
from ..db import session_scope, utcnow
from ..errors import InvalidInput, JobConflict, NotFound
from ..orm import Dataset, ModelVersion, TrainingRun
from .dataset_service import default_dataset
from .events import record_event
from .model_service import golden_check, model_service

_job_lock = threading.Lock()


def job_running() -> bool:
    return _job_lock.locked()


def _metrics_summary(md: dict[str, Any]) -> dict[str, Any]:
    keep = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "log_loss", "brier", "threshold", "confusion_matrix", "support")
    return {
        "validation": {k: md["metrics"]["validation"].get(k) for k in keep},
        "test": {k: md["metrics"]["test"].get(k) for k in keep},
        "cv": next((r for r in md["model_selection"]["results"] if r["model"] == md["model_type"]), None),
    }


def _next_version(session: Session, models_dir: Path) -> str:
    n = (session.scalar(select(func.count(ModelVersion.id))) or 0) + 1
    while (models_dir / f"v{n}").exists() or session.scalars(select(ModelVersion).where(ModelVersion.version == f"v{n}")).first():
        n += 1
    return f"v{n}"


def _log(job_id: str, stage: str, message: str) -> None:
    with session_scope() as s:
        run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
        run.stage = stage
        run.message = message
        run.log = [*(run.log or []), {"ts": utcnow().isoformat(), "stage": stage, "message": message}]


def create_job(session: Session, config: TrainConfig, dataset_id: int | None, trigger: str) -> TrainingRun:
    if dataset_id is None:
        ds = default_dataset(session)
        if ds is None:
            raise NotFound("No default dataset is registered")
    else:
        ds = session.get(Dataset, dataset_id)
        if ds is None:
            raise NotFound(f"Dataset {dataset_id} not found")
    if ds.validation_status == "failed":
        raise InvalidInput(f"Dataset {ds.id} failed validation and cannot be used for training")
    run = TrainingRun(job_id=uuid.uuid4().hex, trigger=trigger, status="queued", config=config.to_dict(), dataset_id=ds.id, log=[])
    session.add(run)
    session.flush()
    return run


def start_background_job(settings: Settings, config: TrainConfig, dataset_id: int | None, trigger: str = "api") -> str:
    if not _job_lock.acquire(blocking=False):
        raise JobConflict("A training job is already running; wait for it to finish")
    try:
        with session_scope() as s:
            job_id = create_job(s, config, dataset_id, trigger).job_id
    except Exception:
        _job_lock.release()
        raise
    threading.Thread(target=_worker, args=(settings, job_id), name=f"train-{job_id[:8]}", daemon=True).start()
    return job_id


def run_job_blocking(settings: Settings, config: TrainConfig, dataset_id: int | None = None, trigger: str = "cli") -> str:
    if not _job_lock.acquire(blocking=False):
        raise JobConflict("A training job is already running")
    try:
        with session_scope() as s:
            job_id = create_job(s, config, dataset_id, trigger).job_id
    except Exception:
        _job_lock.release()
        raise
    _worker(settings, job_id)
    return job_id


def _worker(settings: Settings, job_id: str) -> None:
    try:
        _execute(settings, job_id)
    finally:
        _job_lock.release()


def _fail(job_id: str, reason: str, kind: str = "retrain_failed", details: dict[str, Any] | None = None) -> None:
    with session_scope() as s:
        run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
        run.status = "failed"
        run.reason = reason
        run.finished_at = utcnow()
        run.log = [*(run.log or []), {"ts": utcnow().isoformat(), "stage": "failed", "message": reason}]
        active = model_service.active
        record_event(
            s,
            kind,
            "warning",
            f"Training job {job_id[:8]} failed: {reason}. Active model {active.version if active else 'none'} is unchanged.",
            {"job_id": job_id, **(details or {})},
        )


def _execute(settings: Settings, job_id: str) -> None:
    with session_scope() as s:
        run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
        run.status = "running"
        run.started_at = utcnow()
        config = TrainConfig.from_dict(run.config)
        ds = s.get(Dataset, run.dataset_id)
        assert ds is not None
        ds_info = {"id": ds.id, "name": ds.name, "sha256": ds.sha256, "rows": ds.n_rows}
        ds_path = ds.path
        active = model_service.active
        run.active_version_before = active.version if active else None

    try:
        _log(job_id, "ingestion", f"Loading dataset '{ds_info['name']}' ({ds_info['rows']} rows)")
        raw = load_csv(ds_path)
        outcome = train_model(raw, config, progress=lambda stage, msg: _log(job_id, stage, msg))
    except DataValidationError as exc:
        failed = [c for c in exc.report["checks"] if c["status"] == "fail"]
        _fail(job_id, f"Dataset failed validation ({', '.join(c['name'] for c in failed)})", details={"failed_checks": failed})
        return
    except (OSError, ValueError) as exc:
        _fail(job_id, f"{type(exc).__name__}: {str(exc)[:300]}" if isinstance(exc, ValueError) else "Dataset file could not be read")
        return
    except Exception as exc:  # unexpected error: keep the active model, record the reason
        _fail(job_id, f"Unexpected {type(exc).__name__} during training", details={"trace": traceback.format_exc(limit=3)[-1500:]})
        return

    md = outcome.metadata
    md["dataset"] = ds_info
    md["training_job_id"] = job_id
    _log(job_id, "registration", "Comparing candidate with the active model")

    val = outcome.splits.validation
    y_val = val[C.TARGET].to_numpy()
    cand_auc = float(md["metrics"]["validation"]["roc_auc"])
    comparison: dict[str, Any] = {
        "metric": "validation ROC-AUC (threshold-independent)",
        "validation_rows": int(len(val)),
        "candidate": round(cand_auc, 4),
        "tolerance": settings.promotion_tolerance,
        "min_required": config.min_validation_roc_auc,
    }
    active = model_service.active
    if active is not None:
        try:
            act_auc = float(roc_auc_score(y_val, active.model.predict_proba(val[active.features])[:, 1]))
            comparison.update({"active_version": active.version, "active": round(act_auc, 4), "difference": round(cand_auc - act_auc, 4)})
            act_ds = (active.metadata.get("dataset") or {}).get("sha256")
            act_cfg = active.metadata.get("config") or {}
            split_keys = ("seed", "validation_size", "test_size", "sample_rows")
            if (act_ds and act_ds != ds_info["sha256"]) or any(act_cfg.get(k) != config.to_dict().get(k) for k in split_keys):
                comparison["note"] = (
                    "The active model was trained with a different dataset version or split, so some of these validation rows "
                    "may have been in its training data; the comparison can favour the active model."
                )
        except Exception as exc:
            comparison["active_error"] = f"Active model could not be scored on the candidate's validation rows ({type(exc).__name__})"
    consistency = golden_check(outcome.model, md)
    comparison["candidate_consistency_check"] = consistency

    if cand_auc < config.min_validation_roc_auc:
        decision, reason = "rejected", f"Quality gate failed: validation ROC-AUC {cand_auc:.4f} < required {config.min_validation_roc_auc:.4f}"
    elif not consistency.get("passed"):
        decision, reason = "rejected", "Training-serving consistency check failed on golden examples"
    elif "active" in comparison and cand_auc < comparison["active"] - settings.promotion_tolerance:
        decision, reason = "rejected", f"Candidate ROC-AUC {cand_auc:.4f} is worse than active {comparison['active']:.4f} by more than the {settings.promotion_tolerance} tolerance"
    else:
        decision = "promoted"
        reason = (
            f"Passed quality gate and consistency check; candidate ROC-AUC {cand_auc:.4f}"
            + (f" vs active {comparison['active']:.4f}" if "active" in comparison else " (no previous active model)")
        )

    summary = _metrics_summary(md)
    mlflow: dict[str, Any] = {}
    version: str | None = None
    artifact_info: dict[str, Any] | None = None
    if decision == "promoted":
        try:
            with session_scope() as s:
                version = _next_version(s, settings.models_dir)
            artifact_info = save_artifact(settings.models_dir, version, outcome.model, md)
        except Exception as exc:
            _fail(job_id, f"Artifact could not be written ({type(exc).__name__})")
            return

    if settings.mlflow_enabled:
        _log(job_id, "registration", "Logging run to MLflow")
        mlflow = log_training_run(
            tracking_uri=settings.resolved_mlflow_uri,
            artifact_root=settings.mlflow_artifact_root,
            run_name=f"{md['model_type']}-{version or 'rejected'}-{job_id[:6]}",
            params={**config.to_dict(), "selected_model": md["model_type"], "threshold": md["threshold"], "dataset_sha256": ds_info["sha256"]},
            metrics={"validation": summary["validation"], "test": summary["test"], "cv_roc_auc_mean": (summary["cv"] or {}).get("roc_auc", {}).get("mean")},
            tags={"decision": decision, "version": version or "", "job_id": job_id},
            artifact_dir=Path(artifact_info["path"]) if artifact_info else None,
        )

    if decision == "rejected":
        with session_scope() as s:
            run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
            run.status = "rejected"
            run.decision = "rejected"
            run.reason = reason
            run.candidate_model_type = md["model_type"]
            run.candidate_metrics = summary
            run.comparison = comparison
            run.mlflow_run_id = mlflow.get("run_id")
            run.mlflow_error = mlflow.get("error")
            run.finished_at = utcnow()
            run.log = [*(run.log or []), {"ts": utcnow().isoformat(), "stage": "rejected", "message": reason}]
            record_event(s, "retrain_rejected", "warning", f"Candidate rejected: {reason}. Active model {active.version if active else 'none'} kept.", {"job_id": job_id, "comparison": comparison})
        return

    assert version is not None and artifact_info is not None
    previous = active.version if active else None
    try:
        with session_scope() as s:
            for prev in s.scalars(select(ModelVersion).where(ModelVersion.status == "active")).all():
                prev.status = "archived"
            run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
            s.add(
                ModelVersion(
                    version=version,
                    model_type=md["model_type"],
                    status="active",
                    artifact_path=artifact_info["model_path"],
                    sha256=artifact_info["sha256"],
                    size_bytes=artifact_info["size_bytes"],
                    feature_set=md["feature_set"],
                    threshold=float(md["threshold"]),
                    metrics=summary,
                    dataset_id=ds_info["id"],
                    training_run_id=run.id,
                    mlflow_run_id=mlflow.get("run_id"),
                    activated_at=utcnow(),
                )
            )
            run.status = "succeeded"
            run.decision = "promoted"
            run.reason = reason
            run.new_version = version
            run.candidate_model_type = md["model_type"]
            run.candidate_metrics = summary
            run.comparison = comparison
            run.mlflow_run_id = mlflow.get("run_id")
            run.mlflow_error = mlflow.get("error")
            run.finished_at = utcnow()
            run.log = [*(run.log or []), {"ts": utcnow().isoformat(), "stage": "promoted", "message": f"Registered and activated {version}"}]
            record_event(s, "model_promoted", "info", f"Model {version} ({md['model_type']}) promoted. {reason}", {"job_id": job_id, "previous": previous})
        with session_scope() as s:
            model_service.load_active(s)
    except Exception as exc:
        _rollback_promotion(version, previous, artifact_info["path"])
        _fail(job_id, f"Promotion failed and was rolled back ({type(exc).__name__}); previous model restored")


def _rollback_promotion(version: str, previous: str | None, artifact_dir: str) -> None:
    with session_scope() as s:
        new = s.scalars(select(ModelVersion).where(ModelVersion.version == version)).first()
        if new is not None:
            s.delete(new)
        if previous:
            prev = s.scalars(select(ModelVersion).where(ModelVersion.version == previous)).first()
            if prev is not None:
                prev.status = "active"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    try:
        with session_scope() as s:
            model_service.load_active(s)
    except Exception:
        pass


def activate_version(session: Session, version: str) -> ModelVersion:
    """Manual rollback / roll-forward to a registered version (artifact integrity is re-verified)."""
    if job_running():
        raise JobConflict("Cannot switch models while a training job is running")
    target = session.scalars(select(ModelVersion).where(ModelVersion.version == version)).first()
    if target is None:
        raise NotFound(f"Model version {version} not found")
    previous = session.scalars(select(ModelVersion).where(ModelVersion.status == "active")).all()
    for p in previous:
        p.status = "archived"
    target.status = "active"
    target.activated_at = utcnow()
    session.flush()
    try:
        model_service.load_active(session)
    except Exception:
        target.status = "archived"
        for p in previous:
            p.status = "active"
        session.flush()
        model_service.load_active(session)
        raise
    record_event(session, "model_activated", "info", f"Model {version} manually activated", {"previous": [p.version for p in previous]})
    return target


def run_dict(run: TrainingRun) -> dict[str, Any]:
    return {
        "job_id": run.job_id,
        "trigger": run.trigger,
        "status": run.status,
        "stage": run.stage,
        "message": run.message,
        "config": run.config,
        "dataset_id": run.dataset_id,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_s": round((run.finished_at - run.started_at).total_seconds(), 1) if run.finished_at and run.started_at else None,
        "candidate_model_type": run.candidate_model_type,
        "candidate_metrics": run.candidate_metrics,
        "comparison": run.comparison,
        "active_version_before": run.active_version_before,
        "decision": run.decision,
        "reason": run.reason,
        "new_version": run.new_version,
        "mlflow_run_id": run.mlflow_run_id,
        "mlflow_error": run.mlflow_error,
        "log": run.log,
    }
