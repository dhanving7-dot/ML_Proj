"""Persistence model: datasets, model versions, training runs, predictions, monitoring and lab runs."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, UTCDateTime, utcnow


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(20))  # default | upload
    n_rows: Mapped[int] = mapped_column(Integer)
    n_columns: Mapped[int] = mapped_column(Integer)
    positive_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(30))
    validation_report: Mapped[dict[str, Any]] = mapped_column(JSON)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    trigger: Mapped[str] = mapped_column(String(20))  # cli | api | dvc
    status: Mapped[str] = mapped_column(String(20))  # queued | running | succeeded | rejected | failed
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    candidate_model_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    candidate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    comparison: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    active_version_before: Mapped[str | None] = mapped_column(String(20), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)  # promoted | rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), index=True)  # active | archived
    artifact_path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    feature_set: Mapped[str] = mapped_column(String(40))
    threshold: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    training_run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_created", "created_at"), Index("ix_predictions_model", "model_version"))

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(20))  # online | batch | replay
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    model_version: Mapped[str] = mapped_column(String(20))
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON)
    feature_vector: Mapped[list[float]] = mapped_column(JSON)
    imputed_fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    threshold: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    timings: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = (Index("ix_request_logs_ts", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    method: Mapped[str] = mapped_column(String(10))
    path: Mapped[str] = mapped_column(String(200))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)


class MonitoringEvent(Base):
    __tablename__ = "monitoring_events"
    __table_args__ = (Index("ix_monitoring_events_ts", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    kind: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(10))  # info | warning | critical
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    module: Mapped[str] = mapped_column(String(10))
    experiment: Mapped[str] = mapped_column(String(60))
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20))
    duration_ms: Mapped[float] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
