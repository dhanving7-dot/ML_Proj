"""Monitoring and laboratory endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_session
from ..services import monitoring_service
from ..services.lab_service import lab_service

router = APIRouter(prefix="/api", tags=["monitoring", "lab"])


@router.get("/monitoring/summary", summary="Request volume, latency, error rate and prediction distribution")
def monitoring_summary(hours: int = Query(24, ge=1, le=720), session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    return monitoring_service.summary(session, hours)


@router.get("/monitoring/drift", summary="Feature and prediction drift (PSI / KL) vs the training reference")
def monitoring_drift(window: int | None = Query(None, ge=10, le=5000), session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return monitoring_service.drift(session, settings, window)


@router.post("/monitoring/drift/check", summary="Run a drift check and record a monitoring event")
def monitoring_drift_check(session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return monitoring_service.check_drift(session, settings)


@router.get("/monitoring/performance", summary="Live performance from recorded ground-truth outcomes")
def monitoring_performance(session: Session = Depends(get_session, scope="function"), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return monitoring_service.performance(session, settings)


@router.get("/monitoring/events", summary="Monitoring event log")
def monitoring_events(limit: int = Query(50, ge=1, le=500), kind: str | None = Query(None, max_length=40), session: Session = Depends(get_session, scope="function")) -> dict[str, Any]:
    return {"items": monitoring_service.events(session, limit, kind)}


@router.get("/lab/catalog", summary="Available M2-M6 experiments and their parameter schemas")
def lab_catalog() -> dict[str, Any]:
    return {"items": lab_service.catalog()}


@router.post("/lab/{module}/{key}", summary="Run a laboratory experiment")
def lab_run(
    module: str,
    key: str,
    params: dict[str, Any] = Body(default_factory=dict),
    session: Session = Depends(get_session, scope="function"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return lab_service.run(session, settings, module, key, params)
