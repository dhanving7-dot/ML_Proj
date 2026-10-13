"""Helper for recording monitoring events."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..orm import MonitoringEvent


def record_event(session: Session, kind: str, severity: str, message: str, details: dict[str, Any] | None = None) -> MonitoringEvent:
    ev = MonitoringEvent(kind=kind, severity=severity, message=message, details=details)
    session.add(ev)
    return ev
