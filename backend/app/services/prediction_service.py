"""Online and batch inference with persistence of every prediction (inputs, feature vector, timings)."""
from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from ml import contract as C

from ..db import utcnow
from ..orm import Prediction
from .model_service import model_service

NULLABLE = [f.name for f in C.NUMERIC_FEATURES if f.nullable]


def _imputed(record: dict[str, Any], features: list[str]) -> list[str]:
    return [n for n in NULLABLE if n in features and record.get(n) is None]


def predict(session: Session, records: list[dict[str, Any]], *, source: str, batch_id: str | None = None) -> tuple[list[Prediction], dict[str, float], Any]:
    """Score API-validated records and persist one ``Prediction`` row per record.

    Rows are flushed in the caller's transaction, so a failure leaves no partial batch.
    """
    model_service.sync(session)
    t0 = time.perf_counter()
    am, proba, vectors, timings = model_service.predict(records)
    threshold = am.threshold
    created = utcnow()
    rows: list[Prediction] = []
    t_persist = time.perf_counter()
    per_row_ms = (t_persist - t0) * 1000 / max(len(records), 1)
    for rec, p, vec in zip(records, proba, vectors):
        row = Prediction(
            request_id=uuid.uuid4().hex,
            batch_id=batch_id,
            source=source,
            created_at=created,
            model_version=am.version,
            inputs=rec,
            feature_vector=[round(float(v), 6) for v in vec],
            imputed_fields=_imputed(rec, am.features),
            probability=float(p),
            prediction=int(p >= threshold),
            threshold=threshold,
            latency_ms=round(per_row_ms, 3),
            timings={k: round(v, 3) for k, v in timings.items()},
        )
        rows.append(row)
    session.add_all(rows)
    session.flush()
    timings["persistence_ms"] = (time.perf_counter() - t_persist) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    if len(rows) == 1:
        rows[0].latency_ms = round(timings["total_ms"], 3)
        rows[0].timings = {k: round(v, 3) for k, v in timings.items()}
    return rows, {k: round(v, 3) for k, v in timings.items()}, am
