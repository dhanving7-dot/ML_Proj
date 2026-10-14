"""Online and batch inference with persistence of every prediction (inputs, feature vector, scores, decision)."""
from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from ml import contract as C

from ..db import utcnow
from ..orm import Prediction
from .model_service import ActiveModel, model_service


def _r(x: float, nd: int = 8) -> float:
    return round(float(x), nd)


def predict(session: Session, records: list[dict[str, Any]], *, source: str, batch_id: str | None = None) -> tuple[list[Prediction], dict[str, float], ActiveModel]:
    """Score API-validated records and persist one ``Prediction`` row per record.

    Rows are flushed in the caller's transaction, so a failure leaves no partial batch.
    """
    model_service.sync(session)
    t0 = time.perf_counter()
    am, frame, vectors, s, explanations, timings = model_service.predict(records)
    policy = am.policy
    created = utcnow()
    rows: list[Prediction] = []
    t_persist = time.perf_counter()
    per_row_ms = (t_persist - t0) * 1000 / max(len(records), 1)
    for i, rec in enumerate(records):
        features = {k: float(rec[k]) for k in C.FEATURE_NAMES}
        rows.append(
            Prediction(
                request_id=uuid.uuid4().hex,
                batch_id=batch_id,
                source=source,
                created_at=created,
                model_version=am.version,
                transaction_id=rec.get(C.TRANSACTION_ID),
                amount=features["Amount"],
                time_s=features["Time"],
                inputs=features,
                feature_vector=[round(float(v), 6) for v in vectors[i]],
                lr_probability=_r(s["lr_probability"][i], 10),
                anomaly_score=_r(s["anomaly_score"][i]),
                anomaly_percentile=_r(s["anomaly_percentile"][i]),
                hybrid_score=_r(s["hybrid_score"][i]),
                decision=C.DECISIONS[int(s["decision"][i])],
                lr_flag=int(s["lr_flag"][i]),
                if_flag=int(s["if_flag"][i]),
                policy=policy,
                explanation={**explanations[i], "lr_rarity": _r(s["lr_rarity"][i]), "if_rarity": _r(s["if_rarity"][i])},
                out_of_range=explanations[i]["out_of_training_range"],
                latency_ms=round(per_row_ms, 3),
                timings={k: round(v, 3) for k, v in timings.items()},
            )
        )
    session.add_all(rows)
    session.flush()
    timings["persistence_ms"] = (time.perf_counter() - t_persist) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    if len(rows) == 1:
        rows[0].latency_ms = round(timings["total_ms"], 3)
        rows[0].timings = {k: round(v, 3) for k, v in timings.items()}
    return rows, {k: round(v, 3) for k, v in timings.items()}, am


def response_dict(row: Prediction, model_type: str) -> dict[str, Any]:
    exp = row.explanation or {}
    return {
        "request_id": row.request_id,
        "created_at": row.created_at,
        "model_version": row.model_version,
        "model_type": model_type,
        "transaction_id": row.transaction_id,
        "decision": row.decision,
        "scores": {
            "lr_probability": row.lr_probability,
            "anomaly_score": row.anomaly_score,
            "anomaly_percentile": row.anomaly_percentile,
            "lr_rarity": exp.get("lr_rarity", 0.0),
            "if_rarity": exp.get("if_rarity", 0.0),
            "hybrid_score": row.hybrid_score,
        },
        "flags": {"logistic_regression": bool(row.lr_flag), "isolation_forest": bool(row.if_flag)},
        "policy": row.policy,
        "explanation": exp,
        "out_of_training_range": row.out_of_range or [],
        "latency_ms": row.latency_ms,
        "timings": row.timings or {},
    }
