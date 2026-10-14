"""Holds the active model in memory and guarantees it is the trusted, registered artifact."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import contract as C
from ml.features import coerce_features
from ml.fraud_model import FraudModel
from ml.registry import load_artifact, read_metadata

from ..db import utcnow
from ..orm import ModelVersion
from ..schemas import TransactionFeatures
from .events import record_event


@dataclass
class ActiveModel:
    version: str
    model_type: str
    model: FraudModel
    metadata: dict[str, Any]
    artifact_path: str
    loaded_at: datetime
    consistency: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    @property
    def policy(self) -> dict[str, Any]:
        return self.model.policy.to_dict()


def serving_frame(records: list[dict[str, Any]], features: list[str] | None = None) -> pd.DataFrame:
    """The serving path: API-validated records -> contract-coerced frame (same function training uses)."""
    return coerce_features(records, features or C.FEATURE_NAMES)


def validate_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Validate one JSON record exactly as POST /api/predict does."""
    return TransactionFeatures.model_validate(rec).model_dump()


def golden_check(model: FraudModel, metadata: dict[str, Any]) -> dict[str, Any]:
    """Replay the golden examples stored at training time through the *serving* path.

    Each stored raw input is validated with the API schema, transformed by the loaded artifact
    and scored; the feature vector, both component scores, the hybrid score and the decision
    must reproduce what training recorded.
    """
    golden = metadata.get("golden_examples") or {}
    inputs = golden.get("inputs") or []
    if not inputs:
        return {"passed": False, "reason": "No golden examples stored with this model"}
    validated = [validate_record(rec) for rec in inputs]
    frame = serving_frame(validated)
    vectors = model.transform(frame)
    scores = model.score_transformed(vectors)
    diffs = {
        "max_feature_vector_diff": float(np.max(np.abs(np.round(vectors, 8) - np.asarray(golden["feature_vectors"], float)))),
        "max_lr_probability_diff": float(np.max(np.abs(scores["lr_probability"] - np.asarray(golden["lr_probability"], float)))),
        "max_anomaly_score_diff": float(np.max(np.abs(scores["anomaly_score"] - np.asarray(golden["anomaly_score"], float)))),
        "max_hybrid_score_diff": float(np.max(np.abs(scores["hybrid_score"] - np.asarray(golden["hybrid_score"], float)))),
    }
    decision_mismatches = int(np.sum(scores["decision"] != np.asarray(golden["decision"], int)))
    passed = (
        diffs["max_feature_vector_diff"] < 1e-6
        and diffs["max_lr_probability_diff"] < 1e-9
        and diffs["max_anomaly_score_diff"] < 1e-9
        and diffs["max_hybrid_score_diff"] < 1e-9
        and decision_mismatches == 0
    )
    return {"passed": bool(passed), "examples": len(inputs), **diffs, "decision_mismatches": decision_mismatches, "checked_at": utcnow().isoformat()}


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: ActiveModel | None = None
        self.load_error: str | None = "Model not loaded yet"
        self._last_sync = 0.0

    @property
    def active(self) -> ActiveModel | None:
        with self._lock:
            return self._active

    def sync(self, session: Session, min_interval_s: float = 5.0) -> None:
        """Reload if another process (CLI / DVC pipeline) activated a different version.

        Throttled so the registry query runs at most every ``min_interval_s`` seconds.
        """
        now = time.monotonic()
        if now - self._last_sync < min_interval_s:
            return
        self._last_sync = now
        current = session.scalars(select(ModelVersion.version).where(ModelVersion.status == "active").order_by(ModelVersion.id.desc())).first()
        loaded = self._active.version if self._active else None
        if current != loaded:
            try:
                self.load_active(session)
            except Exception:
                pass  # load_active already recorded the failure; keep serving health/docs

    def load_active(self, session: Session) -> ActiveModel | None:
        rec = session.scalars(select(ModelVersion).where(ModelVersion.status == "active").order_by(ModelVersion.id.desc())).first()
        if rec is None:
            with self._lock:
                self._active = None
                self.load_error = "No active model is registered. Train one with `python -m scripts.train`."
            return None
        try:
            model = load_artifact(rec.artifact_path, rec.sha256)
            if not isinstance(model, FraudModel):
                raise TypeError(f"Artifact {rec.version} is a {type(model).__name__}, not a FraudModel")
            metadata = read_metadata(Path(rec.artifact_path).parent)
            consistency = golden_check(model, metadata)
        except Exception as exc:
            with self._lock:
                self._active = None
                self.load_error = f"Active model {rec.version} could not be loaded: {type(exc).__name__}"
            record_event(session, "model_load_failed", "critical", self.load_error, {"version": rec.version})
            raise
        active = ActiveModel(
            version=rec.version,
            model_type=rec.model_type,
            model=model,
            metadata=metadata,
            artifact_path=rec.artifact_path,
            loaded_at=utcnow(),
            consistency=consistency,
        )
        with self._lock:
            self._active = active
            self.load_error = None
        record_event(
            session,
            "model_loaded",
            "info" if consistency.get("passed") else "critical",
            f"Loaded model {rec.version}; golden-example consistency check {'passed' if consistency.get('passed') else 'FAILED'}.",
            {"version": rec.version, "consistency": consistency},
        )
        return active

    def require(self) -> ActiveModel:
        am = self.active
        if am is None:
            from ..errors import ModelUnavailable

            raise ModelUnavailable(self.load_error or "No model loaded")
        return am

    def predict(self, records: list[dict[str, Any]]) -> tuple[ActiveModel, pd.DataFrame, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]], dict[str, float]]:
        """Return (model, frame, feature vectors, scores, explanations, timings) for API-validated records."""
        am = self.require()
        t0 = time.perf_counter()
        frame = serving_frame(records)
        Xt = am.model.transform(frame)
        t1 = time.perf_counter()
        scores = am.model.score_transformed(Xt)
        t2 = time.perf_counter()
        explanations = am.model.explain(frame, scores, Xt)
        t3 = time.perf_counter()
        timings = {"preprocessing_ms": (t1 - t0) * 1000, "inference_ms": (t2 - t1) * 1000, "explanation_ms": (t3 - t2) * 1000}
        return am, frame, Xt, scores, explanations, timings


model_service = ModelService()
