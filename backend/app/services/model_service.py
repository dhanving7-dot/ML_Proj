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
from sklearn.pipeline import Pipeline
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.features import coerce_features
from ml.registry import load_artifact, read_metadata
from ml.training import base_pipeline

from ..db import utcnow
from ..orm import ModelVersion
from ..schemas import StudentFeatures
from .events import record_event


@dataclass
class ActiveModel:
    version: str
    model_type: str
    model: Any
    metadata: dict[str, Any]
    artifact_path: str
    loaded_at: datetime
    consistency: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def serving_frame(records: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    """The serving path: API-validated records -> contract-coerced frame (same function training uses)."""
    return coerce_features(records, features)


def golden_check(model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Replay the golden examples stored at training time through the *serving* path.

    Validates each stored raw input with the API schema, rebuilds the feature vector with the loaded
    artifact and compares both the feature vector and the probability with what training produced.
    """
    golden = metadata.get("golden_examples") or {}
    inputs = golden.get("inputs") or []
    if not inputs:
        return {"passed": False, "reason": "No golden examples stored with this model"}
    features = list(metadata["features"])
    full = [{**{k: None for k in StudentFeatures.model_fields}, **row} for row in inputs]
    validated = [StudentFeatures.model_validate(_fill_required(rec)).model_dump() for rec in full]
    frame = serving_frame(validated, features)
    proba = model.predict_proba(frame)[:, 1]
    vectors = np.asarray(base_pipeline(model)[:-1].transform(frame), float)
    expected_p = np.asarray(golden["probabilities"], float)
    expected_v = np.asarray(golden.get("feature_vectors", vectors), float)
    p_diff = float(np.max(np.abs(proba - expected_p)))
    v_diff = float(np.max(np.abs(np.round(vectors, 8) - expected_v)))
    return {
        "passed": bool(p_diff < 1e-9 and v_diff < 1e-6),
        "examples": len(inputs),
        "max_probability_diff": p_diff,
        "max_feature_vector_diff": v_diff,
        "checked_at": utcnow().isoformat(),
    }


def _fill_required(rec: dict[str, Any]) -> dict[str, Any]:
    """Golden inputs only contain the model's feature set; fill unused contract fields with valid placeholders.

    Placeholders never reach the model because ``serving_frame`` selects only the model's features.
    """
    out = dict(rec)
    for name, fld in StudentFeatures.model_fields.items():
        if out.get(name) is None and fld.is_required():
            out[name] = _placeholder(name)
    return out


def _placeholder(name: str) -> Any:
    from ml import contract as C

    f = C.FEATURES_BY_NAME[name]
    if isinstance(f, C.NumericFeature):
        return int(f.minimum) if f.integer else float(f.minimum)
    return f.categories[0]


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: ActiveModel | None = None
        self.load_error: str | None = "Model not loaded yet"

    @property
    def active(self) -> ActiveModel | None:
        with self._lock:
            return self._active

    def sync(self, session: Session, min_interval_s: float = 5.0) -> None:
        """Reload if another process (CLI / DVC pipeline) activated a different version.

        Throttled so the registry query runs at most every ``min_interval_s`` seconds.
        """
        now = time.monotonic()
        if now - getattr(self, "_last_sync", 0.0) < min_interval_s:
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

    def predict(self, records: list[dict[str, Any]]) -> tuple[ActiveModel, np.ndarray, np.ndarray, dict[str, float]]:
        """Return (model, probabilities, feature vectors, timings) for API-validated records."""
        am = self.active
        if am is None:
            from ..errors import ModelUnavailable

            raise ModelUnavailable(self.load_error or "No model loaded")
        t0 = time.perf_counter()
        frame = serving_frame(records, am.features)
        pipe = base_pipeline(am.model)
        Xt = np.asarray(pipe[:-1].transform(frame), float)
        t1 = time.perf_counter()
        if isinstance(am.model, Pipeline):
            proba = pipe[-1].predict_proba(Xt)[:, 1]
        else:  # calibrated wrapper: run the full model so calibration is applied
            proba = am.model.predict_proba(frame)[:, 1]
        t2 = time.perf_counter()
        return am, proba, Xt, {"preprocessing_ms": (t1 - t0) * 1000, "inference_ms": (t2 - t1) * 1000}


model_service = ModelService()
