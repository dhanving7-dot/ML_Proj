"""Evaluation laboratory: compare LR, Isolation Forest and the hybrid, and try other decision policies.

What-if policies are evaluated on the active model's **validation** split only. The test split is
reported exactly as it was evaluated once at training time and is never re-scored here, so the
explorer cannot be used to tune on test data.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ml import contract as C
from ml import metrics as M
from ml.data import load_csv, prepare_splits
from ml.fraud_model import DecisionPolicy, fuse
from ml.training import evaluate_scores

from ..errors import InvalidInput, NotFound
from ..orm import Dataset
from .dataset_service import split_params
from .model_service import ActiveModel, model_service

_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
_lock = threading.Lock()


def _validation_scores(session: Session, am: ActiveModel) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recreate the model's validation split from its own dataset + split parameters and score it once."""
    with _lock:
        if am.version in _cache:
            return _cache[am.version]
    ds_info = am.metadata.get("dataset") or {}
    ds = session.get(Dataset, ds_info.get("id")) if ds_info.get("id") is not None else None
    if ds is None or ds.sha256 != ds_info.get("sha256"):
        raise NotFound("The dataset this model was trained on is no longer registered, so its validation split cannot be rebuilt")
    p = split_params(am.metadata)
    splits = prepare_splits(
        load_csv(ds.path), seed=p["seed"], validation=p["validation_size"], test=p["test_size"], drop_duplicates=p["drop_duplicates"], sample_rows=p["sample_rows"]
    )
    val = splits.validation
    y = val[C.TARGET].to_numpy()
    scores = am.model.score(val[am.features])
    expected = am.metadata["splits"]["validation"]
    if len(y) != expected["rows"] or int(y.sum()) != expected["fraud_rows"]:
        raise InvalidInput("The rebuilt validation split does not match the one recorded at training time")
    with _lock:
        _cache.clear()
        _cache[am.version] = (y, scores)
    return y, scores


def summary(am: ActiveModel) -> dict[str, Any]:
    md = am.metadata
    return {
        "model_version": am.version,
        "policy": md["policy"],
        "tuning": md["tuning"],
        "normalisation": md["normalisation"],
        "fusion_formula": md["fusion_formula"],
        "decision_rules": md["decision_rules"],
        "splits": md["splits"],
        "metrics": md["metrics"],
        "calibration": md["calibration"],
        "curves": md["curves"],
        "model_selection": md["model_selection"],
        "components": md["components"],
        "methodology": [
            "Stratified 70/15/15 split on Class with a fixed seed after dropping exact duplicate rows.",
            "The scaler, Logistic Regression and Isolation Forest are fitted on the training split only; LR's C is chosen by stratified CV inside the training split.",
            "The Isolation Forest never sees labels when fitted on all training rows; its flag is the top share of training anomaly scores.",
            "The LR threshold, the fusion weight and the REVIEW / HIGH_RISK thresholds are chosen on the validation split.",
            "The test split is scored once, after every choice is fixed; these numbers are shown read-only.",
        ],
    }


def explore(session: Session, body: dict[str, Any]) -> dict[str, Any]:
    am = model_service.require()
    if body["review_threshold"] > body["high_risk_threshold"]:
        raise InvalidInput("The REVIEW threshold must not exceed the HIGH_RISK threshold")
    y, base = _validation_scores(session, am)
    policy = DecisionPolicy(
        weight=float(body["weight"]),
        t_review=float(body["review_threshold"]),
        t_high=float(body["high_risk_threshold"]),
        lr_threshold=float(body["lr_threshold"]),
        if_flag_percentile=1.0 - float(body["if_flag_rate"]),
        if_review_percentile=body.get("if_review_percentile"),
    )
    hybrid = fuse(base["lr_rarity"], base["if_rarity"], policy.weight)
    return {
        "split": "validation",
        "model_version": am.version,
        "rows": int(len(y)),
        "fraud_rows": int(y.sum()),
        "policy": policy.to_dict(),
        "active_policy": am.policy,
        "metrics": evaluate_scores(y, base, policy),
        "curves": {
            "pr": {
                "logistic_regression": M.pr_points(y, base["lr_probability"]),
                "isolation_forest": M.pr_points(y, base["anomaly_score"]),
                "hybrid": M.pr_points(y, hybrid),
            },
            "hybrid_threshold_sweep": M.threshold_sweep(y, hybrid),
        },
        "note": "Computed on the validation split. The test split is never re-scored with a what-if policy.",
    }
