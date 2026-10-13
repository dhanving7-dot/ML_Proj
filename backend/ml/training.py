"""Leakage-safe training pipeline (M1/M5/M6).

Protocol
--------
1. Validate the raw dataset against the feature contract.
2. Clean and split: stratified train / validation / test (70 / 15 / 15 by default).
3. Model selection: stratified K-fold cross-validation of every candidate on the *train* split only.
4. Refit the selected candidate on the full train split.
5. Optional probability calibration and decision-threshold tuning on the *validation* split
   (split in half when calibrating, so calibration and threshold tuning use different rows).
6. One final evaluation on the untouched *test* split.
7. Golden examples, a drift reference profile and permutation importances are stored with the artifact.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import contract as C
from . import metrics as M
from .data import Splits, clean_for_training, split_data, validate_dataframe
from .drift import build_reference_profile
from .estimators import CLASSIFIER_LABELS, available_classifiers, make_classifier
from .features import ENGINEERED_FEATURES, build_pipeline, transformed_feature_names

ProgressFn = Callable[[str, str], None]

DEFAULT_CANDIDATES = ("logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm")


class DataValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
        super().__init__(f"Dataset failed validation: {failed}")
        self.report = report


@dataclass
class TrainConfig:
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    feature_set: str = "full"
    engineered_features: bool = True
    cv_folds: int = 5
    selection_metric: str = "roc_auc"
    calibration: str = "none"  # none | sigmoid | isotonic
    threshold_strategy: str = "f1"  # f1 | fixed
    fixed_threshold: float = 0.5
    seed: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    sample_rows: int | None = None
    min_validation_roc_auc: float = 0.75
    permutation_repeats: int = 5
    importance_rows: int = 3000
    golden_examples: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        if "candidates" in data:
            data["candidates"] = tuple(data["candidates"])
        cfg = cls(**data)
        cfg.check()
        return cfg

    def check(self) -> None:
        if self.feature_set not in C.FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {list(C.FEATURE_SETS)}")
        if self.calibration not in ("none", "sigmoid", "isotonic"):
            raise ValueError("calibration must be none, sigmoid or isotonic")
        if self.threshold_strategy not in ("f1", "fixed"):
            raise ValueError("threshold_strategy must be f1 or fixed")
        if self.selection_metric not in ("roc_auc", "f1", "accuracy", "neg_log_loss"):
            raise ValueError("selection_metric must be roc_auc, f1, accuracy or neg_log_loss")
        if not 2 <= self.cv_folds <= 10:
            raise ValueError("cv_folds must be between 2 and 10")
        if not self.candidates:
            raise ValueError("At least one candidate model is required")
        unknown = [c for c in self.candidates if c not in CLASSIFIER_LABELS]
        if unknown:
            raise ValueError(f"Unknown candidates: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = list(self.candidates)
        return d


@dataclass
class TrainingOutcome:
    model: Any
    metadata: dict[str, Any]
    splits: Splits = field(repr=False)


def base_pipeline(model: Any) -> Pipeline:
    """Return the underlying preprocessing+estimator Pipeline, unwrapping calibration."""
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV):
        est = model.estimator
        return est.estimator if isinstance(est, FrozenEstimator) else est
    raise TypeError(f"Unsupported model type {type(model).__name__}")


def _noop(stage: str, message: str) -> None:
    return None


def train_model(raw: pd.DataFrame, config: TrainConfig, progress: ProgressFn = _noop) -> TrainingOutcome:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    progress("validation", "Validating dataset against the feature contract")
    report = validate_dataframe(raw)
    if report["status"] == "failed":
        raise DataValidationError(report)
    timings["validation_s"] = time.perf_counter() - t0

    progress("preprocessing", "Cleaning and splitting data (stratified)")
    df = clean_for_training(raw)
    if config.sample_rows and config.sample_rows < len(df):
        df, _ = train_test_split(df, train_size=config.sample_rows, stratify=df[C.TARGET], random_state=config.seed)
        df = df.reset_index(drop=True)
    splits = split_data(df, seed=config.seed, validation=config.validation_size, test=config.test_size)
    features = list(C.FEATURE_SETS[config.feature_set])
    X_tr, y_tr = splits.train[features], splits.train[C.TARGET].to_numpy()
    X_va, y_va = splits.validation[features], splits.validation[C.TARGET].to_numpy()
    X_te, y_te = splits.test[features], splits.test[C.TARGET].to_numpy()

    # ---------------------------------------------------------------- model selection (train only)
    available = set(available_classifiers())
    skipped = [c for c in config.candidates if c not in available]
    candidates = [c for c in config.candidates if c in available]
    if not candidates:
        raise ValueError(f"None of the requested candidates are installed: {list(config.candidates)}")
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy", "neg_log_loss": "neg_log_loss"}
    cv_results: list[dict[str, Any]] = []
    t_sel = time.perf_counter()
    for i, name in enumerate(candidates, 1):
        progress("training", f"Cross-validating candidate {i}/{len(candidates)}: {CLASSIFIER_LABELS[name]}")
        pipe = build_pipeline(make_classifier(name, seed=config.seed), features, engineered=config.engineered_features)
        res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=None)
        row: dict[str, Any] = {"model": name, "label": CLASSIFIER_LABELS[name]}
        for metric in scoring:
            scores = res[f"test_{metric}"]
            row[metric] = {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4), "folds": [round(float(s), 4) for s in scores]}
        row["fit_time_s"] = round(float(res["fit_time"].mean()), 3)
        cv_results.append(row)
    timings["model_selection_s"] = time.perf_counter() - t_sel
    best = max(cv_results, key=lambda r: r[config.selection_metric]["mean"])
    best_name = best["model"]

    progress("training", f"Refitting {CLASSIFIER_LABELS[best_name]} on the full training split")
    t_fit = time.perf_counter()
    pipeline = build_pipeline(make_classifier(best_name, seed=config.seed), features, engineered=config.engineered_features)
    pipeline.fit(X_tr, y_tr)
    timings["final_fit_s"] = time.perf_counter() - t_fit

    # ---------------------------------------------------------------- calibration + threshold (validation only)
    progress("evaluation", "Calibrating probabilities and tuning the decision threshold on validation data")
    uncalibrated_val = pipeline.predict_proba(X_va)[:, 1]
    model: Any = pipeline
    if config.calibration != "none":
        X_cal, X_thr, y_cal, y_thr = train_test_split(X_va, y_va, test_size=0.5, stratify=y_va, random_state=config.seed)
        model = CalibratedClassifierCV(FrozenEstimator(pipeline), method=config.calibration)
        model.fit(X_cal, y_cal)
        thr_proba = model.predict_proba(X_thr)[:, 1]
        y_thr_eval = y_thr
        calibration_rows = {"calibration": int(len(y_cal)), "threshold_tuning": int(len(y_thr))}
    else:
        thr_proba = uncalibrated_val
        y_thr_eval = y_va
        calibration_rows = {"calibration": 0, "threshold_tuning": int(len(y_va))}
    threshold = (
        M.best_f1_threshold(y_thr_eval, thr_proba) if config.threshold_strategy == "f1" else float(config.fixed_threshold)
    )
    val_proba = model.predict_proba(X_va)[:, 1]
    validation_metrics = M.classification_metrics(y_va, val_proba, threshold)

    # ---------------------------------------------------------------- final test evaluation (once)
    progress("evaluation", "Evaluating once on the held-out test split")
    test_proba = model.predict_proba(X_te)[:, 1]
    test_metrics = M.classification_metrics(y_te, test_proba, threshold)
    test_metrics_at_05 = M.classification_metrics(y_te, test_proba, 0.5)

    # ---------------------------------------------------------------- explanations + references
    progress("evaluation", "Computing permutation importance on validation data")
    n_imp = min(config.importance_rows, len(X_va))
    imp_idx = np.random.default_rng(config.seed).choice(len(X_va), n_imp, replace=False)
    perm = permutation_importance(
        model, X_va.iloc[imp_idx], y_va[imp_idx], scoring="roc_auc", n_repeats=config.permutation_repeats, random_state=config.seed, n_jobs=1
    )
    permutation = sorted(
        (
            {"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
            for f, m, s in zip(features, perm.importances_mean, perm.importances_std)
        ),
        key=lambda r: -r["importance"],
    )
    impurity: list[dict[str, Any]] | None = None
    est = pipeline.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        names = transformed_feature_names(pipeline)
        impurity = sorted(
            ({"feature": n, "importance": round(float(v), 5)} for n, v in zip(names, est.feature_importances_)),
            key=lambda r: -r["importance"],
        )[:30]

    golden_idx = np.random.default_rng(config.seed + 1).choice(len(X_te), min(config.golden_examples, len(X_te)), replace=False)
    golden_X = X_te.iloc[golden_idx]
    golden = {
        "inputs": [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
            for row in golden_X.astype(object).to_dict(orient="records")
        ],
        "probabilities": [float(p) for p in model.predict_proba(golden_X)[:, 1]],
        "feature_vectors": _vector_digest(pipeline, golden_X),
    }
    reference = build_reference_profile(X_tr, val_proba)

    timings["total_s"] = time.perf_counter() - t0
    metadata: dict[str, Any] = {
        "task": "binary classification: PlacementStatus (1 = Placed)",
        "model_type": best_name,
        "model_label": CLASSIFIER_LABELS[best_name],
        "estimator_params": {k: _jsonable(v) for k, v in est.get_params().items()},
        "config": config.to_dict(),
        "feature_set": config.feature_set,
        "features": features,
        "feature_schema": C.feature_schema(config.feature_set),
        "engineered_features": list(ENGINEERED_FEATURES) if config.engineered_features else [],
        "transformed_features": transformed_feature_names(pipeline),
        "excluded_columns": C.EXCLUDED_COLUMNS,
        "class_labels": C.CLASS_LABELS,
        "threshold": round(threshold, 4),
        "calibration": {"method": config.calibration, "rows": calibration_rows},
        "data_validation": report,
        "splits": splits.describe(),
        "candidates_skipped": skipped,
        "model_selection": {
            "protocol": f"stratified {config.cv_folds}-fold cross-validation on the training split",
            "metric": config.selection_metric,
            "selected": best_name,
            "results": cv_results,
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "test_at_0_5": test_metrics_at_05,
            "validation_uncalibrated_roc_auc": round(float(M.roc_auc_score(y_va, uncalibrated_val)), 4),
        },
        "curves": {
            "test_roc": M.roc_points(y_te, test_proba),
            "test_pr": M.pr_points(y_te, test_proba),
            "test_calibration": M.calibration_points(y_te, test_proba),
            "validation_calibration_uncalibrated": M.calibration_points(y_va, uncalibrated_val),
            "validation_threshold_sweep": M.threshold_sweep(y_va, val_proba),
        },
        "feature_importance": {"permutation_validation_roc_auc": permutation, "impurity_top30": impurity},
        "reference_profile": reference,
        "golden_examples": golden,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    progress("evaluation", "Training complete")
    return TrainingOutcome(model=model, metadata=metadata, splits=splits)


def _vector_digest(pipeline: Pipeline, X: pd.DataFrame) -> list[list[float]]:
    """Rounded transformed feature vectors for golden examples (used to diff train vs serve paths)."""
    vec = np.asarray(pipeline[:-1].transform(X), dtype=float)
    return np.round(vec, 8).tolist()


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return str(v)


def evaluate_on(model: Any, X: pd.DataFrame, y: np.ndarray, threshold: float) -> dict[str, Any]:
    proba = model.predict_proba(X)[:, 1]
    return M.classification_metrics(y, proba, threshold)
