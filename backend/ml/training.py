"""Leakage-safe training of the hybrid fraud model (M1/M5/M6).

Protocol
--------
1. Validate the raw dataset against the feature contract.
2. Clean (drop exact duplicates, documented) and split: stratified train / validation / test (70 / 15 / 15).
3. Fit the scaler on the training split only.
4. Logistic Regression: choose ``C`` by stratified K-fold CV on the training split (average precision),
   imbalance handled inside each training fold (class weights or undersampling), refit on the whole
   training split, prior-correct the intercept.
5. Isolation Forest: fit on the training split without labels (``all_train``) or on its legitimate rows
   (``legitimate_train``); its score reference is the same rows.
6. On the validation split only: the LR threshold, the fusion weight ``w`` and the REVIEW / HIGH_RISK thresholds.
7. One final evaluation of LR, Isolation Forest and the hybrid on the untouched test split.
8. Golden examples, a drift reference profile, training feature ranges and explanations are stored with the artifact.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import contract as C
from . import metrics as M
from .data import Splits, prepare_splits, validate_dataframe
from .drift import build_reference_profile
from .features import AMOUNT_TRANSFORMS, TIME_ENCODINGS, build_preprocessor, describe_feature
from .fraud_model import DecisionPolicy, FraudModel, RarityReference, decide, fuse

ProgressFn = Callable[[str, str], None]

MODEL_TYPE = "hybrid_lr_iforest"
MODEL_LABEL = "Hybrid: Logistic Regression + Isolation Forest"


class DataValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
        super().__init__(f"Dataset failed validation: {failed}")
        self.report = report


@dataclass
class TrainConfig:
    # data
    seed: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    drop_duplicates: bool = True
    sample_rows: int | None = None
    time_encoding: str = "cyclical_hour"
    amount_transform: str = "log1p"
    # Logistic Regression
    lr_imbalance: str = "class_weight"  # class_weight | undersample | none
    lr_undersample_ratio: float = 20.0  # legitimate rows kept per fraud row when undersampling
    lr_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    lr_calibration: str = "prior_correction"  # prior_correction | none
    lr_threshold_strategy: str = "f1"  # f1 | f2
    cv_folds: int = 5
    # Isolation Forest
    if_n_estimators: int = 300
    if_max_samples: int = 256
    if_max_features: float = 1.0
    if_reference: str = "all_train"  # all_train | legitimate_train
    if_flag_rate: float = 0.01  # label-free anomaly flag: top 1 % of the training reference
    # fusion + decisions (tuned on validation)
    fusion_weight: float | None = None  # None = choose w on the validation split
    fusion_grid_step: float = 0.05
    high_risk_strategy: str = "f1"  # f1 | precision_target
    high_risk_precision_target: float = 0.8
    review_recall_target: float = 0.9
    max_review_rate: float = 0.01
    if_review_percentile: float | None = None  # optional anomaly review trigger, e.g. 0.999 = top 0.1 %
    # promotion gate + artifacts
    min_validation_average_precision: float = 0.3
    auto_promote: bool = True
    golden_examples: int = 20
    permutation_repeats: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        if "lr_c_grid" in data:
            data["lr_c_grid"] = tuple(float(c) for c in data["lr_c_grid"])
        cfg = cls(**data)
        cfg.check()
        return cfg

    def check(self) -> None:
        if self.time_encoding not in TIME_ENCODINGS:
            raise ValueError(f"time_encoding must be one of {TIME_ENCODINGS}")
        if self.amount_transform not in AMOUNT_TRANSFORMS:
            raise ValueError(f"amount_transform must be one of {AMOUNT_TRANSFORMS}")
        if self.lr_imbalance not in ("class_weight", "undersample", "none"):
            raise ValueError("lr_imbalance must be class_weight, undersample or none")
        if self.lr_calibration not in ("prior_correction", "none"):
            raise ValueError("lr_calibration must be prior_correction or none")
        if self.lr_threshold_strategy not in ("f1", "f2"):
            raise ValueError("lr_threshold_strategy must be f1 or f2")
        if self.if_reference not in ("all_train", "legitimate_train"):
            raise ValueError("if_reference must be all_train or legitimate_train")
        if self.high_risk_strategy not in ("f1", "precision_target"):
            raise ValueError("high_risk_strategy must be f1 or precision_target")
        if not self.lr_c_grid or any(c <= 0 for c in self.lr_c_grid):
            raise ValueError("lr_c_grid must contain positive values")
        if not 2 <= self.cv_folds <= 10:
            raise ValueError("cv_folds must be between 2 and 10")
        if self.fusion_weight is not None and not 0.0 <= self.fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be between 0 and 1")
        if not 0 < self.fusion_grid_step <= 0.5:
            raise ValueError("fusion_grid_step must be in (0, 0.5]")
        for name in ("if_flag_rate", "max_review_rate"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("review_recall_target", "high_risk_precision_target"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.if_review_percentile is not None and not 0.5 <= self.if_review_percentile < 1:
            raise ValueError("if_review_percentile must be in [0.5, 1) or null")
        if self.lr_undersample_ratio < 1:
            raise ValueError("lr_undersample_ratio must be >= 1")
        if not 0 < self.validation_size < 0.5 or not 0 < self.test_size < 0.5:
            raise ValueError("validation_size and test_size must be in (0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lr_c_grid"] = list(self.lr_c_grid)
        return d

    def split_key(self) -> dict[str, Any]:
        """The parameters that determine which rows land in which split."""
        return {k: getattr(self, k) for k in ("seed", "validation_size", "test_size", "drop_duplicates", "sample_rows")}


@dataclass
class TrainingOutcome:
    model: FraudModel
    metadata: dict[str, Any]
    splits: Splits = field(repr=False)


def _noop(stage: str, message: str) -> None:
    return None


# --------------------------------------------------------------------------- building blocks


def make_lr(C_value: float, imbalance: str, seed: int) -> LogisticRegression:
    return LogisticRegression(C=C_value, class_weight="balanced" if imbalance == "class_weight" else None, max_iter=5000, random_state=seed)


def undersample(y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Indices keeping every fraud row and ``ratio`` legitimate rows per fraud (training folds only)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = min(len(neg), int(round(ratio * len(pos))))
    return np.sort(np.concatenate([pos, rng.choice(neg, k, replace=False)]))


def prior_offset(y_fit: np.ndarray, imbalance: str, keep_neg_fraction: float = 1.0) -> float:
    """Logit shift introduced by class weighting / undersampling (King & Zeng prior correction).

    The fitted model behaves as if fraud were ``w1 * keep1 / (w0 * keep0)`` times more common
    than it is; subtracting the log of that factor restores the population prior.
    """
    if imbalance == "class_weight":
        n0, n1 = int(np.sum(y_fit == 0)), int(np.sum(y_fit == 1))
        return float(np.log(n0 / n1))  # balanced weights: w1 / w0 = n0 / n1
    if imbalance == "undersample":
        return float(-np.log(keep_neg_fraction))
    return 0.0


def evaluate_scores(y: np.ndarray, base: dict[str, np.ndarray], policy: DecisionPolicy) -> dict[str, Any]:
    """Metrics for LR, Isolation Forest and the hybrid under a given policy (scores are not re-computed)."""
    hybrid = fuse(base["lr_rarity"], base["if_rarity"], policy.weight)
    dec = decide(hybrid, policy.t_review, policy.t_high, base["anomaly_percentile"], policy.if_review_percentile)
    review_or_high = (dec >= 1).astype(float)
    out = {
        "logistic_regression": M.classification_metrics(y, base["lr_probability"], policy.lr_threshold, probabilistic=True),
        "isolation_forest": M.classification_metrics(y, base["anomaly_percentile"], policy.if_flag_percentile, probabilistic=False),
        "hybrid_high_risk": M.classification_metrics(y, hybrid, policy.t_high, probabilistic=False),
        "hybrid_review_or_high": M.classification_metrics(y, review_or_high, 0.5, probabilistic=False)
        | _ranking(y, hybrid, threshold=policy.t_review),
        "decisions": decision_table(y, dec),
    }
    return out


def _ranking(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Ranking metrics of the hybrid score (the REVIEW band is a decision, not a score)."""
    two = len(np.unique(y)) == 2
    return {
        "threshold": round(float(threshold), 6),
        "roc_auc": round(float(roc_auc_score(y, score)), 4) if two else None,
        "average_precision": round(float(average_precision_score(y, score)), 4) if two else None,
    }


def decision_table(y: np.ndarray, decisions: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(y).astype(int)
    rows = []
    for code, name in enumerate(C.DECISIONS):
        m = decisions == code
        rows.append(
            {
                "decision": name,
                "transactions": int(m.sum()),
                "fraud": int(np.sum(m & (y == 1))),
                "legitimate": int(np.sum(m & (y == 0))),
                "share_of_transactions": round(float(m.mean()) if len(m) else 0.0, 6),
                "share_of_all_fraud": round(float(np.sum(m & (y == 1)) / max(int(y.sum()), 1)), 4),
                "fraud_rate_in_band": round(float(np.sum(m & (y == 1)) / m.sum()), 4) if m.any() else None,
            }
        )
    return rows


def tune_policy(y_va: np.ndarray, base_va: dict[str, np.ndarray], config: TrainConfig) -> tuple[DecisionPolicy, dict[str, Any]]:
    """Choose w, the LR threshold and the REVIEW / HIGH_RISK thresholds on validation data only."""
    beta = 2.0 if config.lr_threshold_strategy == "f2" else 1.0
    lr_threshold = M.best_fbeta_threshold(y_va, base_va["lr_probability"], beta)

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, config.fusion_grid_step), 4)
    weight_curve = []
    for w in grid:
        h = fuse(base_va["lr_rarity"], base_va["if_rarity"], float(w))
        weight_curve.append({"weight_lr": float(w), "average_precision": round(float(average_precision_score(y_va, h)), 4), "roc_auc": round(float(roc_auc_score(y_va, h)), 4)})
    if config.fusion_weight is None:
        # Highest validation AP; ties go to the larger LR weight (the simpler, supervised component).
        best = max(weight_curve, key=lambda d: (d["average_precision"], d["weight_lr"]))
        weight = best["weight_lr"]
        weight_source = "tuned on validation (max average precision)"
    else:
        weight = float(config.fusion_weight)
        weight_source = "fixed in configuration"

    hybrid = fuse(base_va["lr_rarity"], base_va["if_rarity"], weight)
    notes: list[str] = []
    if config.high_risk_strategy == "precision_target":
        t_high = M.threshold_for_precision(y_va, hybrid, config.high_risk_precision_target)
        if t_high is None:
            t_high = M.best_f1_threshold(y_va, hybrid)
            notes.append(f"Precision target {config.high_risk_precision_target} unreachable on validation; HIGH_RISK uses the F1-optimal threshold.")
    else:
        t_high = M.best_f1_threshold(y_va, hybrid)
    review = M.threshold_for_recall(y_va, hybrid, config.review_recall_target, config.max_review_rate)
    t_review = review["threshold"]
    if t_review > t_high:
        notes.append("The REVIEW threshold would exceed the HIGH_RISK threshold, so the REVIEW band is empty.")
        t_review = t_high
    policy = DecisionPolicy(
        weight=weight,
        t_review=float(t_review),
        t_high=float(t_high),
        lr_threshold=float(lr_threshold),
        if_flag_percentile=float(1.0 - config.if_flag_rate),
        if_review_percentile=config.if_review_percentile,
    )
    info = {
        "weight_source": weight_source,
        "weight_curve": weight_curve,
        "lr_threshold_rule": f"maximise F{int(beta)} of the LR probability on validation",
        "high_risk_rule": (
            f"lowest hybrid score with validation precision >= {config.high_risk_precision_target}"
            if config.high_risk_strategy == "precision_target"
            else "hybrid score maximising validation F1"
        ),
        "review_rule": f"highest hybrid score keeping validation recall >= {config.review_recall_target}, "
        f"capped so at most {config.max_review_rate:.2%} of transactions reach REVIEW or above",
        "review_binding_constraint": review["binding"],
        "if_flag_rule": f"anomaly score in the top {config.if_flag_rate:.2%} of the training reference (label-free)",
        "if_review_trigger": (
            f"REVIEW also when the anomaly percentile >= {config.if_review_percentile}" if config.if_review_percentile is not None else "disabled"
        ),
        "notes": notes,
    }
    return policy, info


# --------------------------------------------------------------------------- training


def train_model(raw: pd.DataFrame, config: TrainConfig, progress: ProgressFn = _noop) -> TrainingOutcome:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    progress("validation", "Validating dataset against the feature contract")
    report = validate_dataframe(raw, drop_duplicates=config.drop_duplicates)
    if report["status"] == "failed":
        raise DataValidationError(report)
    timings["validation_s"] = time.perf_counter() - t0

    progress("preprocessing", "Cleaning (duplicates dropped before splitting) and splitting (stratified)")
    splits = prepare_splits(
        raw,
        seed=config.seed,
        validation=config.validation_size,
        test=config.test_size,
        drop_duplicates=config.drop_duplicates,
        sample_rows=config.sample_rows,
    )
    for name, part in (("training", splits.train), ("validation", splits.validation), ("test", splits.test)):
        if part[C.TARGET].sum() < 2:
            raise ValueError(f"The {name} split has fewer than 2 fraud rows; use more data")
    features = list(C.FEATURE_NAMES)
    X_tr, y_tr = splits.train[features], splits.train[C.TARGET].to_numpy()
    X_va, y_va = splits.validation[features], splits.validation[C.TARGET].to_numpy()
    X_te, y_te = splits.test[features], splits.test[C.TARGET].to_numpy()

    pre = build_preprocessor(config.time_encoding, config.amount_transform)
    pre.fit(X_tr)
    Xt_tr, Xt_va, Xt_te = (np.asarray(pre.transform(X), float) for X in (X_tr, X_va, X_te))
    names = list(pre.named_steps["engineer"].output_names())

    # ---------------------------------------------------------------- Logistic Regression: C by CV on train
    t_sel = time.perf_counter()
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_results: list[dict[str, Any]] = []
    for i, c_value in enumerate(config.lr_c_grid, 1):
        progress("training", f"Logistic Regression: {config.cv_folds}-fold CV for C={c_value} ({i}/{len(config.lr_c_grid)})")
        aps, aucs, fit_s = [], [], []
        for f_tr, f_va in cv.split(X_tr, y_tr):
            fold_pre = clone(pre).fit(X_tr.iloc[f_tr])  # scaler re-fitted inside the fold: no leakage
            Xf_tr = np.asarray(fold_pre.transform(X_tr.iloc[f_tr]), float)
            Xf_va = np.asarray(fold_pre.transform(X_tr.iloc[f_va]), float)
            yf = y_tr[f_tr]
            idx = undersample(yf, config.lr_undersample_ratio, config.seed) if config.lr_imbalance == "undersample" else np.arange(len(yf))
            ts = time.perf_counter()
            m = make_lr(c_value, config.lr_imbalance, config.seed).fit(Xf_tr[idx], yf[idx])
            fit_s.append(time.perf_counter() - ts)
            s = m.decision_function(Xf_va)
            aps.append(average_precision_score(y_tr[f_va], s))
            aucs.append(roc_auc_score(y_tr[f_va], s))
        cv_results.append(
            {
                "C": c_value,
                "average_precision": {"mean": round(float(np.mean(aps)), 4), "std": round(float(np.std(aps)), 4), "folds": [round(float(a), 4) for a in aps]},
                "roc_auc": {"mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4), "folds": [round(float(a), 4) for a in aucs]},
                "fit_time_s": round(float(np.mean(fit_s)), 3),
            }
        )
    best = max(cv_results, key=lambda r: (r["average_precision"]["mean"], -r["C"]))
    best_c = float(best["C"])
    timings["lr_model_selection_s"] = time.perf_counter() - t_sel

    progress("training", f"Refitting Logistic Regression (C={best_c}, {config.lr_imbalance}) on the full training split")
    t_fit = time.perf_counter()
    if config.lr_imbalance == "undersample":
        idx = undersample(y_tr, config.lr_undersample_ratio, config.seed)
        keep_neg = float(np.sum(y_tr[idx] == 0) / max(np.sum(y_tr == 0), 1))
    else:
        idx = np.arange(len(y_tr))
        keep_neg = 1.0
    lr = make_lr(best_c, config.lr_imbalance, config.seed).fit(Xt_tr[idx], y_tr[idx])
    offset = prior_offset(y_tr[idx], config.lr_imbalance, keep_neg) if config.lr_calibration == "prior_correction" else 0.0
    timings["lr_fit_s"] = time.perf_counter() - t_fit

    # ---------------------------------------------------------------- Isolation Forest (labels unused for fitting)
    progress("training", f"Fitting Isolation Forest ({config.if_n_estimators} trees) on the training split ({config.if_reference})")
    t_if = time.perf_counter()
    if_rows = np.arange(len(y_tr)) if config.if_reference == "all_train" else np.flatnonzero(y_tr == 0)
    iforest = IsolationForest(
        n_estimators=config.if_n_estimators,
        max_samples=min(config.if_max_samples, len(if_rows)),
        max_features=config.if_max_features,
        random_state=config.seed,
        n_jobs=-1,
    ).fit(Xt_tr[if_rows])
    timings["if_fit_s"] = time.perf_counter() - t_if

    progress("training", "Scoring the training split to build the rarity references")
    t_ref = time.perf_counter()
    lr_logit_tr = lr.decision_function(Xt_tr) - offset
    anomaly_tr = -iforest.score_samples(Xt_tr)
    lr_ref = RarityReference(lr_logit_tr)
    if_ref = RarityReference(anomaly_tr[if_rows])
    timings["references_s"] = time.perf_counter() - t_ref

    feature_ranges = {f: (float(X_tr[f].min()), float(X_tr[f].max())) for f in features}
    provisional = DecisionPolicy(weight=1.0, t_review=1.0, t_high=1.0, lr_threshold=0.5, if_flag_percentile=1 - config.if_flag_rate)
    model = FraudModel(pre, lr, offset, iforest, lr_ref, if_ref, provisional, feature_ranges)

    # ---------------------------------------------------------------- validation-only tuning
    progress("evaluation", "Tuning the fusion weight and decision thresholds on the validation split")
    t_tune = time.perf_counter()
    base_va = model.score_transformed(Xt_va)
    policy, tuning = tune_policy(y_va, base_va, config)
    model.policy = policy
    timings["tuning_s"] = time.perf_counter() - t_tune
    scores_va = model.score_transformed(Xt_va)

    # ---------------------------------------------------------------- final test evaluation (once)
    progress("evaluation", "Evaluating LR, Isolation Forest and the hybrid once on the held-out test split")
    scores_te = model.score_transformed(Xt_te)
    validation_metrics = evaluate_scores(y_va, scores_va, policy)
    test_metrics = evaluate_scores(y_te, scores_te, policy)

    uncorrected_va = 1.0 / (1.0 + np.exp(-np.clip(scores_va["lr_logit"] + offset, -700, 700)))
    calibration = {
        "method": config.lr_calibration,
        "logit_offset": round(offset, 6),
        "validation": M.calibration_points(y_va, scores_va["lr_probability"]),
        "validation_without_correction": M.calibration_points(y_va, uncorrected_va),
        "test": M.calibration_points(y_te, scores_te["lr_probability"]),
    }
    curves = {
        "test_pr": {
            "logistic_regression": M.pr_points(y_te, scores_te["lr_probability"]),
            "isolation_forest": M.pr_points(y_te, scores_te["anomaly_score"]),
            "hybrid": M.pr_points(y_te, scores_te["hybrid_score"]),
        },
        "test_roc": {
            "logistic_regression": M.roc_points(y_te, scores_te["lr_probability"]),
            "isolation_forest": M.roc_points(y_te, scores_te["anomaly_score"]),
            "hybrid": M.roc_points(y_te, scores_te["hybrid_score"]),
        },
        "validation_threshold_sweep": {
            "hybrid": M.threshold_sweep(y_va, scores_va["hybrid_score"]),
            "logistic_regression": M.threshold_sweep(y_va, scores_va["lr_probability"]),
        },
        "fusion_weight": tuning["weight_curve"],
    }

    # ---------------------------------------------------------------- explanations of the LR component
    progress("evaluation", "Computing coefficient and permutation importance (validation split)")
    coefficients = sorted(
        ({"feature": n, "label": describe_feature(n), "coefficient": round(float(c), 5)} for n, c in zip(names, lr.coef_[0])),
        key=lambda d: -abs(d["coefficient"]),
    )
    rng = np.random.default_rng(config.seed)
    base_ap = average_precision_score(y_va, scores_va["lr_logit"])
    permutation = []
    for f in features:
        drops = []
        for _ in range(config.permutation_repeats):
            Xp = X_va.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base_ap - average_precision_score(y_va, lr.decision_function(np.asarray(pre.transform(Xp), float))))
        permutation.append({"feature": f, "importance": round(float(np.mean(drops)), 5), "std": round(float(np.std(drops)), 5)})
    permutation.sort(key=lambda d: -d["importance"])

    # ---------------------------------------------------------------- golden examples + references
    g_rng = np.random.default_rng(config.seed + 1)
    pos_idx, neg_idx = np.flatnonzero(y_te == 1), np.flatnonzero(y_te == 0)
    n_pos = min(len(pos_idx), max(1, config.golden_examples // 4))
    golden_idx = np.concatenate([g_rng.choice(pos_idx, n_pos, replace=False), g_rng.choice(neg_idx, min(len(neg_idx), config.golden_examples - n_pos), replace=False)])
    golden_X = X_te.iloc[golden_idx]
    golden_scores = model.score(golden_X)
    golden = {
        "inputs": [{k: float(v) for k, v in row.items()} for row in golden_X.to_dict(orient="records")],
        "feature_vectors": np.round(model.transform(golden_X), 8).tolist(),
        "lr_probability": [float(v) for v in golden_scores["lr_probability"]],
        "anomaly_score": [float(v) for v in golden_scores["anomaly_score"]],
        "hybrid_score": [float(v) for v in golden_scores["hybrid_score"]],
        "decision": [int(v) for v in golden_scores["decision"]],
    }
    reference = build_reference_profile(
        X_tr,
        {"lr_probability": scores_va["lr_probability"], "anomaly_percentile": scores_va["anomaly_percentile"], "hybrid_score": scores_va["hybrid_score"]},
    )

    timings["total_s"] = time.perf_counter() - t0
    metadata: dict[str, Any] = {
        "task": "binary fraud detection: Class (1 = fraud) with a three-level risk decision",
        "model_type": MODEL_TYPE,
        "model_label": MODEL_LABEL,
        "components": {
            "logistic_regression": {
                "C": best_c,
                "imbalance": config.lr_imbalance,
                "class_weight": "balanced" if config.lr_imbalance == "class_weight" else None,
                "training_rows": int(len(idx)),
                "training_fraud_rows": int(y_tr[idx].sum()),
                "intercept": float(lr.intercept_[0]),
                "prior_correction_offset": round(offset, 6),
                "params": {k: _jsonable(v) for k, v in lr.get_params().items()},
            },
            "isolation_forest": {
                "reference": config.if_reference,
                "fit_rows": int(len(if_rows)),
                "labels_used_for_fitting": config.if_reference == "legitimate_train",
                "params": {k: _jsonable(v) for k, v in iforest.get_params().items()},
                "anomaly_score": "-score_samples(x); higher = easier to isolate (not a probability)",
            },
            "references": {"logistic_regression": lr_ref.summary(), "isolation_forest": if_ref.summary()},
        },
        "policy": policy.to_dict(),
        "tuning": {k: v for k, v in tuning.items() if k != "weight_curve"},
        "normalisation": "rarity(s) = -log10(max(share of training-reference scores > s, 1/(n_ref+1))) / log10(n_ref+1)",
        "fusion_formula": "hybrid = w * rarity_LR + (1 - w) * rarity_IF",
        "decision_rules": {
            "HIGH_RISK": f"hybrid >= {policy.t_high:.6f}",
            "REVIEW": f"{policy.t_review:.6f} <= hybrid < {policy.t_high:.6f}"
            + (f", or anomaly percentile >= {policy.if_review_percentile}" if policy.if_review_percentile is not None else ""),
            "LOW_RISK": f"hybrid < {policy.t_review:.6f}",
        },
        "config": config.to_dict(),
        "feature_set": f"time:{config.time_encoding}|amount:{config.amount_transform}",
        "features": features,
        "feature_schema": C.feature_schema(),
        "transformed_features": names,
        "feature_ranges": {k: list(v) for k, v in feature_ranges.items()},
        "excluded_columns": C.EXCLUDED_COLUMNS,
        "class_labels": C.CLASS_LABELS,
        "data_validation": report,
        "splits": splits.describe(),
        "model_selection": {
            "protocol": f"stratified {config.cv_folds}-fold CV of Logistic Regression C on the training split; scaler and imbalance handling inside each fold",
            "metric": "average_precision",
            "selected_C": best_c,
            "results": cv_results,
        },
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "calibration": calibration,
        "curves": curves,
        "feature_importance": {"lr_coefficients": coefficients, "lr_permutation_validation_ap": permutation},
        "reference_profile": reference,
        "golden_examples": golden,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    progress("evaluation", "Training complete")
    return TrainingOutcome(model=model, metadata=metadata, splits=splits)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return str(v)


def headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact per-component summary for the registry and MLflow."""
    keep = ("precision", "recall", "f1", "average_precision", "roc_auc", "false_positives", "false_negatives", "alert_rate", "threshold", "confusion_matrix", "support", "positives", "brier")
    return {k: {m: v.get(m) for m in keep if m in v} for k, v in metrics.items() if k != "decisions"} | {"decisions": metrics.get("decisions")}
