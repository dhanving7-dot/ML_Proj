"""M5: validation protocols, metrics, calibration, hyperparameter search, learning curves, statistical tests.

Every score here is average precision (PR-AUC) or ROC-AUC because accuracy is uninformative at a
0.17 % fraud rate. Logistic regression trains on the full training split; tree models train on
every training fraud plus an undersampled set of legitimate rows (stated in each result).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)

from .. import contract as C
from .. import metrics as M
from ..estimators import CLASSIFIER_LABELS, make_classifier, optional_available
from ..features import build_pipeline
from .common import LabData, r

FEATS = list(C.FEATURE_NAMES)
ENRICHED_LEGIT = 20000


def _pipe(model: str, **params: Any):
    if model == "logistic_regression":
        params = {"class_weight": "balanced", **params}
    return build_pipeline(make_classifier(model, seed=0, **params))


def _train_frame(data: LabData, model: str):
    """Logistic regression is fast enough for the whole training split; trees use an enriched sample."""
    if model == "logistic_regression":
        return data.train, "full training split (real fraud rate)"
    df = data.enriched("train", ENRICHED_LEGIT)
    return df, f"every training fraud + {ENRICHED_LEGIT:,} legitimate rows (fraud rate {df[C.TARGET].mean():.2%})"


def cross_validation(data: LabData, k: int = 5, model: str = "logistic_regression", sample_size: int = 20000) -> dict[str, Any]:
    df = data.sample("train", sample_size)
    X, y = df[FEATS], df[C.TARGET].to_numpy()
    out = {}
    for name, splitter in (("kfold", KFold(n_splits=k, shuffle=True, random_state=0)), ("stratified_kfold", StratifiedKFold(n_splits=k, shuffle=True, random_state=0))):
        folds = []
        res = cross_validate(_pipe(model), X, y, cv=splitter, scoring=["average_precision", "roc_auc"], return_indices=True, error_score=np.nan)
        for i, val_idx in enumerate(res["indices"]["test"]):
            ap, auc = res["test_average_precision"][i], res["test_roc_auc"][i]
            folds.append(
                {
                    "fold": i + 1,
                    "validation_rows": int(len(val_idx)),
                    "fraud_rows": int(y[val_idx].sum()),
                    "positive_rate": r(y[val_idx].mean(), 5),
                    "average_precision": r(ap) if np.isfinite(ap) else None,
                    "roc_auc": r(auc) if np.isfinite(auc) else None,
                }
            )
        aps = np.array([f["average_precision"] for f in folds if f["average_precision"] is not None], float)
        out[name] = {
            "folds": folds,
            "average_precision_mean": r(aps.mean()) if len(aps) else None,
            "average_precision_std": r(aps.std()) if len(aps) else None,
            "fraud_count_std": r(np.std([f["fraud_rows"] for f in folds]), 3),
            "positive_rate_std": r(np.std([f["positive_rate"] for f in folds]), 6),
            "undefined_folds": int(sum(f["average_precision"] is None for f in folds)),
        }
    return {
        "split": data.split_info(df, 0, f"{k}-fold CV on a stratified {len(df):,}-row training sample ({int(y.sum())} fraud rows)."),
        "model": model,
        "k": k,
        "overall_positive_rate": r(y.mean(), 5),
        **out,
        "note": "With very few positives, plain K-fold can give folds with far fewer (or no) frauds, making their scores noisy or undefined. Stratified K-fold keeps the fraud count per fold equal.",
    }


def nested_cv(data: LabData, outer_k: int = 5, inner_k: int = 3, n_legit: int = 8000) -> dict[str, Any]:
    df = data.enriched("train", n_legit)
    X, y = df[FEATS], df[C.TARGET].to_numpy()
    grid = {"model__max_depth": [2, 4, 6, 8, 12, None], "model__min_samples_leaf": [1, 20]}
    inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=1)
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=2)
    t = time.perf_counter()
    flat = GridSearchCV(_pipe("decision_tree"), grid, cv=outer, scoring="average_precision", n_jobs=-1).fit(X, y)
    search = GridSearchCV(_pipe("decision_tree"), grid, cv=inner, scoring="average_precision", n_jobs=-1)
    res = cross_validate(search, X, y, cv=outer, scoring="average_precision", return_estimator=True)
    chosen = [
        {
            "outer_fold": i + 1,
            "outer_average_precision": r(s),
            "inner_best_average_precision": r(est.best_score_),
            "best_params": {k.replace("model__", ""): v for k, v in est.best_params_.items()},
        }
        for i, (s, est) in enumerate(zip(res["test_score"], res["estimator"]))
    ]
    nested_mean = float(np.mean(res["test_score"]))
    return {
        "split": data.split_info(df, 0, f"Decision-tree grid search on every training fraud + {n_legit:,} legitimate rows."),
        "grid": {k.replace("model__", ""): [("none" if v is None else v) for v in vals] for k, vals in grid.items()},
        "non_nested": {
            "best_score": r(flat.best_score_),
            "best_params": {k.replace("model__", ""): v for k, v in flat.best_params_.items()},
            "explanation": "Same folds used to pick hyperparameters and report the score: optimistically biased.",
        },
        "nested": {"outer_folds": chosen, "mean": r(nested_mean), "std": r(np.std(res["test_score"]))},
        "optimism_gap": r(flat.best_score_ - nested_mean),
        "seconds": r(time.perf_counter() - t, 2),
    }


def evaluation_metrics(data: LabData, model: str = "logistic_regression", threshold: float = 0.5) -> dict[str, Any]:
    tr, note = _train_frame(data, model)
    va = data.validation
    pipe = _pipe(model).fit(tr[FEATS], tr[C.TARGET])
    p = pipe.predict_proba(va[FEATS])[:, 1]
    y = va[C.TARGET].to_numpy()
    return {
        "split": data.split_info(tr, va, f"Fitted on {note}; evaluated on the full validation split."),
        "model": model,
        "label": CLASSIFIER_LABELS[model],
        "metrics": M.classification_metrics(y, p, threshold),
        "roc": M.roc_points(y, p),
        "pr": M.pr_points(y, p),
        "no_skill_precision": r(y.mean(), 6),
        "threshold_sweep": M.threshold_sweep(y, p),
        "best_f1_threshold": r(M.best_f1_threshold(y, p), 6),
        "accuracy_baseline": r(1 - y.mean(), 6),
        "note": "Predicting 'legitimate' for every transaction already scores this accuracy, which is why precision, recall and PR-AUC are used instead.",
    }


def calibration(data: LabData, model: str = "random_forest") -> dict[str, Any]:
    tr, note = _train_frame(data, model)
    va = data.validation
    X_cal, X_eval, y_cal, y_eval = train_test_split(va[FEATS], va[C.TARGET].to_numpy(), test_size=0.5, stratify=va[C.TARGET], random_state=0)
    base = _pipe(model).fit(tr[FEATS], tr[C.TARGET])
    out = {}
    for method in ("uncalibrated", "sigmoid", "isotonic"):
        if method == "uncalibrated":
            p = base.predict_proba(X_eval)[:, 1]
        else:
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)
            p = cal.predict_proba(X_eval)[:, 1]
        cp = M.calibration_points(y_eval, p)
        out[method] = {
            "label": {"uncalibrated": "Uncalibrated", "sigmoid": "Platt scaling (sigmoid)", "isotonic": "Isotonic regression"}[method],
            "brier": cp["brier"],
            "ece": cp["ece"],
            "mean_predicted": cp["mean_predicted"],
            "observed_rate": cp["observed_rate"],
            "log_loss": r(M.log_loss(y_eval, np.clip(p, 1e-7, 1 - 1e-7)), 5),
            "average_precision": r(average_precision_score(y_eval, p)),
            "reliability": cp["points"],
            "histogram": cp["histogram"],
        }
    return {
        "split": data.split_info(tr, va, f"Model fitted on {note}; calibrators fitted on {len(y_cal):,} validation rows; all methods scored on the other {len(y_eval):,}."),
        "model": model,
        "label": CLASSIFIER_LABELS[model],
        "methods": out,
        "note": "Training on a class-weighted or enriched sample inflates probabilities far above the real 0.17 % rate; calibration on data with the real rate brings the mean prediction back down.",
    }


def _search_space(model: str) -> tuple[dict[str, list], dict[str, Any]]:
    if model == "hist_gradient_boosting":
        grid = {"model__learning_rate": [0.03, 0.1, 0.3], "model__max_depth": [3, None], "model__l2_regularization": [0.0, 1.0]}
        dist = {"model__learning_rate": loguniform(0.01, 0.5), "model__max_depth": randint(2, 12), "model__l2_regularization": uniform(0, 5)}
    elif model == "random_forest":
        grid = {"model__n_estimators": [100, 200], "model__max_depth": [6, 12, None], "model__min_samples_leaf": [1, 5]}
        dist = {"model__n_estimators": randint(50, 300), "model__max_depth": randint(3, 20), "model__min_samples_leaf": randint(1, 20)}
    else:
        raise ValueError("model must be hist_gradient_boosting or random_forest")
    return grid, dist


def hyperparameter_search(data: LabData, model: str = "hist_gradient_boosting", budget: int = 12, n_legit: int = 8000) -> dict[str, Any]:
    df = data.enriched("train", n_legit)
    X, y = df[FEATS], df[C.TARGET].to_numpy()
    va = data.validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    grid, dist = _search_space(model)
    base = _pipe(model)
    results: dict[str, Any] = {}

    def trace(scores: list[float]) -> list[dict]:
        best = -np.inf
        out = []
        for i, s in enumerate(scores, 1):
            best = max(best, s)
            out.append({"iteration": i, "score": r(s), "best_so_far": r(best)})
        return out

    def finish(name: str, best_params: dict, scores: list[float], seconds: float) -> None:
        fitted = _pipe(model, **{k.replace("model__", ""): v for k, v in best_params.items()}).fit(X, y)
        val_ap = average_precision_score(va[C.TARGET], fitted.predict_proba(va[FEATS])[:, 1])
        results[name] = {
            "evaluations": len(scores),
            "best_cv_average_precision": r(max(scores)),
            "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
            "validation_average_precision": r(val_ap),
            "seconds": r(seconds, 2),
            "trace": trace(scores),
        }

    t = time.perf_counter()
    gs = GridSearchCV(base, grid, cv=cv, scoring="average_precision", n_jobs=-1).fit(X, y)
    finish("grid", gs.best_params_, list(gs.cv_results_["mean_test_score"]), time.perf_counter() - t)

    t = time.perf_counter()
    rs = RandomizedSearchCV(base, dist, n_iter=budget, cv=cv, scoring="average_precision", random_state=0, n_jobs=-1).fit(X, y)
    finish("random", rs.best_params_, list(rs.cv_results_["mean_test_score"]), time.perf_counter() - t)

    if optional_available("optuna"):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            if model == "hist_gradient_boosting":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }
            return float(cross_val_score(_pipe(model, **params), X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean())

        t = time.perf_counter()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=budget)
        finish("bayesian", {f"model__{k}": v for k, v in study.best_params.items()}, [tr.value for tr in study.trials], time.perf_counter() - t)
    else:
        results["bayesian"] = {"error": "Optuna is not installed"}

    return {
        "split": data.split_info(df, va, f"3-fold stratified CV on every training fraud + {n_legit:,} legitimate rows selects parameters; the validation split scores the winners."),
        "model": model,
        "budget": budget,
        "grid_size": int(np.prod([len(v) for v in grid.values()])),
        "space": {
            "grid": {k.replace("model__", ""): [("none" if v is None else v) for v in vals] for k, vals in grid.items()},
            "random_and_bayesian": "continuous / integer ranges (log-uniform learning rate)" if model == "hist_gradient_boosting" else "integer ranges",
        },
        "results": results,
        "note": "Bayesian optimisation uses Optuna's TPE sampler, which proposes new trials from a model of earlier results.",
    }


VALIDATION_PARAMS = {
    "decision_tree": ("max_depth", [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]),
    "logistic_regression": ("C", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]),
    "random_forest": ("min_samples_leaf", [1, 2, 5, 10, 20, 50, 100, 200]),
}


def learning_curves(data: LabData, model: str = "decision_tree", n_legit: int = 12000) -> dict[str, Any]:
    if model not in VALIDATION_PARAMS:
        raise ValueError(f"model must be one of {list(VALIDATION_PARAMS)}")
    df = data.enriched("train", n_legit)
    X, y = df[FEATS], df[C.TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    extra = {"max_depth": 8} if model == "decision_tree" else ({"n_estimators": 100} if model == "random_forest" else {})
    sizes, tr_s, va_s = learning_curve(_pipe(model, **extra), X, y, train_sizes=np.linspace(0.1, 1.0, 8), cv=cv, scoring="average_precision", n_jobs=-1, shuffle=True, random_state=0)
    lc = [
        {"train_size": int(n), "train_mean": r(a.mean()), "train_std": r(a.std()), "validation_mean": r(b.mean()), "validation_std": r(b.std())}
        for n, a, b in zip(sizes, tr_s, va_s)
    ]
    pname, prange = VALIDATION_PARAMS[model]
    base = _pipe(model, **({"n_estimators": 100} if model == "random_forest" else {}))
    tr_v, va_v = validation_curve(base, X, y, param_name=f"model__{pname}", param_range=prange, cv=cv, scoring="average_precision", n_jobs=-1)
    best_val = max(b.mean() for b in va_v)
    vc = []
    for v, a, b in zip(prange, tr_v, va_v):
        gap = a.mean() - b.mean()
        diag = "overfitting" if gap > 0.05 else ("underfitting" if b.mean() < best_val - 0.03 else "good fit")
        vc.append({"value": v, "train_mean": r(a.mean()), "validation_mean": r(b.mean()), "gap": r(gap), "diagnosis": diag})
    return {
        "split": data.split_info(df, 0, f"5-fold stratified CV on every training fraud + {n_legit:,} legitimate rows."),
        "model": model,
        "label": CLASSIFIER_LABELS[model],
        "learning_curve": lc,
        "validation_curve": {"parameter": pname, "points": vc},
        "rules": "gap = train - validation average precision; overfitting if gap > 0.05; underfitting if validation is more than 0.03 below the best value.",
    }


def statistical_comparison(data: LabData, model_a: str = "logistic_regression", model_b: str = "random_forest") -> dict[str, Any]:
    va = data.validation
    y = va[C.TARGET].to_numpy()
    scores = {}
    for m in (model_a, model_b):
        tr, _ = _train_frame(data, m)
        fitted = _pipe(m).fit(tr[FEATS], tr[C.TARGET])
        # Rank logistic regression by its log-odds: with class weights its probabilities saturate at
        # exactly 1.0 for extreme frauds, and those ties would understate its average precision.
        scores[m] = fitted.decision_function(va[FEATS]) if m == "logistic_regression" else fitted.predict_proba(va[FEATS])[:, 1]
    pa, pb = scores[model_a], scores[model_b]
    # Thresholds are chosen on one half of the validation split and McNemar runs on the other half.
    idx_tune, idx_test = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    ta = M.best_f1_threshold(y[idx_tune], pa[idx_tune])
    tb = M.best_f1_threshold(y[idx_tune], pb[idx_tune])
    yt = y[idx_test]
    pred_a, pred_b = (pa[idx_test] >= ta).astype(int), (pb[idx_test] >= tb).astype(int)
    mc = M.mcnemar_test(yt, pred_a, pred_b)
    boot = M.paired_bootstrap_ap(y, pa, pb, n_boot=1000)
    return {
        "split": data.split_info(0, va, "Both models are scored on exactly the same validation rows (paired)."),
        "classification": {
            "model_a": model_a,
            "model_b": model_b,
            "label_a": CLASSIFIER_LABELS[model_a],
            "label_b": CLASSIFIER_LABELS[model_b],
            "threshold_a": r(ta, 6),
            "threshold_b": r(tb, 6),
            "test_half_rows": int(len(yt)),
            "test_half_frauds": int(yt.sum()),
            "metrics_a": M.classification_metrics(yt, pa[idx_test], ta, probabilistic=False),
            "metrics_b": M.classification_metrics(yt, pb[idx_test], tb, probabilistic=False),
            "mcnemar": mc,
        },
        "ranking": {
            "average_precision_a": r(average_precision_score(y, pa)),
            "average_precision_b": r(average_precision_score(y, pb)),
            "roc_auc_a": r(roc_auc_score(y, pa)),
            "roc_auc_b": r(roc_auc_score(y, pb)),
            "paired_bootstrap": boot,
            "interpretation": "Difference = AP(A) - AP(B), stratified paired bootstrap; a 95% interval excluding 0 indicates a reliable difference.",
            "score_note": "Logistic regression is ranked by its log-odds (decision function) because its class-weighted probabilities saturate at 1.0.",
        },
    }
