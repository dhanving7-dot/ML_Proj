"""Evaluation metrics, curves and statistical comparison tests (M5)."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sstats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _downsample(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": _r(threshold),
        "accuracy": _r(accuracy_score(y_true, pred)),
        "precision": _r(precision_score(y_true, pred, zero_division=0)),
        "recall": _r(recall_score(y_true, pred, zero_division=0)),
        "f1": _r(f1_score(y_true, pred, zero_division=0)),
        "specificity": _r(tn / (tn + fp) if (tn + fp) else 0.0),
        "roc_auc": _r(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": _r(average_precision_score(y_true, proba)) if len(np.unique(y_true)) == 2 else None,
        "log_loss": _r(log_loss(y_true, proba, labels=[0, 1])),
        "brier": _r(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": int(len(y_true)),
    }


def roc_points(y_true: np.ndarray, proba: np.ndarray, max_points: int = 120) -> list[dict]:
    fpr, tpr, thr = roc_curve(y_true, proba)
    idx = _downsample(len(fpr), max_points)
    return [
        {"fpr": _r(fpr[i]), "tpr": _r(tpr[i]), "threshold": _r(min(thr[i], 1.0))}
        for i in idx
    ]


def pr_points(y_true: np.ndarray, proba: np.ndarray, max_points: int = 120) -> list[dict]:
    precision, recall, thr = precision_recall_curve(y_true, proba)
    thr = np.append(thr, 1.0)
    idx = _downsample(len(precision), max_points)
    return [
        {"recall": _r(recall[i]), "precision": _r(precision[i]), "threshold": _r(thr[i])}
        for i in idx
    ]


def calibration_points(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(proba, bins=bins)
    # Expected calibration error with the same uniform bins.
    ids = np.clip(np.digitize(proba, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(np.mean(y_true[m]) - np.mean(proba[m]))
    return {
        "points": [{"mean_predicted": _r(p), "fraction_positive": _r(f)} for p, f in zip(mean_pred, frac_pos)],
        "histogram": [{"bin_start": _r(bins[i], 2), "count": int(counts[i])} for i in range(n_bins)],
        "ece": _r(ece),
        "brier": _r(brier_score_loss(y_true, proba)),
    }


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": _r(t, 3),
                "precision": _r(precision_score(y_true, pred, zero_division=0)),
                "recall": _r(recall_score(y_true, pred, zero_division=0)),
                "f1": _r(f1_score(y_true, pred, zero_division=0)),
                "accuracy": _r(accuracy_score(y_true, pred)),
                "predicted_positive_rate": _r(pred.mean()),
            }
        )
    return rows


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, Any] = {
        "rmse": _r(root_mean_squared_error(y_true, y_pred)),
        "mae": _r(mean_absolute_error(y_true, y_pred)),
        "r2": _r(r2_score(y_true, y_pred)),
        "support": int(len(y_true)),
    }
    # MAPE is only meaningful when no target is (near) zero.
    if np.all(np.abs(y_true) > 1e-9):
        out["mape"] = _r(mean_absolute_percentage_error(y_true, y_pred))
    else:
        out["mape"] = None
        out["mape_note"] = "Undefined: the target contains zeros."
    return out


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """McNemar's test on paired classifier predictions over the same examples.

    Uses the exact binomial test when the number of discordant pairs is small (< 25),
    otherwise the chi-squared statistic with continuity correction.
    """
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    both_ok = int(np.sum(a_ok & b_ok))
    both_wrong = int(np.sum(~a_ok & ~b_ok))
    discordant = n01 + n10
    if discordant == 0:
        stat, p, method = 0.0, 1.0, "no discordant pairs"
    elif discordant < 25:
        p = float(sstats.binomtest(min(n01, n10), discordant, 0.5).pvalue)
        stat, method = float(min(n01, n10)), "exact binomial"
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / discordant
        p = float(sstats.chi2.sf(stat, df=1))
        method = "chi-squared with continuity correction"
    return {
        "table": {"both_correct": both_ok, "a_only_correct": n01, "b_only_correct": n10, "both_wrong": both_wrong},
        "statistic": _r(stat),
        "p_value": float(f"{p:.6g}"),
        "method": method,
        "significant_at_0_05": bool(p < 0.05),
    }


def paired_bootstrap_rmse(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """Paired bootstrap of RMSE(A) - RMSE(B) over the same resampled examples."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = np.sqrt(np.mean((y_true[idx] - pred_a[idx]) ** 2))
        rb = np.sqrt(np.mean((y_true[idx] - pred_b[idx]) ** 2))
        diffs[i] = ra - rb
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    observed = float(np.sqrt(np.mean((y_true - pred_a) ** 2)) - np.sqrt(np.mean((y_true - pred_b) ** 2)))
    counts, edges = np.histogram(diffs, bins=30)
    return {
        "observed_difference": _r(observed),
        "ci_95": [_r(lo), _r(hi)],
        "p_a_not_worse": _r(float(np.mean(diffs <= 0))),
        "n_boot": n_boot,
        "significant": bool(lo > 0 or hi < 0),
        "histogram": [{"start": _r(edges[i]), "count": int(counts[i])} for i in range(len(counts))],
    }
