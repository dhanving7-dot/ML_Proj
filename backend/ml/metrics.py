"""Evaluation metrics, curves, threshold selection and statistical comparison tests (M5).

The fraud data is severely imbalanced (about 0.17 % positives), so every report leads with
precision, recall, F1 and average precision (PR-AUC); accuracy is included only for completeness.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sstats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    r2_score,
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


def _two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def confusion(y_true: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred).astype(int)
    return {
        "tn": int(np.sum((y == 0) & (p == 0))),
        "fp": int(np.sum((y == 0) & (p == 1))),
        "fn": int(np.sum((y == 1) & (p == 0))),
        "tp": int(np.sum((y == 1) & (p == 1))),
    }


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5, *, probabilistic: bool = True) -> dict[str, Any]:
    """Threshold metrics plus ranking metrics. ``probabilistic`` adds Brier score and log loss.

    Predictions are ``score >= threshold``. Ranking metrics (ROC-AUC, average precision) do not
    depend on the threshold and are ``None`` when only one class is present.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    cm = confusion(y, pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out: dict[str, Any] = {
        "threshold": _r(threshold, 6),
        "precision": _r(precision),
        "recall": _r(recall),
        "f1": _r(f1),
        "fraud_detection_rate": _r(recall),
        "specificity": _r(specificity),
        "false_positive_rate": _r(1 - specificity),
        "accuracy": _r((tp + tn) / max(len(y), 1)),
        "balanced_accuracy": _r((recall + specificity) / 2),
        "roc_auc": _r(roc_auc_score(y, s)) if _two_classes(y) else None,
        "average_precision": _r(average_precision_score(y, s)) if _two_classes(y) else None,
        "confusion_matrix": cm,
        "false_positives": fp,
        "false_negatives": fn,
        "alerts": int(pred.sum()),
        "alert_rate": _r(pred.mean() if len(pred) else 0.0, 6),
        "support": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": _r(y.mean() if len(y) else 0.0, 6),
    }
    if probabilistic:
        p = np.clip(s, 1e-12, 1 - 1e-12)
        out["log_loss"] = _r(log_loss(y, p, labels=[0, 1]))
        out["brier"] = _r(brier_score_loss(y, p), 6)
    return out


def roc_points(y_true: np.ndarray, score: np.ndarray, max_points: int = 160) -> list[dict]:
    fpr, tpr, thr = roc_curve(y_true, score, drop_intermediate=True)
    idx = _downsample(len(fpr), max_points)
    return [{"fpr": _r(fpr[i], 6), "tpr": _r(tpr[i]), "threshold": _r(min(thr[i], 1e6), 6)} for i in idx]


def pr_points(y_true: np.ndarray, score: np.ndarray, max_points: int = 200) -> list[dict]:
    """PR curve keeping the points where recall changes (the curve's real corners).

    Downsampling uniformly by threshold would spend almost every point on the long
    recall = 1, precision ~ 0 tail of an imbalanced problem.
    """
    precision, recall, thr = precision_recall_curve(y_true, score)
    thr = np.append(thr, np.inf)
    keep = np.flatnonzero(np.r_[True, np.diff(recall) != 0] | np.r_[np.diff(recall) != 0, True])
    if len(keep) > max_points:
        keep = keep[_downsample(len(keep), max_points)]
    return [{"recall": _r(recall[i]), "precision": _r(precision[i]), "threshold": _r(min(thr[i], 1e6), 6)} for i in keep]


def calibration_points(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Reliability diagram (uniform bins), expected calibration error and Brier score."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    points, hist = [], []
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        hist.append({"bin_start": _r(bins[b], 2), "count": int(m.sum())})
        if m.any():
            points.append({"mean_predicted": _r(p[m].mean(), 6), "fraction_positive": _r(y[m].mean(), 6), "count": int(m.sum())})
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {
        "points": points,
        "histogram": hist,
        "ece": _r(ece, 6),
        "brier": _r(brier_score_loss(y, p), 6),
        "mean_predicted": _r(p.mean(), 6),
        "observed_rate": _r(y.mean(), 6),
    }


def threshold_sweep(y_true: np.ndarray, score: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    thresholds = np.round(np.arange(0.01, 0.991, 0.01), 3) if thresholds is None else np.asarray(thresholds, float)
    pos = max(int(y.sum()), 1)
    rows = []
    for t in thresholds:
        pred = s >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / pos
        rows.append(
            {
                "threshold": _r(t, 4),
                "precision": _r(precision),
                "recall": _r(recall),
                "f1": _r(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0),
                "alert_rate": _r(pred.mean(), 6),
                "alerts": int(pred.sum()),
                "false_positives": fp,
            }
        )
    return rows


def best_fbeta_threshold(y_true: np.ndarray, score: np.ndarray, beta: float = 1.0) -> float:
    """Score cut-off maximising F-beta (predictions are ``score >= threshold``)."""
    precision, recall, thr = precision_recall_curve(y_true, score)
    p, r = precision[:-1], recall[:-1]
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((b2 * p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    return float(thr[int(np.argmax(f))])


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    return best_fbeta_threshold(y_true, score, 1.0)


def threshold_for_precision(y_true: np.ndarray, score: np.ndarray, target: float) -> float | None:
    """Lowest cut-off whose precision reaches ``target`` (maximises recall under the constraint)."""
    precision, _, thr = precision_recall_curve(y_true, score)
    ok = np.flatnonzero(precision[:-1] >= target)
    return float(thr[ok[0]]) if len(ok) else None


def threshold_for_recall(y_true: np.ndarray, score: np.ndarray, target: float, max_alert_rate: float | None = None) -> dict[str, Any]:
    """Highest cut-off that still reaches ``target`` recall, optionally capped by an alert-rate budget.

    Returns the threshold and which constraint decided it, so the choice is auditable.
    """
    s = np.asarray(score, dtype=float)
    _, recall, thr = precision_recall_curve(y_true, s)
    ok = np.flatnonzero(recall[:-1] >= target)
    t_recall = float(thr[ok[-1]]) if len(ok) else float(np.min(s))
    out: dict[str, Any] = {"threshold": t_recall, "binding": "recall_target", "recall_target": target}
    if max_alert_rate is not None:
        rate = float(np.mean(s >= t_recall))
        if rate > max_alert_rate:
            srt = np.sort(s)
            # smallest unique cut-off whose alert rate is within the budget
            cand = np.unique(srt)
            rates = 1.0 - np.searchsorted(srt, cand, side="left") / len(srt)
            within = cand[rates <= max_alert_rate]
            out.update({"threshold": float(within[0]) if len(within) else float(srt[-1]) + 1e-12, "binding": "max_alert_rate"})
        out["max_alert_rate"] = max_alert_rate
    return out


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


def paired_bootstrap_ap(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Stratified paired bootstrap of AP(A) - AP(B).

    Fraud and legitimate rows are resampled separately so every replicate keeps the observed
    number of frauds (a plain bootstrap could draw replicates with almost no positives).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[i] = average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    counts, edges = np.histogram(diffs, bins=30)
    return {
        "observed_difference": _r(average_precision_score(y, a) - average_precision_score(y, b)),
        "ci_95": [_r(lo), _r(hi)],
        "p_a_not_better": _r(float(np.mean(diffs <= 0))),
        "n_boot": n_boot,
        "significant": bool(lo > 0 or hi < 0),
        "histogram": [{"start": _r(edges[i]), "count": int(counts[i])} for i in range(len(counts))],
    }
