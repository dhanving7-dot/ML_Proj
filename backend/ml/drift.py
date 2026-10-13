"""Drift statistics from the handout: PSI, KL divergence, summary-statistic shifts, skewness/kurtosis.

Reference profiles are computed from the *training* split at training time and stored in the
model metadata, so serving-time drift is always measured against what the model learned from.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import contract as C

EPS = 1e-4
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def psi_status(psi: float | None) -> str:
    if psi is None:
        return "insufficient_data"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "investigate"
    return "significant_shift"


def _safe_props(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    p = counts / total if total else np.zeros_like(counts, dtype=float)
    return np.clip(p, EPS, None)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((A - E) * ln(A / E)) over bins of proportions."""
    e, a = _safe_props(np.asarray(expected, float)), _safe_props(np.asarray(actual, float))
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P || Q) in bits, where P is the serving distribution and Q the training reference."""
    p, q = _safe_props(np.asarray(p, float)), _safe_props(np.asarray(q, float))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def numeric_reference(values: pd.Series, n_bins: int = 10) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce")
    clean = v.dropna().to_numpy(float)
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # low-cardinality column: fall back to unique values as bin edges
        uniq = np.unique(clean)
        edges = np.concatenate([uniq - 0.5, [uniq[-1] + 0.5]])
    inner = edges[1:-1]
    counts = np.bincount(np.searchsorted(inner, clean, side="right"), minlength=len(edges) - 1)
    return {
        "kind": "numeric",
        "edges": [float(e) for e in edges],
        "proportions": (counts / max(counts.sum(), 1)).round(6).tolist(),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "null_rate": float(v.isna().mean()),
        "skewness": float(sstats.skew(clean)),
        "kurtosis": float(sstats.kurtosis(clean)),
        "n": int(len(v)),
    }


def categorical_reference(values: pd.Series) -> dict[str, Any]:
    vc = values.astype(object).where(values.notna(), "__missing__").astype(str).value_counts(normalize=True)
    return {"kind": "categorical", "proportions": {str(k): round(float(p), 6) for k, p in vc.items()}, "n": int(len(values))}


def build_reference_profile(train_X: pd.DataFrame, val_proba: np.ndarray | None = None) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in train_X.columns:
        f = C.FEATURES_BY_NAME.get(name)
        if isinstance(f, C.NumericFeature):
            features[name] = numeric_reference(train_X[name])
        else:
            features[name] = categorical_reference(train_X[name])
    profile: dict[str, Any] = {"features": features}
    if val_proba is not None:
        profile["prediction"] = numeric_reference(pd.Series(val_proba))
    return profile


def compare_numeric(ref: dict[str, Any], values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce")
    clean = v.dropna().to_numpy(float)
    edges = np.asarray(ref["edges"], float)
    if len(clean) == 0:
        return {"kind": "numeric", "psi": None, "kl_divergence": None, "status": "insufficient_data", "n": 0}
    counts = np.bincount(np.searchsorted(edges[1:-1], clean, side="right"), minlength=len(edges) - 1)
    actual = counts / counts.sum()
    expected = np.asarray(ref["proportions"], float)
    p = psi(expected, actual)
    return {
        "kind": "numeric",
        "psi": round(p, 5),
        "kl_divergence": round(kl_divergence(actual, expected), 5),
        "status": psi_status(p),
        "n": int(len(v)),
        "bins": [
            {"bin": f"{edges[i]:.2f}-{edges[i + 1]:.2f}", "expected": round(float(expected[i]), 4), "actual": round(float(actual[i]), 4)}
            for i in range(len(expected))
        ],
        "summary": {
            "reference_mean": round(ref["mean"], 4),
            "current_mean": round(float(clean.mean()), 4),
            "reference_std": round(ref["std"], 4),
            "current_std": round(float(clean.std(ddof=1)), 4) if len(clean) > 1 else None,
            "reference_null_rate": round(ref["null_rate"], 4),
            "current_null_rate": round(float(v.isna().mean()), 4),
            "reference_skewness": round(ref["skewness"], 4),
            "current_skewness": round(float(sstats.skew(clean)), 4) if len(clean) > 2 else None,
            "reference_kurtosis": round(ref["kurtosis"], 4),
            "current_kurtosis": round(float(sstats.kurtosis(clean)), 4) if len(clean) > 3 else None,
        },
    }


def compare_categorical(ref: dict[str, Any], values: pd.Series) -> dict[str, Any]:
    cur = values.astype(object).where(values.notna(), "__missing__").astype(str).value_counts(normalize=True)
    cats = sorted(set(ref["proportions"]) | set(cur.index))
    expected = np.array([ref["proportions"].get(c, 0.0) for c in cats])
    actual = np.array([float(cur.get(c, 0.0)) for c in cats])
    if len(values) == 0:
        return {"kind": "categorical", "psi": None, "kl_divergence": None, "status": "insufficient_data", "n": 0}
    p = psi(expected, actual)
    return {
        "kind": "categorical",
        "psi": round(p, 5),
        "kl_divergence": round(kl_divergence(actual, expected), 5),
        "status": psi_status(p),
        "n": int(len(values)),
        "bins": [{"bin": c, "expected": round(float(e), 4), "actual": round(float(a), 4)} for c, e, a in zip(cats, expected, actual)],
        "unseen_categories": [c for c in cur.index if c not in ref["proportions"]],
    }


def drift_report(reference: dict[str, Any], current_X: pd.DataFrame, current_proba: np.ndarray | None = None) -> dict[str, Any]:
    features = {}
    for name, ref in reference["features"].items():
        if name not in current_X.columns:
            continue
        features[name] = (
            compare_numeric(ref, current_X[name]) if ref["kind"] == "numeric" else compare_categorical(ref, current_X[name])
        )
    out: dict[str, Any] = {"features": features}
    if current_proba is not None and "prediction" in reference:
        out["prediction"] = compare_numeric(reference["prediction"], pd.Series(current_proba))
    flagged = [n for n, r in features.items() if r["status"] in ("investigate", "significant_shift")]
    out["flagged_features"] = flagged
    out["thresholds"] = {"stable_below": PSI_STABLE, "significant_at_or_above": PSI_SIGNIFICANT}
    return out
