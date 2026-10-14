"""Hybrid fraud model: Logistic Regression + Isolation Forest + documented score fusion.

Scores
------
* ``lr_probability``: fraud probability from a class-weighted Logistic Regression. Class
  weighting (or undersampling) shifts the model's implied prior to roughly 50 % fraud, so the
  intercept is corrected by ``log(w1 * keep1 / (w0 * keep0))`` (King & Zeng prior correction).
  The result is an approximately calibrated probability; calibration is *measured* (Brier, ECE)
  on the validation and test splits rather than assumed.
* ``anomaly_score``: ``-IsolationForest.score_samples(x)``; higher means easier to isolate.
  It is **not** a probability. ``anomaly_percentile`` is the share of training-reference
  transactions with a lower or equal anomaly score.

Normalisation ("rarity")
------------------------
Each score ``s`` is compared with the same score computed on the training split (the
reference, ``n_ref`` rows). With ``e(s)`` = share of reference rows scoring strictly higher::

    rarity(s) = -log10(max(e(s), 1 / (n_ref + 1))) / log10(n_ref + 1)        in [0, 1]

``rarity = 0.5`` means "as rare as the top ~1/sqrt(n_ref) of reference transactions";
``rarity = 1`` means "more extreme than every reference transaction". Both scores end up in
the same unit (decades of rarity), so they can be combined without adding incompatible raw
scores. Logistic Regression rarity is computed on its log-odds to avoid float saturation near 1.

Fusion and decisions
--------------------
::

    hybrid = w * rarity_LR + (1 - w) * rarity_IF                               in [0, 1]
    HIGH_RISK  if hybrid >= t_high
    REVIEW     if t_review <= hybrid < t_high
               or (optional anomaly trigger) anomaly_percentile >= q_review
    LOW_RISK   otherwise

``w``, ``t_review`` and ``t_high`` are tuned on the validation split only (see ``training.py``).
The anomaly review trigger is off by default; when enabled it routes extreme anomalies to human
review even if the supervised score is low (it never produces HIGH_RISK on its own).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import contract as C
from .features import describe_feature


class RarityReference:
    """Empirical reference distribution of a score, used to map scores to rarity in [0, 1]."""

    def __init__(self, scores: np.ndarray):
        s = np.sort(np.asarray(scores, dtype=float))
        if len(s) == 0:
            raise ValueError("A rarity reference needs at least one score")
        self.sorted_scores = s
        self.n = int(len(s))
        self.cap = float(np.log10(self.n + 1))

    def cdf(self, scores: np.ndarray) -> np.ndarray:
        """Share of reference rows with a score <= s."""
        return np.searchsorted(self.sorted_scores, np.asarray(scores, dtype=float), side="right") / self.n

    def exceedance(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 - self.cdf(scores)

    def rarity(self, scores: np.ndarray) -> np.ndarray:
        e = np.maximum(self.exceedance(scores), 1.0 / (self.n + 1))
        return np.clip(-np.log10(e) / self.cap, 0.0, 1.0)

    def summary(self) -> dict[str, Any]:
        q = np.quantile(self.sorted_scores, [0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
        return {"n": self.n, "quantiles": {k: float(v) for k, v in zip(("min", "p50", "p90", "p99", "p99.9", "max"), q)}}


@dataclass
class DecisionPolicy:
    weight: float
    t_review: float
    t_high: float
    lr_threshold: float
    if_flag_percentile: float
    if_review_percentile: float | None = None  # None = anomaly review trigger disabled

    def to_dict(self) -> dict[str, float | None]:
        # Not rounded: thresholds are exact score values of validation rows, and rounding would move rows across them.
        return {
            "fusion_weight_lr": float(self.weight),
            "fusion_weight_if": float(1 - self.weight),
            "review_threshold": float(self.t_review),
            "high_risk_threshold": float(self.t_high),
            "lr_threshold": float(self.lr_threshold),
            "if_flag_percentile": float(self.if_flag_percentile),
            "if_review_percentile": None if self.if_review_percentile is None else float(self.if_review_percentile),
        }


def fuse(lr_rarity: np.ndarray, if_rarity: np.ndarray, weight: float) -> np.ndarray:
    return weight * np.asarray(lr_rarity, float) + (1.0 - weight) * np.asarray(if_rarity, float)


def decide(
    hybrid: np.ndarray,
    t_review: float,
    t_high: float,
    anomaly_percentile: np.ndarray | None = None,
    if_review_percentile: float | None = None,
) -> np.ndarray:
    """Decision codes: 0 = LOW_RISK, 1 = REVIEW, 2 = HIGH_RISK."""
    h = np.asarray(hybrid, float)
    review = h >= t_review
    if if_review_percentile is not None and anomaly_percentile is not None:
        review = review | (np.asarray(anomaly_percentile, float) >= if_review_percentile)
    return np.where(h >= t_high, 2, np.where(review, 1, 0)).astype(int)


def decision_names(codes: np.ndarray) -> list[str]:
    return [C.DECISIONS[int(c)] for c in codes]


class FraudModel:
    """Fitted preprocessing + Logistic Regression + Isolation Forest + references + decision policy.

    One object is persisted per model version, so training and serving share every fitted
    parameter and every rule. It is built by :func:`ml.training.train_model`.
    """

    def __init__(
        self,
        preprocessor: Pipeline,
        lr: Any,
        lr_logit_offset: float,
        iforest: Any,
        lr_reference: RarityReference,
        if_reference: RarityReference,
        policy: DecisionPolicy,
        feature_ranges: dict[str, tuple[float, float]],
    ):
        self.preprocessor = preprocessor
        self.lr = lr
        self.lr_logit_offset = float(lr_logit_offset)
        self.iforest = iforest
        self.lr_reference = lr_reference
        self.if_reference = if_reference
        self.policy = policy
        self.feature_ranges = feature_ranges
        self.features = list(C.FEATURE_NAMES)
        self.classes_ = np.array([0, 1])

    # ---------------------------------------------------------------- components
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X[self.features]), dtype=float)

    def transformed_names(self) -> list[str]:
        return list(self.preprocessor.named_steps["engineer"].output_names())

    def lr_logit(self, Xt: np.ndarray) -> np.ndarray:
        return self.lr.decision_function(Xt) - self.lr_logit_offset

    def anomaly_score(self, Xt: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(Xt)

    # ---------------------------------------------------------------- scoring
    def score_transformed(self, Xt: np.ndarray, policy: DecisionPolicy | None = None) -> dict[str, np.ndarray]:
        p = policy or self.policy
        logit = self.lr_logit(Xt)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
        anomaly = self.anomaly_score(Xt)
        r_lr = self.lr_reference.rarity(logit)
        r_if = self.if_reference.rarity(anomaly)
        hybrid = fuse(r_lr, r_if, p.weight)
        pct = self.if_reference.cdf(anomaly)
        return {
            "lr_logit": logit,
            "lr_probability": prob,
            "anomaly_score": anomaly,
            "anomaly_percentile": pct,
            "lr_rarity": r_lr,
            "if_rarity": r_if,
            "hybrid_score": hybrid,
            "decision": decide(hybrid, p.t_review, p.t_high, pct, p.if_review_percentile),
            "lr_flag": (prob >= p.lr_threshold).astype(int),
            "if_flag": (pct >= p.if_flag_percentile).astype(int),
        }

    def score(self, X: pd.DataFrame, policy: DecisionPolicy | None = None) -> dict[str, np.ndarray]:
        return self.score_transformed(self.transform(X), policy)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Logistic Regression probability in scikit-learn shape (used by generic tooling)."""
        p = self.score(X)["lr_probability"]
        return np.column_stack([1 - p, p])

    # ---------------------------------------------------------------- explanation
    def lr_contributions(self, xt_row: np.ndarray) -> list[dict[str, Any]]:
        """Exact additive decomposition of the LR log-odds: coefficient x standardised value."""
        coef = np.asarray(self.lr.coef_[0], float)
        names = self.transformed_names()
        contrib = coef * np.asarray(xt_row, float)
        return [
            {"feature": n, "label": describe_feature(n), "standardised_value": float(v), "coefficient": float(c), "contribution": float(k)}
            for n, v, c, k in zip(names, xt_row, coef, contrib)
        ]

    def out_of_range(self, X: pd.DataFrame) -> list[list[str]]:
        """Per row, the raw features outside the range seen in the training split (a warning, not a rejection)."""
        vals = X[self.features].to_numpy(dtype=float)
        lo = np.array([self.feature_ranges[f][0] for f in self.features])
        hi = np.array([self.feature_ranges[f][1] for f in self.features])
        mask = (vals < lo) | (vals > hi)
        return [[self.features[j] for j in np.flatnonzero(row)] for row in mask]

    def explain(self, X: pd.DataFrame, scores: dict[str, np.ndarray], Xt: np.ndarray | None = None, top: int = 6) -> list[dict[str, Any]]:
        """Facts derived from the fitted models only; nothing is inferred about what a PCA component means."""
        Xt = self.transform(X) if Xt is None else Xt
        oor = self.out_of_range(X)
        p = self.policy
        lr_exc = self.lr_reference.exceedance(scores["lr_logit"])
        if_exc = self.if_reference.exceedance(scores["anomaly_score"])
        intercept = float(self.lr.intercept_[0])
        out = []
        for i in range(len(X)):
            contrib = sorted(self.lr_contributions(Xt[i]), key=lambda d: -abs(d["contribution"]))
            out.append(
                {
                    "lr_top_contributions": contrib[:top],
                    "lr_intercept": intercept,
                    "lr_prior_correction": -self.lr_logit_offset,
                    "lr_logit": float(scores["lr_logit"][i]),
                    "hybrid_components": {
                        "lr": float(p.weight * scores["lr_rarity"][i]),
                        "isolation_forest": float((1 - p.weight) * scores["if_rarity"][i]),
                    },
                    "lr_exceedance": float(lr_exc[i]),
                    "if_exceedance": float(if_exc[i]),
                    "anomaly_review_trigger": bool(p.if_review_percentile is not None and scores["anomaly_percentile"][i] >= p.if_review_percentile),
                    "out_of_training_range": oor[i],
                }
            )
        return out
