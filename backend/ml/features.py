"""Shared feature pipeline: type coercion, feature engineering and preprocessing.

Training and serving both call :func:`coerce_features` and use the fitted preprocessing
``Pipeline`` stored in the model artifact, so scaling parameters are learned once (on the
training split only) and reused unchanged in production.

Engineered features
-------------------
* ``amount_log`` = ``log1p(Amount)``: ``Amount`` is strongly right-skewed (median 22, max 25,691).
* ``hour_sin`` / ``hour_cos``: ``Time`` counts seconds since the first transaction in the dataset
  and only ever increases, so new transactions would fall outside the training range. It is
  therefore encoded as a position on a 24-hour cycle, ``h = (Time mod 86400) / 3600``. The clock
  time of ``Time = 0`` is not published, so ``h`` is "hours since the first transaction's clock
  time", not wall-clock hour.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import contract as C

SECONDS_PER_DAY = 86_400.0
TIME_ENCODINGS = ("cyclical_hour", "raw", "none")
AMOUNT_TRANSFORMS = ("log1p", "raw")


def coerce_features(data: pd.DataFrame | Iterable[Mapping[str, Any]], names: Iterable[str] = C.FEATURE_NAMES) -> pd.DataFrame:
    """Return a float frame with exactly the contract columns, in contract order.

    Unparsable, non-finite and out-of-range values become NaN (the training path drops such
    rows and the API rejects them before this point).
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
    out: dict[str, pd.Series] = {}
    for name in names:
        f = C.FEATURES_BY_NAME[name]
        raw = df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        col = pd.to_numeric(raw, errors="coerce").astype(float)
        ok = np.isfinite(col.to_numpy(dtype=float, na_value=np.nan))
        if f.minimum is not None:
            ok &= (col >= f.minimum).to_numpy()
        if f.maximum is not None:
            ok &= (col <= f.maximum).to_numpy()
        out[name] = col.where(ok)
    return pd.DataFrame(out, index=df.index)


def hour_of_cycle(time_s: np.ndarray | pd.Series) -> np.ndarray:
    return (np.asarray(time_s, dtype=float) % SECONDS_PER_DAY) / 3600.0


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering (nothing is learned), so it behaves identically in training and serving."""

    def __init__(self, time_encoding: str = "cyclical_hour", amount_transform: str = "log1p"):
        self.time_encoding = time_encoding
        self.amount_transform = amount_transform

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TransactionFeatureEngineer":
        if self.time_encoding not in TIME_ENCODINGS:
            raise ValueError(f"time_encoding must be one of {TIME_ENCODINGS}")
        if self.amount_transform not in AMOUNT_TRANSFORMS:
            raise ValueError(f"amount_transform must be one of {AMOUNT_TRANSFORMS}")
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def output_names(self) -> list[str]:
        names = list(C.PCA_FEATURES)
        names.append("amount_log" if self.amount_transform == "log1p" else "Amount")
        if self.time_encoding == "cyclical_hour":
            names += ["hour_sin", "hour_cos"]
        elif self.time_encoding == "raw":
            names.append("Time")
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for v in C.PCA_FEATURES:
            out[v] = X[v].astype(float)
        amount = X["Amount"].astype(float)
        if self.amount_transform == "log1p":
            out["amount_log"] = np.log1p(amount)
        else:
            out["Amount"] = amount
        if self.time_encoding == "cyclical_hour":
            angle = 2 * np.pi * hour_of_cycle(X["Time"]) / 24.0
            out["hour_sin"] = np.sin(angle)
            out["hour_cos"] = np.cos(angle)
        elif self.time_encoding == "raw":
            out["Time"] = X["Time"].astype(float)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.output_names(), dtype=object)


def build_preprocessor(time_encoding: str = "cyclical_hour", amount_transform: str = "log1p", scaler: str = "standard") -> Pipeline:
    """[feature engineering] -> scaling. Fitted on the training split only."""
    steps: list[tuple[str, Any]] = [("engineer", TransactionFeatureEngineer(time_encoding, amount_transform))]
    if scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        steps.append(("scale", MinMaxScaler()))
    elif scaler != "none":
        raise ValueError("scaler must be standard, minmax or none")
    return Pipeline(steps)


def build_pipeline(estimator: Any, *, time_encoding: str = "cyclical_hour", amount_transform: str = "log1p", scaler: str = "standard") -> Pipeline:
    """Full model pipeline used by the laboratory: preprocessing -> estimator."""
    pre = build_preprocessor(time_encoding, amount_transform, scaler)
    return Pipeline([*pre.steps, ("model", estimator)])


def transformed_feature_names(preprocessor: Pipeline) -> list[str]:
    return list(preprocessor.named_steps["engineer"].output_names())


def describe_feature(name: str) -> str:
    """Human-readable description of a transformed feature. PCA components are never given a meaning."""
    if name in C.PCA_FEATURES:
        return f"{name} (anonymised PCA component)"
    return {
        "amount_log": "log(1 + Amount)",
        "Amount": "Amount",
        "hour_sin": "hour-of-cycle (sine)",
        "hour_cos": "hour-of-cycle (cosine)",
        "Time": "Time (raw seconds)",
    }.get(name, name)
