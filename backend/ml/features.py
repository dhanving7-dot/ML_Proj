"""Shared feature pipeline: type coercion, feature engineering and preprocessing.

Training and serving both call :func:`coerce_features` and use the fitted
``Pipeline`` stored in the model artifact, so imputation values, scaling
parameters and encoder vocabularies are learned once (on training data only)
and reused unchanged in production.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import contract as C

ENGINEERED_FEATURES = ("sgpa_trend", "sgpa_volatility")


def coerce_features(data: pd.DataFrame | Iterable[Mapping[str, Any]], names: Iterable[str] = C.FEATURE_NAMES) -> pd.DataFrame:
    """Return a frame with exactly the contract columns, in contract order, with contract dtypes.

    Numeric values outside the contract range or unparsable become NaN; missing categoricals
    become ``np.nan`` (which ``SimpleImputer`` recognises, unlike ``None``).
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
    names = list(names)
    out: dict[str, pd.Series] = {}
    for name in names:
        f = C.FEATURES_BY_NAME[name]
        raw = df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        if isinstance(f, C.NumericFeature):
            col = pd.to_numeric(raw, errors="coerce").astype(float)
            out[name] = col.where((col >= f.minimum) & (col <= f.maximum))
        else:
            col = raw.astype(object)
            out[name] = col.where(col.notna(), np.nan)
    return pd.DataFrame(out, index=df.index)


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered academic features derived from per-semester SGPA values.

    * ``sgpa_trend``: least-squares slope of SGPA across the available semesters.
    * ``sgpa_volatility``: standard deviation of the semester SGPAs.

    It is stateless (nothing is learned), so it behaves identically in training and serving.
    """

    def __init__(self, semesters: tuple[str, ...] = C.SEMESTERS):
        self.semesters = semesters

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AcademicFeatureEngineer":
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        sem = [s for s in self.semesters if s in X.columns]
        values = X[sem].to_numpy(dtype=float)
        t = np.arange(len(sem), dtype=float)
        t_c = t - t.mean()
        mask = ~np.isnan(values)
        filled = np.where(mask, values, 0.0)
        counts = mask.sum(axis=1)
        means = np.divide(filled.sum(axis=1), counts, out=np.full(len(X), np.nan), where=counts > 0)
        centred = np.where(mask, values - means[:, None], 0.0)
        denom = (np.where(mask, t_c, 0.0) ** 2).sum(axis=1)
        slope = np.divide((centred * t_c).sum(axis=1), denom, out=np.full(len(X), np.nan), where=denom > 0)
        var = np.divide((centred**2).sum(axis=1), counts - 1, out=np.full(len(X), np.nan), where=counts > 1)
        X["sgpa_trend"] = slope
        X["sgpa_volatility"] = np.sqrt(var)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([*self.feature_names_in_, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor(
    feature_names: Iterable[str],
    *,
    engineered: bool = True,
    numeric_imputer: str = "median",
    scaler: str = "standard",
    add_missing_indicator: bool = True,
) -> ColumnTransformer:
    """ColumnTransformer that imputes, scales and encodes the contract features."""
    numeric, nominal, ordinal = C.split_feature_kinds(list(feature_names))
    if engineered:
        numeric = [*numeric, *ENGINEERED_FEATURES]
    num_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=numeric_imputer, add_indicator=add_missing_indicator))
    ]
    if scaler == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        num_steps.append(("scale", MinMaxScaler()))
    transformers: list[tuple[str, Any, list[str]]] = [("num", Pipeline(num_steps), numeric)]
    if nominal:
        transformers.append(
            (
                "nom",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                nominal,
            )
        )
    if ordinal:
        cats = [list(C.FEATURES_BY_NAME[n].categories) for n in ordinal]  # type: ignore[union-attr]
        transformers.append(
            (
                "ord",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                ordinal,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(estimator: Any, feature_names: Iterable[str], *, engineered: bool = True, **pre_kwargs: Any) -> Pipeline:
    """Full model pipeline: [feature engineering] -> preprocessing -> estimator."""
    feature_names = list(feature_names)
    steps: list[tuple[str, Any]] = []
    if engineered:
        sems = tuple(s for s in C.SEMESTERS if s in feature_names)
        steps.append(("engineer", AcademicFeatureEngineer(semesters=sems)))
    steps.append(("preprocess", build_preprocessor(feature_names, engineered=engineered, **pre_kwargs)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every fitted step except the estimator (the exact vector the model sees)."""
    return np.asarray(pipeline[:-1].transform(X), dtype=float)
