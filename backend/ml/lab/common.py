"""Shared data access and helpers for laboratory experiments."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import contract as C
from ..data import Splits, clean_for_training, load_csv, split_data

# Numeric features used for distance-based (M4) and linear-regression (M2) experiments.
CORE_NUMERIC = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular",
]


class LabData:
    """Cleaned dataset split with the same seed and fractions as production training."""

    def __init__(self, raw: pd.DataFrame, seed: int = 42, source: str = "dataset"):
        self.seed = seed
        self.source = source
        self.clean = clean_for_training(raw)
        self.splits: Splits = split_data(self.clean, seed=seed)

    @property
    def train(self) -> pd.DataFrame:
        return self.splits.train

    @property
    def validation(self) -> pd.DataFrame:
        return self.splits.validation

    def sample(self, part: str, n: int | None, *, stratify: bool = True, seed: int | None = None) -> pd.DataFrame:
        df = self.train if part == "train" else self.validation
        if n is None or n >= len(df):
            return df
        strat = df[C.TARGET] if stratify else None
        out, _ = train_test_split(df, train_size=n, stratify=strat, random_state=self.seed if seed is None else seed)
        return out.reset_index(drop=True)

    def split_info(self, train_rows: int, validation_rows: int, note: str = "") -> dict[str, Any]:
        return {
            "source": self.source,
            "seed": self.seed,
            "train_rows_used": int(train_rows),
            "validation_rows_used": int(validation_rows),
            "test_split": "not used (reserved for the production model's final evaluation)",
            "note": note,
        }


_cache: dict[str, LabData] = {}
_lock = threading.Lock()


def get_lab_data(path: str | Path, cache_key: str) -> LabData:
    with _lock:
        if cache_key not in _cache:
            _cache.clear()
            _cache[cache_key] = LabData(load_csv(path), source=Path(path).name)
        return _cache[cache_key]


def numeric_matrix_pipeline(scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def numeric_preprocessor(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", numeric_matrix_pipeline(), columns)], remainder="drop", verbose_feature_names_out=False)


def r(x: Any, nd: int = 4) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return [r(v, nd) for v in x.tolist()]
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return round(f, nd) if np.isfinite(f) else None


def scatter_sample(n_total: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n_total <= max_points:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, max_points, replace=False))
