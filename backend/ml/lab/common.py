"""Shared data access and helpers for laboratory experiments."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import contract as C
from ..data import Splits, load_csv, prepare_splits
from ..features import hour_of_cycle

# Columns used for distance-based (M4) and regression (M2/M3) experiments.
PCA = list(C.PCA_FEATURES)
CLUSTER_FEATURES = [*PCA, "amount_log"]


def with_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_log"] = np.log1p(out["Amount"])
    out["hour"] = hour_of_cycle(out["Time"]).astype(int)
    return out


class LabData:
    """Cleaned dataset split with the same seed, fractions and duplicate policy as production training."""

    def __init__(self, raw: pd.DataFrame, seed: int = 42, source: str = "dataset"):
        self.seed = seed
        self.source = source
        self.splits: Splits = prepare_splits(raw, seed=seed)
        self.train = with_derived(self.splits.train)
        self.validation = with_derived(self.splits.validation)
        # Full cleaned frame (train + validation only; the test split is never exposed to the lab).
        self.development = pd.concat([self.train, self.validation], ignore_index=True)

    def sample(self, part: str, n: int | None, *, stratify: bool = True, seed: int | None = None) -> pd.DataFrame:
        df = self.train if part == "train" else self.validation
        if n is None or n >= len(df):
            return df
        strat = df[C.TARGET] if stratify else None
        out, _ = train_test_split(df, train_size=n, stratify=strat, random_state=self.seed if seed is None else seed)
        return out.reset_index(drop=True)

    def enriched(self, part: str, n_legit: int, *, n_fraud: int | None = None, seed: int | None = None) -> pd.DataFrame:
        """Every fraud row of the split (or ``n_fraud`` of them) plus ``n_legit`` legitimate rows.

        Used where an experiment needs enough positives to be informative or fast; the fraud rate
        of the result is far higher than the real 0.17 %, which every experiment states.
        """
        df = self.train if part == "train" else self.validation
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pos = np.flatnonzero(df[C.TARGET].to_numpy() == 1)
        neg = np.flatnonzero(df[C.TARGET].to_numpy() == 0)
        if n_fraud is not None:
            pos = rng.choice(pos, min(n_fraud, len(pos)), replace=False)
        neg = rng.choice(neg, min(n_legit, len(neg)), replace=False)
        idx = rng.permutation(np.concatenate([pos, neg]))
        return df.iloc[idx].reset_index(drop=True)

    def split_info(self, train: pd.DataFrame | int, validation: pd.DataFrame | int, note: str = "") -> dict[str, Any]:
        def rows(x: pd.DataFrame | int) -> tuple[int, int | None]:
            if isinstance(x, pd.DataFrame):
                return int(len(x)), int(x[C.TARGET].sum())
            return int(x), None

        tr, tr_f = rows(train)
        va, va_f = rows(validation)
        return {
            "source": self.source,
            "seed": self.seed,
            "train_rows_used": tr,
            "train_fraud_rows": tr_f,
            "validation_rows_used": va,
            "validation_fraud_rows": va_f,
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


def scaled_matrix(train: pd.DataFrame, columns: list[str], *others: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Standardise ``columns`` with statistics learned on ``train`` only."""
    sc = StandardScaler().fit(train[columns])
    return (sc.transform(train[columns]), *(sc.transform(o[columns]) for o in others))


def numeric_matrix_pipeline(scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps or [("identity", "passthrough")])


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


def scatter_sample(n_total: int, max_points: int, seed: int = 0, must_include: np.ndarray | None = None) -> np.ndarray:
    """Indices for a scatter plot; ``must_include`` (e.g. all fraud rows) is always kept."""
    if n_total <= max_points:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    keep = np.asarray(must_include if must_include is not None else [], dtype=int)
    rest = np.setdiff1d(np.arange(n_total), keep)
    extra = rng.choice(rest, max(0, max_points - len(keep)), replace=False)
    return np.sort(np.concatenate([keep, extra]))
