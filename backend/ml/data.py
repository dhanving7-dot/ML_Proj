"""Dataset ingestion, validation, cleaning, splitting and descriptive statistics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.model_selection import train_test_split

from . import contract as C

MIN_TRAINING_ROWS = 200


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV keeping categorical columns as plain strings."""
    return pd.read_csv(path, low_memory=False)


# --------------------------------------------------------------------------- validation


def _check(name: str, status: str, message: str, **details: Any) -> dict:
    return {"name": name, "status": status, "message": message, "details": details or None}


def validate_dataframe(df: pd.DataFrame, *, require_target: bool = True) -> dict:
    """Validate a raw dataframe against the feature contract.

    Returns a JSON-serialisable report. ``status`` is ``failed`` when the data cannot be
    used for training, ``passed_with_warnings`` when it can but issues were found.
    """
    checks: list[dict] = []
    required = list(C.RAW_REQUIRED_COLUMNS if require_target else C.FEATURE_NAMES)
    missing_cols = [c for c in required if c not in df.columns]
    checks.append(
        _check(
            "required_columns",
            "fail" if missing_cols else "pass",
            f"Missing required columns: {missing_cols}" if missing_cols else "All contract columns are present.",
            missing=missing_cols,
        )
    )
    known = set(C.RAW_REQUIRED_COLUMNS) | set(C.EXCLUDED_COLUMNS)
    extra = [c for c in df.columns if c not in known]
    checks.append(
        _check(
            "unexpected_columns",
            "warn" if extra else "pass",
            f"Columns not in the contract will be ignored: {extra}" if extra else "No unexpected columns.",
            columns=extra,
        )
    )
    excluded_present = {c: r for c, r in C.EXCLUDED_COLUMNS.items() if c in df.columns}
    if excluded_present:
        checks.append(
            _check(
                "excluded_columns",
                "pass",
                "Columns present but excluded from model inputs (leakage / identifier / semantics).",
                columns=excluded_present,
            )
        )

    n = len(df)
    checks.append(
        _check(
            "row_count",
            "fail" if n < MIN_TRAINING_ROWS else "pass",
            f"{n} rows (minimum {MIN_TRAINING_ROWS}).",
            rows=n,
        )
    )

    if require_target and C.TARGET in df.columns:
        target = pd.to_numeric(df[C.TARGET], errors="coerce")
        bad = int((~target.isin([0, 1])).sum())
        checks.append(
            _check(
                "target_values",
                "fail" if bad else "pass",
                f"{bad} rows have a target outside {{0, 1}}." if bad else "Target is binary {0, 1}.",
                invalid_rows=bad,
            )
        )
        if not bad and n:
            rate = float(target.mean())
            checks.append(
                _check(
                    "class_balance",
                    "warn" if min(rate, 1 - rate) < 0.05 else "pass",
                    f"Positive (placed) rate {rate:.3f}.",
                    positive_rate=rate,
                )
            )

    type_issues: dict[str, int] = {}
    range_issues: dict[str, int] = {}
    null_issues: dict[str, int] = {}
    for f in C.NUMERIC_FEATURES:
        if f.name not in df.columns:
            continue
        raw = df[f.name]
        col = pd.to_numeric(raw, errors="coerce")
        non_numeric = int((col.isna() & raw.notna()).sum())
        if non_numeric:
            type_issues[f.name] = non_numeric
        out = int(((col < f.minimum) | (col > f.maximum)).sum())
        if out:
            range_issues[f.name] = out
        nulls = int(col.isna().sum())
        if nulls and not f.nullable:
            null_issues[f.name] = nulls
    enum_issues: dict[str, dict[str, int]] = {}
    for f in (*C.NOMINAL_FEATURES, *C.ORDINAL_FEATURES):
        if f.name not in df.columns:
            continue
        vals = df[f.name].dropna().astype(str)
        unknown = vals[~vals.isin(f.categories)]
        if len(unknown):
            enum_issues[f.name] = {str(k): int(v) for k, v in unknown.value_counts().head(10).items()}
        nulls = int(df[f.name].isna().sum())
        if nulls and not f.nullable:
            null_issues[f.name] = nulls

    def frac_fail(counts: dict[str, int]) -> bool:
        return bool(n) and any(v / n > 0.2 for v in counts.values())

    checks.append(
        _check(
            "numeric_types",
            "fail" if frac_fail(type_issues) else ("warn" if type_issues else "pass"),
            "Non-numeric values in numeric columns (treated as missing)." if type_issues else "Numeric columns parse.",
            columns=type_issues,
        )
    )
    checks.append(
        _check(
            "value_ranges",
            "fail" if frac_fail(range_issues) else ("warn" if range_issues else "pass"),
            "Values outside contract ranges (treated as missing)." if range_issues else "All values within ranges.",
            columns=range_issues,
        )
    )
    checks.append(
        _check(
            "categorical_domains",
            "warn" if enum_issues else "pass",
            "Unknown categories found (the one-hot encoder ignores them)." if enum_issues else "Categories match.",
            columns=enum_issues,
        )
    )
    checks.append(
        _check(
            "unexpected_nulls",
            "fail" if frac_fail(null_issues) else ("warn" if null_issues else "pass"),
            "Nulls in non-nullable columns (imputed during training)." if null_issues else "No unexpected nulls.",
            columns=null_issues,
        )
    )
    nullable_missing = {
        f.name: int(pd.to_numeric(df[f.name], errors="coerce").isna().sum())
        for f in C.NUMERIC_FEATURES
        if f.nullable and f.name in df.columns
    }
    checks.append(
        _check(
            "expected_missingness",
            "pass",
            "Missing values in nullable columns (handled by imputation + missing indicators).",
            columns=nullable_missing,
        )
    )

    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[C.ID_COLUMN].duplicated().sum()) if C.ID_COLUMN in df.columns else 0
    checks.append(
        _check(
            "duplicates",
            "warn" if dup_rows or dup_ids else "pass",
            f"{dup_rows} duplicate rows, {dup_ids} duplicate IDs.",
            duplicate_rows=dup_rows,
            duplicate_ids=dup_ids,
        )
    )

    if all(s in df.columns for s in C.SEMESTERS) and "CGPA" in df.columns:
        sg = df[list(C.SEMESTERS)].apply(pd.to_numeric, errors="coerce")
        gap = (pd.to_numeric(df["CGPA"], errors="coerce") - sg.mean(axis=1)).abs()
        inconsistent = int((gap > 1.0).sum())
        checks.append(
            _check(
                "cgpa_consistency",
                "warn" if inconsistent else "pass",
                f"{inconsistent} rows where CGPA differs from the mean SGPA by more than 1.0.",
                inconsistent_rows=inconsistent,
                median_gap=float(gap.median()) if len(gap) else None,
            )
        )

    if "CGPA_Tier" in df.columns and "CGPA" in df.columns:
        lo, hi = C.CGPA_TIER_RULE["cutoffs"]
        low, mid, high = C.CGPA_TIER_RULE["labels"]
        cg = pd.to_numeric(df["CGPA"], errors="coerce")
        expected = np.select([cg < lo, cg < hi], [low, mid], high)
        valid = cg.notna() & df["CGPA_Tier"].notna()
        agree = float((expected[valid.to_numpy()] == df.loc[valid, "CGPA_Tier"].astype(str).to_numpy()).mean()) if valid.any() else 1.0
        mismatched = int(valid.sum() - round(agree * valid.sum()))
        checks.append(
            _check(
                "cgpa_tier_consistency",
                "warn" if agree < 0.99 else "pass",
                f"CGPA_Tier matches the rule CGPA<{lo} Low / <{hi} Mid / else High for {agree:.1%} of rows "
                f"({mismatched} rows disagree). The column is excluded from model inputs.",
                agreement=round(agree, 4),
                mismatched_rows=mismatched,
                rule=f"<{lo} {low}, <{hi} {mid}, else {high}",
            )
        )

    statuses = {c["status"] for c in checks}
    status = "failed" if "fail" in statuses else ("passed_with_warnings" if "warn" in statuses else "passed")
    return {"status": status, "n_rows": n, "n_columns": int(df.shape[1]), "checks": checks}


# --------------------------------------------------------------------------- cleaning / splitting


def clean_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types to the contract. Out-of-range or unparsable numerics become NaN (imputed later).

    Rows without a valid binary target are dropped. Categorical values are kept as strings so
    the fitted encoders (not this function) decide how unknown values are handled.
    """
    from .features import coerce_features  # same coercion the serving path uses

    out = coerce_features(df)
    if C.ID_COLUMN in df.columns:
        out.insert(0, C.ID_COLUMN, df[C.ID_COLUMN])
    for extra in (C.TARGET, C.REGRESSION_TARGET, C.ANOMALY_LABEL, "CGPA_Tier"):
        if extra in df.columns:
            out[extra] = df[extra]
    if C.TARGET in out.columns:
        out[C.TARGET] = pd.to_numeric(out[C.TARGET], errors="coerce")
        out = out[out[C.TARGET].isin([0, 1])].copy()
        out[C.TARGET] = out[C.TARGET].astype(int)
    return out.reset_index(drop=True)


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    sizes: tuple[float, float, float]

    def describe(self) -> dict:
        def part(d: pd.DataFrame) -> dict:
            return {"rows": int(len(d)), "positive_rate": round(float(d[C.TARGET].mean()), 4)}

        return {
            "strategy": "stratified train/validation/test",
            "seed": self.seed,
            "fractions": {"train": self.sizes[0], "validation": self.sizes[1], "test": self.sizes[2]},
            "train": part(self.train),
            "validation": part(self.validation),
            "test": part(self.test),
            "usage": {
                "train": "model fitting and cross-validated model selection",
                "validation": "probability calibration and decision-threshold tuning",
                "test": "single final evaluation; never used for selection or tuning",
            },
        }


def split_data(df: pd.DataFrame, *, seed: int = 42, validation: float = 0.15, test: float = 0.15) -> Splits:
    """Stratified train/validation/test split. The test part is held out for the final report only."""
    train_val, test_df = train_test_split(df, test_size=test, stratify=df[C.TARGET], random_state=seed)
    rel_val = validation / (1.0 - test)
    train_df, val_df = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[C.TARGET], random_state=seed
    )
    return Splits(
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        seed,
        (round(1 - validation - test, 4), validation, test),
    )


# --------------------------------------------------------------------------- statistics


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def column_statistics(df: pd.DataFrame, bins: int = 20) -> list[dict]:
    """Per-column descriptive statistics including skewness and excess kurtosis."""
    out: list[dict] = []
    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        num = pd.to_numeric(s, errors="coerce") if s.dtype != object else None
        if num is not None and num.notna().sum() > 0 and pd.api.types.is_numeric_dtype(s):
            v = num.dropna().to_numpy(dtype=float)
            counts, edges = np.histogram(v, bins=bins)
            out.append(
                {
                    "name": name,
                    "kind": "numeric",
                    "missing": missing,
                    "missing_rate": round(missing / max(len(s), 1), 4),
                    "mean": _finite(v.mean()),
                    "std": _finite(v.std(ddof=1)) if len(v) > 1 else None,
                    "min": _finite(v.min()),
                    "q25": _finite(np.quantile(v, 0.25)),
                    "median": _finite(np.median(v)),
                    "q75": _finite(np.quantile(v, 0.75)),
                    "max": _finite(v.max()),
                    "skewness": _finite(sstats.skew(v)) if len(v) > 2 else None,
                    "kurtosis": _finite(sstats.kurtosis(v)) if len(v) > 3 else None,  # excess (Fisher)
                    "histogram": {"counts": counts.tolist(), "edges": [float(e) for e in edges]},
                }
            )
        else:
            vc = s.astype(str).where(s.notna()).value_counts()
            out.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "missing": missing,
                    "missing_rate": round(missing / max(len(s), 1), 4),
                    "unique": int(vc.size),
                    "top": [{"value": str(k), "count": int(c)} for k, c in vc.head(15).items()],
                }
            )
    return out
