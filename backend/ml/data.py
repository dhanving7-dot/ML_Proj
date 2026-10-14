"""Dataset ingestion, validation, cleaning, splitting and descriptive statistics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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
    return pd.read_csv(path, low_memory=False)


# --------------------------------------------------------------------------- validation


def _check(name: str, status: str, message: str, **details: Any) -> dict:
    return {"name": name, "status": status, "message": message, "details": details or None}


def validate_dataframe(df: pd.DataFrame, *, require_target: bool = True, drop_duplicates: bool = True) -> dict:
    """Validate a raw dataframe against the feature contract.

    Returns a JSON-serialisable report. ``status`` is ``failed`` when the data cannot be used
    for training, ``passed_with_warnings`` when it can but issues were found.
    """
    checks: list[dict] = []
    required = list(C.RAW_REQUIRED_COLUMNS if require_target else C.FEATURE_NAMES)
    missing_cols = [c for c in required if c not in df.columns]
    checks.append(
        _check(
            "required_columns",
            "fail" if missing_cols else "pass",
            f"Missing required columns: {missing_cols}" if missing_cols else f"All {len(required)} contract columns are present.",
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

    n = len(df)
    checks.append(_check("row_count", "fail" if n < MIN_TRAINING_ROWS else "pass", f"{n:,} rows (minimum {MIN_TRAINING_ROWS}).", rows=n))

    # ---------------------------------------------------------------- target
    if require_target and C.TARGET in df.columns:
        target = pd.to_numeric(df[C.TARGET], errors="coerce")
        bad = int((~target.isin([0, 1])).sum())
        checks.append(
            _check(
                "target_values",
                "fail" if bad else "pass",
                f"{bad} rows have a Class outside {{0, 1}}." if bad else "Class is binary {0 = legitimate, 1 = fraud}.",
                invalid_rows=bad,
            )
        )
        if not bad and n:
            positives = int((target == 1).sum())
            rate = positives / n
            checks.append(
                _check(
                    "target_imbalance",
                    "fail" if positives < C.MIN_POSITIVE_ROWS else "pass",
                    (
                        f"Only {positives} fraud rows; at least {C.MIN_POSITIVE_ROWS} are needed to train and evaluate."
                        if positives < C.MIN_POSITIVE_ROWS
                        else f"{positives:,} fraud vs {n - positives:,} legitimate rows ({rate:.4%} fraud). "
                        "Severe imbalance: accuracy is uninformative, so PR-AUC, precision and recall are reported."
                    ),
                    fraud_rows=positives,
                    legitimate_rows=n - positives,
                    fraud_rate=round(rate, 6),
                )
            )

    # ---------------------------------------------------------------- feature types, finiteness, ranges
    present = [f for f in C.NUMERIC_FEATURES if f.name in df.columns]
    type_issues: dict[str, int] = {}
    missing: dict[str, int] = {}
    non_finite: dict[str, int] = {}
    range_issues: dict[str, int] = {}
    constant: list[str] = []
    for f in present:
        raw = df[f.name]
        col = pd.to_numeric(raw, errors="coerce")
        non_numeric = int((col.isna() & raw.notna()).sum())
        if non_numeric:
            type_issues[f.name] = non_numeric
        nulls = int(raw.isna().sum())
        if nulls:
            missing[f.name] = nulls
        inf = int(np.isinf(col.to_numpy(dtype=float, na_value=np.nan)).sum())
        if inf:
            non_finite[f.name] = inf
        out = 0
        if f.minimum is not None:
            out += int((col < f.minimum).sum())
        if f.maximum is not None:
            out += int((col > f.maximum).sum())
        if out:
            range_issues[f.name] = out
        if col.nunique(dropna=True) <= 1 and n > 1:
            constant.append(f.name)

    def frac_fail(counts: dict[str, int]) -> bool:
        return bool(n) and any(v / n > 0.2 for v in counts.values())

    checks.append(
        _check(
            "numeric_types",
            "fail" if frac_fail(type_issues) else ("warn" if type_issues else "pass"),
            "Non-numeric values in numeric columns (those rows are dropped before training)." if type_issues else "Every feature column parses as numeric.",
            columns=type_issues,
        )
    )
    checks.append(
        _check(
            "missing_values",
            "fail" if frac_fail(missing) else ("warn" if missing else "pass"),
            "Missing values found (rows with a missing feature are dropped before training; the API rejects them)." if missing else "No missing values.",
            columns=missing,
            total=int(sum(missing.values())),
        )
    )
    checks.append(
        _check(
            "non_finite_values",
            "fail" if frac_fail(non_finite) else ("warn" if non_finite else "pass"),
            "Infinite values found (dropped before training)." if non_finite else "All values are finite.",
            columns=non_finite,
        )
    )
    checks.append(
        _check(
            "value_ranges",
            "fail" if frac_fail(range_issues) else ("warn" if range_issues else "pass"),
            "Negative Time or Amount values found (dropped before training)." if range_issues else "Time and Amount are non-negative.",
            columns=range_issues,
        )
    )
    checks.append(
        _check(
            "constant_columns",
            "fail" if constant else "pass",
            f"Constant feature columns carry no information: {constant}" if constant else "No constant feature columns.",
            columns=constant,
        )
    )

    # ---------------------------------------------------------------- duplicates
    dup_rows = int(df.duplicated().sum())
    details: dict[str, Any] = {"duplicate_rows": dup_rows}
    if dup_rows and C.TARGET in df.columns and all(c in df.columns for c in C.FEATURE_NAMES):
        details["duplicate_fraud_rows"] = int(df[df.duplicated()][C.TARGET].eq(1).sum())
        feat_dups = df[df.duplicated(subset=list(C.FEATURE_NAMES), keep=False)]
        conflicts = feat_dups.groupby(list(C.FEATURE_NAMES))[C.TARGET].nunique() if len(feat_dups) else pd.Series(dtype=int)
        details["identical_features_with_conflicting_labels"] = int((conflicts > 1).sum())
    action = "dropped before splitting (keep first) so identical rows cannot appear in both training and test" if drop_duplicates else "kept (drop_duplicates is disabled)"
    checks.append(
        _check(
            "duplicates",
            "warn" if dup_rows else "pass",
            f"{dup_rows:,} exact duplicate rows; they are {action}." if dup_rows else "No duplicate rows.",
            action="drop" if drop_duplicates else "keep",
            **details,
        )
    )

    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce")
        checks.append(
            _check(
                "time_order",
                "pass",
                f"Time spans {float(t.min()):,.0f}–{float(t.max()):,.0f} s ({float(t.max() - t.min()) / 3600:.1f} h); rows are {'sorted' if t.is_monotonic_increasing else 'not sorted'} by Time.",
                min=float(t.min()) if t.notna().any() else None,
                max=float(t.max()) if t.notna().any() else None,
                sorted=bool(t.is_monotonic_increasing),
            )
        )

    statuses = {c["status"] for c in checks}
    status = "failed" if "fail" in statuses else ("passed_with_warnings" if "warn" in statuses else "passed")
    return {"status": status, "n_rows": n, "n_columns": int(df.shape[1]), "checks": checks}


# --------------------------------------------------------------------------- cleaning / splitting


def clean_for_training(df: pd.DataFrame, *, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Coerce to the contract and drop rows that cannot be used. Every removal is counted.

    Order: invalid target -> unparsable / missing / non-finite / out-of-range features -> exact duplicates.
    Returns ``(clean_frame, report)``.
    """
    from .features import coerce_features  # same coercion the serving path uses

    report: dict[str, Any] = {"rows_in": int(len(df))}
    out = coerce_features(df)
    if C.TRANSACTION_ID in df.columns:
        out.insert(0, C.TRANSACTION_ID, df[C.TRANSACTION_ID].astype(str))
    if C.TARGET in df.columns:
        target = pd.to_numeric(df[C.TARGET], errors="coerce")
        valid = target.isin([0, 1])
        report["dropped_invalid_target"] = int((~valid).sum())
        out[C.TARGET] = target
        out = out[valid.to_numpy()]
        out[C.TARGET] = out[C.TARGET].astype(int)
    bad = out[list(C.FEATURE_NAMES)].isna().any(axis=1)
    report["dropped_invalid_features"] = int(bad.sum())
    out = out[~bad.to_numpy()]
    if drop_duplicates:
        dup_cols = [c for c in (*C.FEATURE_NAMES, C.TARGET) if c in out.columns]
        dup = out.duplicated(subset=dup_cols, keep="first")
        report["dropped_duplicates"] = int(dup.sum())
        report["dropped_duplicate_frauds"] = int(out.loc[dup.to_numpy(), C.TARGET].eq(1).sum()) if C.TARGET in out.columns else 0
        out = out[~dup.to_numpy()]
    else:
        report["dropped_duplicates"] = 0
    out = out.reset_index(drop=True)
    report["rows_out"] = int(len(out))
    if C.TARGET in out.columns:
        report["fraud_rows_out"] = int(out[C.TARGET].sum())
    report["duplicate_policy"] = "drop exact duplicates (keep first) before splitting" if drop_duplicates else "keep duplicates"
    return out, report


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    sizes: tuple[float, float, float]
    cleaning: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict:
        def part(d: pd.DataFrame) -> dict:
            return {"rows": int(len(d)), "fraud_rows": int(d[C.TARGET].sum()), "positive_rate": round(float(d[C.TARGET].mean()), 6)}

        return {
            "strategy": "stratified train/validation/test on Class",
            "seed": self.seed,
            "fractions": {"train": self.sizes[0], "validation": self.sizes[1], "test": self.sizes[2]},
            "train": part(self.train),
            "validation": part(self.validation),
            "test": part(self.test),
            "cleaning": self.cleaning,
            "usage": {
                "train": "fit the scaler, Logistic Regression (with CV for C) and Isolation Forest; build the score references",
                "validation": "tune the LR threshold, the fusion weight and the REVIEW / HIGH_RISK thresholds",
                "test": "single final evaluation; never used for fitting, selection or tuning",
            },
        }


def split_data(df: pd.DataFrame, *, seed: int = 42, validation: float = 0.15, test: float = 0.15, cleaning: dict | None = None) -> Splits:
    """Stratified train/validation/test split. The test part is held out for the final report only."""
    train_val, test_df = train_test_split(df, test_size=test, stratify=df[C.TARGET], random_state=seed)
    rel_val = validation / (1.0 - test)
    train_df, val_df = train_test_split(train_val, test_size=rel_val, stratify=train_val[C.TARGET], random_state=seed)
    return Splits(
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        seed,
        (round(1 - validation - test, 4), validation, test),
        cleaning or {},
    )


def prepare_splits(
    raw: pd.DataFrame,
    *,
    seed: int = 42,
    validation: float = 0.15,
    test: float = 0.15,
    drop_duplicates: bool = True,
    sample_rows: int | None = None,
) -> Splits:
    """The one function every consumer uses (training, the sample loader, replay, the lab) so splits always match."""
    df, cleaning = clean_for_training(raw, drop_duplicates=drop_duplicates)
    if sample_rows and sample_rows < len(df):
        df, _ = train_test_split(df, train_size=sample_rows, stratify=df[C.TARGET], random_state=seed)
        df = df.reset_index(drop=True)
        cleaning["sampled_rows"] = int(sample_rows)
    return split_data(df, seed=seed, validation=validation, test=test, cleaning=cleaning)


# --------------------------------------------------------------------------- statistics


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def column_statistics(df: pd.DataFrame, bins: int = 20) -> list[dict]:
    """Per-column descriptive statistics including skewness, excess kurtosis and per-class means."""
    out: list[dict] = []
    has_target = C.TARGET in df.columns
    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 0:
            v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            counts, edges = np.histogram(v, bins=bins)
            row: dict[str, Any] = {
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
            if has_target and name != C.TARGET:
                t = pd.to_numeric(df[C.TARGET], errors="coerce")
                col = pd.to_numeric(s, errors="coerce")
                row["mean_legitimate"] = _finite(col[t == 0].mean())
                row["mean_fraud"] = _finite(col[t == 1].mean())
            out.append(row)
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
