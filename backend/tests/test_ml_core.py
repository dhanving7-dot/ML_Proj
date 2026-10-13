"""Unit tests for the ML package: validation, preprocessing, metrics, drift, training and packaging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import contract as C
from ml import metrics as M
from ml.data import clean_for_training, split_data, validate_dataframe
from ml.drift import kl_divergence, psi, psi_status
from ml.features import AcademicFeatureEngineer, build_pipeline, coerce_features
from ml.registry import ArtifactIntegrityError, load_artifact, save_artifact
from ml.training import DataValidationError, TrainConfig, train_model

from .conftest import FAST_CONFIG


# ----------------------------------------------------------------------------- dataset validation
def _status(report: dict, name: str) -> str:
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def test_valid_dataset_passes_with_known_warnings(sample_df):
    report = validate_dataframe(sample_df)
    assert report["status"] in ("passed", "passed_with_warnings")
    assert _status(report, "required_columns") == "pass"
    assert _status(report, "target_values") == "pass"
    assert _status(report, "cgpa_tier_consistency") in ("pass", "warn")


def test_missing_column_fails(sample_df):
    report = validate_dataframe(sample_df.drop(columns=["CGPA"]))
    assert report["status"] == "failed"
    assert _status(report, "required_columns") == "fail"


def test_invalid_target_fails(sample_df):
    bad = sample_df.copy()
    bad.loc[:10, C.TARGET] = 7
    assert _status(validate_dataframe(bad), "target_values") == "fail"


def test_out_of_range_and_unknown_category_warn(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "CGPA"] = 42.0
    bad.loc[1, "City"] = "Atlantis"
    report = validate_dataframe(bad)
    assert _status(report, "value_ranges") == "warn"
    assert _status(report, "categorical_domains") == "warn"


def test_too_few_rows_fails(sample_df):
    assert _status(validate_dataframe(sample_df.head(50)), "row_count") == "fail"


# ----------------------------------------------------------------------------- cleaning / missing data
def test_clean_coerces_out_of_range_and_text_to_nan(sample_df):
    bad = sample_df.head(5).copy()
    bad["CGPA"] = bad["CGPA"].astype(object)
    bad.loc[bad.index[0], "CGPA"] = 99
    bad.loc[bad.index[1], "CGPA"] = "n/a"
    clean = clean_for_training(bad)
    assert np.isnan(clean.loc[0, "CGPA"]) and np.isnan(clean.loc[1, "CGPA"])
    assert clean[C.TARGET].isin([0, 1]).all()


def test_split_is_stratified_and_disjoint(sample_df):
    df = clean_for_training(sample_df)
    s = split_data(df, seed=1)
    ids = [set(p[C.ID_COLUMN]) for p in (s.train, s.validation, s.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [p[C.TARGET].mean() for p in (s.train, s.validation, s.test)]
    assert max(rates) - min(rates) < 0.02


# ----------------------------------------------------------------------------- preprocessing consistency
def test_feature_engineer_slope_and_volatility():
    row = {s: 6.0 + i * 0.5 for i, s in enumerate(C.SEMESTERS)}
    out = AcademicFeatureEngineer().fit(pd.DataFrame([row])).transform(pd.DataFrame([row]))
    assert out["sgpa_trend"].iloc[0] == pytest.approx(0.5)
    assert out["sgpa_volatility"].iloc[0] == pytest.approx(np.std([6.0 + i * 0.5 for i in range(8)], ddof=1))


def test_preprocessing_fitted_on_train_only(sample_df):
    from sklearn.linear_model import LogisticRegression

    df = clean_for_training(sample_df)
    s = split_data(df)
    feats = list(C.FEATURE_NAMES)
    pipe = build_pipeline(LogisticRegression(max_iter=2000), feats).fit(s.train[feats], s.train[C.TARGET])
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    cgpa_idx = list(pipe.named_steps["preprocess"].transformers_[0][2]).index("CGPA")
    assert scaler.mean_[cgpa_idx] == pytest.approx(s.train["CGPA"].mean())
    nullable_idx = list(pipe.named_steps["preprocess"].transformers_[0][2]).index("CodingTestScore")
    assert imputer.statistics_[nullable_idx] == pytest.approx(s.train["CodingTestScore"].median())


def test_train_and_serve_paths_produce_identical_vectors(sample_df):
    """Golden E2E check: CSV training path vs JSON/dict serving path must give the same feature vector."""
    from sklearn.linear_model import LogisticRegression

    df = clean_for_training(sample_df)
    feats = list(C.FEATURE_NAMES)
    pipe = build_pipeline(LogisticRegression(max_iter=2000), feats).fit(df[feats], df[C.TARGET])
    rows = sample_df.head(25)
    train_path = clean_for_training(rows)[feats]
    records = rows[feats].astype(object).where(rows[feats].notna(), None).to_dict(orient="records")
    serve_path = coerce_features(records, feats)
    a = pipe[:-1].transform(train_path)
    b = pipe[:-1].transform(serve_path)
    assert np.max(np.abs(a - b)) == 0.0


def test_unknown_category_encodes_to_zeros(sample_df):
    from sklearn.linear_model import LogisticRegression

    df = clean_for_training(sample_df)
    feats = list(C.FEATURE_NAMES)
    pipe = build_pipeline(LogisticRegression(max_iter=2000), feats).fit(df[feats], df[C.TARGET])
    row = df[feats].head(1).copy()
    row["City"] = "Atlantis"
    names = [str(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    vec = pipe[:-1].transform(row)[0]
    city_cols = [i for i, n in enumerate(names) if n.startswith("City_")]
    assert np.all(vec[city_cols] == 0)


def test_coerce_features_adds_missing_columns_as_nan():
    out = coerce_features([{"CGPA": 8.1}], ["CGPA", "Workshops", "City"])
    assert list(out.columns) == ["CGPA", "Workshops", "City"]
    assert np.isnan(out.loc[0, "Workshops"]) and pd.isna(out.loc[0, "City"])


# ----------------------------------------------------------------------------- metrics
def test_classification_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.1, 0.7, 0.2])
    m = M.classification_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(0.9375)


def test_regression_metrics_and_mape_guard():
    y = np.array([10.0, 20.0, 30.0])
    m = M.regression_metrics(y, np.array([12.0, 18.0, 33.0]))
    assert m["mae"] == pytest.approx(7 / 3, abs=1e-4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3), abs=1e-4)
    assert m["mape"] == pytest.approx((0.2 + 0.1 + 0.1) / 3, abs=1e-4)
    assert M.regression_metrics(np.array([0.0, 1.0]), np.array([0.5, 1.0]))["mape"] is None


def test_mcnemar_table_and_significance():
    y = np.ones(100, int)
    a = np.ones(100, int)
    b = np.ones(100, int)
    b[:40] = 0  # B wrong on 40 rows where A is right
    res = M.mcnemar_test(y, a, b)
    assert res["table"]["a_only_correct"] == 40 and res["table"]["b_only_correct"] == 0
    assert res["significant_at_0_05"] is True


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    res = M.paired_bootstrap_rmse(y, y + rng.normal(scale=2, size=300), y + rng.normal(scale=0.1, size=300), n_boot=300)
    assert res["ci_95"][0] > 0 and res["significant"]


def test_calibration_points_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert M.calibration_points(y, p)["ece"] < 0.02


# ----------------------------------------------------------------------------- drift
def test_psi_and_kl_properties():
    e = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(e, e) == pytest.approx(0.0, abs=1e-9)
    assert kl_divergence(e, e) == pytest.approx(0.0, abs=1e-9)
    shifted = np.array([0.7, 0.1, 0.1, 0.1])
    assert psi(e, shifted) > 0.25 and psi_status(psi(e, shifted)) == "significant_shift"
    assert psi_status(0.05) == "stable" and psi_status(0.15) == "investigate"


# ----------------------------------------------------------------------------- training + packaging
def test_train_save_load_and_integrity(sample_df, tmp_path):
    out = train_model(sample_df, TrainConfig.from_dict(FAST_CONFIG))
    md = out.metadata
    assert 0.5 < md["metrics"]["test"]["roc_auc"] <= 1.0
    assert md["splits"]["usage"]["test"].startswith("single final evaluation")
    assert set(md["features"]) == set(C.FEATURE_NAMES)
    assert all(c not in md["features"] for c in C.EXCLUDED_COLUMNS)
    info = save_artifact(tmp_path, "v1", out.model, md)
    loaded = load_artifact(info["model_path"], info["sha256"])
    X = out.splits.test[md["features"]].head(30)
    assert np.allclose(loaded.predict_proba(X), out.model.predict_proba(X))
    with open(info["model_path"], "ab") as fh:  # tamper with the artifact
        fh.write(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        load_artifact(info["model_path"], info["sha256"])


def test_training_rejects_invalid_dataset(sample_df):
    with pytest.raises(DataValidationError):
        train_model(sample_df.drop(columns=["CodingTestScore"]), TrainConfig.from_dict(FAST_CONFIG))


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"feature_set": "future"})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"unknown_key": 1})


def test_artifact_write_is_atomic(tmp_path):
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("cannot pickle")

    with pytest.raises(RuntimeError):
        save_artifact(tmp_path, "v9", Unpicklable(), {})
    assert not (tmp_path / "v9").exists()
    assert not any(p.name.startswith(".v9.tmp") for p in tmp_path.iterdir())


def test_golden_check_detects_preprocessing_skew(sample_df):
    """If the served preprocessing differs from training (e.g. a re-fitted scaler), the load-time check fails."""
    import copy

    from app.services.model_service import golden_check

    out = train_model(sample_df, TrainConfig.from_dict(FAST_CONFIG))
    assert golden_check(out.model, out.metadata)["passed"] is True
    skewed = copy.deepcopy(out.model)
    scaler = skewed.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    scaler.mean_ = scaler.mean_ + 0.5  # simulate scaler parameters that differ from training
    res = golden_check(skewed, out.metadata)
    assert res["passed"] is False and res["max_feature_vector_diff"] > 0


def test_mlflow_logging_roundtrip(tmp_path):
    from ml.tracking import log_training_run

    res = log_training_run(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=tmp_path / "mlruns",
        run_name="test-run",
        params={"model": "logistic_regression"},
        metrics={"test": {"roc_auc": 0.9, "confusion_matrix": {"tp": 3}}},
        tags={"decision": "promoted"},
    )
    assert "run_id" in res, res
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    run = mlflow.get_run(res["run_id"])
    assert run.data.metrics["test.roc_auc"] == 0.9 and run.data.params["model"] == "logistic_regression"
