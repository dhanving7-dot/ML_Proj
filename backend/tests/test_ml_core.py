"""Unit tests for the ML package: validation, splitting, preprocessing, LR, Isolation Forest, fusion,
decisions, metrics, drift, training and packaging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from ml import contract as C
from ml import metrics as M
from ml.data import clean_for_training, prepare_splits, validate_dataframe
from ml.drift import kl_divergence, psi, psi_status
from ml.features import build_preprocessor, coerce_features, hour_of_cycle
from ml.fraud_model import DecisionPolicy, RarityReference, decide, fuse
from ml.registry import ArtifactIntegrityError, load_artifact, save_artifact
from ml.training import DataValidationError, TrainConfig, evaluate_scores, prior_offset, train_model, undersample

from .conftest import FAST_CONFIG


def _status(report: dict, name: str) -> str:
    return next(c["status"] for c in report["checks"] if c["name"] == name)


# ----------------------------------------------------------------------------- dataset validation
def test_real_sample_passes_and_reports_imbalance(sample_df):
    report = validate_dataframe(sample_df)
    assert report["status"] in ("passed", "passed_with_warnings")
    assert _status(report, "required_columns") == "pass"
    assert _status(report, "target_values") == "pass"
    imb = next(c for c in report["checks"] if c["name"] == "target_imbalance")
    assert imb["details"]["fraud_rows"] == 150 and imb["details"]["legitimate_rows"] == 6000


def test_full_dataset_shape_and_class_counts():
    """The documented facts of creditcard.csv (284,807 x 31; 492 fraud) are checked on the real file."""
    from .conftest import DATASET

    df = pd.read_csv(DATASET)
    assert df.shape == (284807, 31)
    assert list(df.columns) == [*C.FEATURE_NAMES, C.TARGET]
    assert df[C.TARGET].value_counts().to_dict() == {0: 284315, 1: 492}
    assert int(df.isna().sum().sum()) == 0
    report = validate_dataframe(df)
    dup = next(c for c in report["checks"] if c["name"] == "duplicates")
    assert dup["status"] == "warn" and dup["details"]["duplicate_rows"] == 1081
    assert dup["details"]["identical_features_with_conflicting_labels"] == 0


def test_missing_column_and_bad_target_fail(sample_df):
    assert _status(validate_dataframe(sample_df.drop(columns=["V14"])), "required_columns") == "fail"
    bad = sample_df.copy()
    bad.loc[:10, C.TARGET] = 7
    assert _status(validate_dataframe(bad), "target_values") == "fail"


def test_too_few_frauds_fails(sample_df):
    few = pd.concat([sample_df[sample_df[C.TARGET] == 1].head(5), sample_df[sample_df[C.TARGET] == 0].head(500)])
    assert _status(validate_dataframe(few), "target_imbalance") == "fail"


def test_negative_amount_text_and_infinity_are_reported(sample_df):
    bad = sample_df.copy()
    bad["V3"] = bad["V3"].astype(object)
    bad.loc[0, "Amount"] = -5
    bad.loc[1, "V2"] = np.inf
    bad.loc[2, "V3"] = "n/a"
    report = validate_dataframe(bad)
    assert _status(report, "value_ranges") == "warn"
    assert _status(report, "non_finite_values") == "warn"
    assert _status(report, "numeric_types") == "warn"
    clean, rep = clean_for_training(bad)
    assert rep["dropped_invalid_features"] == 3 and len(clean) == len(bad) - 3 - rep["dropped_duplicates"]


# ----------------------------------------------------------------------------- leakage-safe splitting
def test_duplicates_are_dropped_before_splitting(sample_df):
    dup = pd.concat([sample_df, sample_df.head(40)], ignore_index=True)
    clean, rep = clean_for_training(dup, drop_duplicates=True)
    assert rep["dropped_duplicates"] >= 40
    assert not clean.duplicated(subset=[*C.FEATURE_NAMES, C.TARGET]).any()
    kept, rep2 = clean_for_training(dup, drop_duplicates=False)
    assert rep2["dropped_duplicates"] == 0 and len(kept) == len(dup)


def test_split_is_stratified_disjoint_and_reproducible(sample_df):
    s = prepare_splits(sample_df, seed=1)
    keys = [set(map(tuple, p[list(C.FEATURE_NAMES)].round(10).to_numpy())) for p in (s.train, s.validation, s.test)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    rates = [p[C.TARGET].mean() for p in (s.train, s.validation, s.test)]
    assert max(rates) - min(rates) < 0.005
    assert abs(len(s.test) / (len(s.train) + len(s.validation) + len(s.test)) - 0.15) < 0.01
    again = prepare_splits(sample_df, seed=1)
    assert again.test.equals(s.test)


def test_scaler_is_fitted_on_training_rows_only(trained):
    sp = trained.splits
    scaler = trained.model.preprocessor.named_steps["scale"]
    names = trained.model.transformed_names()
    i = names.index("amount_log")
    assert scaler.mean_[i] == pytest.approx(np.log1p(sp.train["Amount"]).mean())
    assert scaler.mean_[i] != pytest.approx(np.log1p(pd.concat([sp.train, sp.validation, sp.test])["Amount"]).mean(), abs=1e-9)


def test_tuning_never_uses_test_rows(trained):
    md = trained.metadata
    assert md["splits"]["usage"]["test"].startswith("single final evaluation")
    # The fusion-weight curve and thresholds are computed from validation rows only.
    assert md["tuning"]["weight_source"].startswith(("tuned on validation", "fixed"))
    val_y = trained.splits.validation[C.TARGET].to_numpy()
    s = trained.model.score(trained.splits.validation[trained.model.features])
    best = max(md["curves"]["fusion_weight"], key=lambda d: (d["average_precision"], d["weight_lr"]))
    assert best["average_precision"] == pytest.approx(average_precision_score(val_y, fuse(s["lr_rarity"], s["if_rarity"], best["weight_lr"])), abs=1e-4)


# ----------------------------------------------------------------------------- preprocessing consistency
def test_hour_of_cycle_encoding():
    assert hour_of_cycle(np.array([0, 3600, 86400 + 7200]))[2] == pytest.approx(2.0)
    pre = build_preprocessor(scaler="none").fit(pd.DataFrame([{n: 0.0 for n in C.FEATURE_NAMES}]))
    out = pre.transform(pd.DataFrame([{**{n: 0.0 for n in C.FEATURE_NAMES}, "Time": 6 * 3600.0, "Amount": np.e - 1}]))
    names = pre.named_steps["engineer"].output_names()
    row = dict(zip(names, np.asarray(out)[0]))
    assert row["hour_sin"] == pytest.approx(1.0) and row["hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert row["amount_log"] == pytest.approx(1.0)


def test_train_and_serve_paths_produce_identical_vectors(trained, sample_df):
    """Golden E2E check: CSV training path vs JSON/dict serving path must give the same feature vector."""
    import io
    import json

    from app.services.model_service import serving_frame, validate_record

    rows = sample_df.head(40)
    train_path, _ = clean_for_training(pd.read_csv(io.StringIO(rows.to_csv(index=False))), drop_duplicates=False)
    records = json.loads(json.dumps(rows[list(C.FEATURE_NAMES)].to_dict(orient="records")))  # full float precision, like an HTTP client
    serve_path = serving_frame([validate_record(r) for r in records])
    a, b = trained.model.transform(train_path), trained.model.transform(serve_path)
    assert np.max(np.abs(a - b)) == 0.0


def test_coerce_features_marks_invalid_values():
    out = coerce_features([{"Time": -1, "Amount": "abc", "V1": float("inf")}], ["Time", "Amount", "V1", "V2"])
    assert out.isna().all(axis=None)


def test_golden_check_detects_preprocessing_skew(trained):
    import copy

    from app.services.model_service import golden_check

    assert golden_check(trained.model, trained.metadata)["passed"] is True
    skewed = copy.deepcopy(trained.model)
    skewed.preprocessor.named_steps["scale"].mean_ = skewed.preprocessor.named_steps["scale"].mean_ + 0.5
    res = golden_check(skewed, trained.metadata)
    assert res["passed"] is False and res["max_feature_vector_diff"] > 0


# ----------------------------------------------------------------------------- Logistic Regression
def test_prior_correction_formula():
    y = np.array([0] * 990 + [1] * 10)
    assert prior_offset(y, "class_weight") == pytest.approx(np.log(99))
    assert prior_offset(y, "undersample", keep_neg_fraction=0.2) == pytest.approx(-np.log(0.2))
    assert prior_offset(y, "none") == 0.0


def test_undersample_keeps_every_fraud():
    y = np.array([0] * 1000 + [1] * 10)
    idx = undersample(y, 5, seed=0)
    assert y[idx].sum() == 10 and (y[idx] == 0).sum() == 50


def test_lr_inference_matches_its_parts(trained):
    m = trained.model
    X = trained.splits.test[m.features].head(50)
    Xt = m.transform(X)
    s = m.score(X)
    expected_logit = m.lr.decision_function(Xt) - m.lr_logit_offset
    assert np.allclose(s["lr_logit"], expected_logit)
    assert np.allclose(s["lr_probability"], 1 / (1 + np.exp(-expected_logit)))
    assert np.all((s["lr_probability"] >= 0) & (s["lr_probability"] <= 1))
    # Exact additive explanation: contributions + intercept - offset = logit.
    contrib = sum(d["contribution"] for d in m.lr_contributions(Xt[0]))
    assert contrib + m.lr.intercept_[0] - m.lr_logit_offset == pytest.approx(s["lr_logit"][0])


def test_prior_correction_improves_calibration(trained):
    cal = trained.metadata["calibration"]
    assert cal["validation"]["ece"] < cal["validation_without_correction"]["ece"]


# ----------------------------------------------------------------------------- Isolation Forest
def test_isolation_forest_ignores_labels(sample_df):
    """With reference 'all_train' the detector is fitted without labels: shuffled labels give identical anomaly scores."""
    cfg = TrainConfig.from_dict({**FAST_CONFIG, "lr_c_grid": [1.0], "cv_folds": 2})
    a = train_model(sample_df, cfg)
    shuffled = sample_df.copy()
    rng = np.random.default_rng(0)
    # Shuffle labels within the training rows only would change the split; instead compare the IF fitted on the same rows.
    Xt = a.model.transform(a.splits.train[a.model.features])
    from sklearn.ensemble import IsolationForest

    params = a.model.iforest.get_params()
    b = IsolationForest(**params).fit(Xt)
    assert np.allclose(a.model.iforest.score_samples(Xt[:200]), b.score_samples(Xt[:200]))
    assert a.metadata["components"]["isolation_forest"]["labels_used_for_fitting"] is False
    del shuffled, rng


def test_anomaly_scores_and_percentiles(trained):
    m = trained.model
    X = trained.splits.validation[m.features]
    s = m.score(X)
    assert np.allclose(s["anomaly_score"], -m.iforest.score_samples(m.transform(X)))
    assert np.all((s["anomaly_percentile"] >= 0) & (s["anomaly_percentile"] <= 1))
    # Fraud is, on average, more anomalous than legitimate traffic.
    y = trained.splits.validation[C.TARGET].to_numpy()
    assert s["anomaly_percentile"][y == 1].mean() > s["anomaly_percentile"][y == 0].mean()


# ----------------------------------------------------------------------------- normalisation, fusion, decisions
def test_rarity_reference_properties():
    ref = RarityReference(np.arange(1000, dtype=float))
    r = ref.rarity(np.array([-5.0, 499.0, 989.0, 998.0, 5000.0]))
    assert r[0] == 0.0 and r[-1] == 1.0
    assert np.all(np.diff(r) >= 0)  # monotone
    assert ref.exceedance(np.array([989.0]))[0] == pytest.approx(0.01)
    assert r[2] == pytest.approx(2 / np.log10(1001))  # exceedance 1 % -> 2 decades of rarity


def test_fusion_formula():
    lr = np.array([0.2, 0.9, 0.5])
    iff = np.array([0.8, 0.1, 0.5])
    assert np.allclose(fuse(lr, iff, 1.0), lr)
    assert np.allclose(fuse(lr, iff, 0.0), iff)
    assert np.allclose(fuse(lr, iff, 0.25), 0.25 * lr + 0.75 * iff)


def test_threshold_decisions_and_anomaly_trigger():
    h = np.array([0.1, 0.39, 0.4, 0.69, 0.7, 0.95])
    assert decide(h, 0.4, 0.7).tolist() == [0, 0, 1, 1, 2, 2]
    pct = np.array([0.9995, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert decide(h, 0.4, 0.7, pct, 0.999).tolist() == [1, 0, 1, 1, 2, 2]
    assert decide(h, 0.4, 0.7, pct, None).tolist() == [0, 0, 1, 1, 2, 2]


def test_stored_policy_reproduces_stored_metrics(trained):
    md = trained.metadata
    m = trained.model
    y = trained.splits.test[C.TARGET].to_numpy()
    s = m.score(trained.splits.test[m.features])
    p = md["policy"]
    policy = DecisionPolicy(p["fusion_weight_lr"], p["review_threshold"], p["high_risk_threshold"], p["lr_threshold"], p["if_flag_percentile"], p["if_review_percentile"])
    again = evaluate_scores(y, s, policy)
    assert again["hybrid_high_risk"]["confusion_matrix"] == md["metrics"]["test"]["hybrid_high_risk"]["confusion_matrix"]
    assert again["decisions"] == md["metrics"]["test"]["decisions"]
    assert p["review_threshold"] <= p["high_risk_threshold"]


# ----------------------------------------------------------------------------- metrics
def test_classification_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.1, 0.7, 0.2])
    m = M.classification_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)
    assert m["false_positives"] == 1 and m["false_negatives"] == 1 and m["fraud_detection_rate"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(0.9375)
    assert m["average_precision"] == pytest.approx(average_precision_score(y, p), abs=1e-4)
    assert "brier" in m and "brier" not in M.classification_metrics(y, p, 0.5, probabilistic=False)


def test_threshold_helpers():
    y = np.array([0] * 90 + [1] * 10)
    s = np.r_[np.linspace(0, 0.5, 90), np.linspace(0.4, 1.0, 10)]
    t = M.threshold_for_recall(y, s, 0.8)
    assert np.mean(s[y == 1] >= t["threshold"]) >= 0.8 and t["binding"] == "recall_target"
    capped = M.threshold_for_recall(y, s, 1.0, max_alert_rate=0.05)
    assert capped["binding"] == "max_alert_rate" and np.mean(s >= capped["threshold"]) <= 0.05
    f1_t = M.best_f1_threshold(y, s)
    assert M.classification_metrics(y, s, f1_t)["f1"] >= M.classification_metrics(y, s, 0.5)["f1"]


def test_pr_points_keep_curve_corners():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.002).astype(int)
    s = rng.random(20000) + y
    pts = M.pr_points(y, s)
    assert len(pts) <= 200 and min(p["recall"] for p in pts) == 0.0 and max(p["recall"] for p in pts) == 1.0


def test_paired_bootstrap_ap_detects_clear_difference():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    good = y + rng.normal(scale=0.3, size=3000)
    bad = y + rng.normal(scale=3.0, size=3000)
    res = M.paired_bootstrap_ap(y, good, bad, n_boot=200)
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


def test_drift_reference_flags_shifted_amounts(trained):
    from ml.drift import drift_report

    ref = trained.metadata["reference_profile"]
    test = trained.splits.test[trained.model.features]
    same = drift_report(ref, test)
    assert "Amount" not in same["flagged_features"]
    shifted = test.copy()
    shifted["Amount"] = shifted["Amount"] * 100
    assert "Amount" in drift_report(ref, shifted)["flagged_features"]


# ----------------------------------------------------------------------------- training + packaging
def test_training_outputs_all_three_evaluations(trained):
    t = trained.metadata["metrics"]["test"]
    for k in ("logistic_regression", "isolation_forest", "hybrid_high_risk", "hybrid_review_or_high"):
        assert {"precision", "recall", "f1", "average_precision", "roc_auc", "confusion_matrix", "false_positives", "false_negatives"} <= set(t[k])
    assert sum(d["transactions"] for d in t["decisions"]) == len(trained.splits.test)
    assert 0.5 < t["logistic_regression"]["roc_auc"] <= 1.0


def test_save_load_and_integrity(trained, tmp_path):
    info = save_artifact(tmp_path, "v1", trained.model, trained.metadata)
    loaded = load_artifact(info["model_path"], info["sha256"])
    X = trained.splits.test[trained.model.features].head(30)
    assert np.allclose(loaded.score(X)["hybrid_score"], trained.model.score(X)["hybrid_score"])
    with open(info["model_path"], "ab") as fh:  # tamper with the artifact
        fh.write(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        load_artifact(info["model_path"], info["sha256"])


def test_training_rejects_invalid_dataset(sample_df):
    with pytest.raises(DataValidationError):
        train_model(sample_df.drop(columns=["V14"]), TrainConfig.from_dict(FAST_CONFIG))


def test_train_config_validation():
    for bad in ({"lr_imbalance": "smote"}, {"unknown_key": 1}, {"fusion_weight": 1.5}, {"lr_c_grid": [-1]}, {"if_review_percentile": 1.2}):
        with pytest.raises(ValueError):
            TrainConfig.from_dict(bad)


def test_artifact_write_is_atomic(tmp_path):
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("cannot pickle")

    with pytest.raises(RuntimeError):
        save_artifact(tmp_path, "v9", Unpicklable(), {})
    assert not (tmp_path / "v9").exists()
    assert not any(p.name.startswith(".v9.tmp") for p in tmp_path.iterdir())


def test_mlflow_logging_roundtrip(tmp_path):
    from ml.tracking import EXPERIMENT_NAME, log_training_run

    assert EXPERIMENT_NAME == "fraud-detection"
    res = log_training_run(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=tmp_path / "mlruns",
        run_name="test-run",
        params={"lr_c_grid": [0.1, 1.0], "policy.fusion_weight_lr": 1.0},
        metrics={"test": {"hybrid_high_risk": {"average_precision": 0.7, "confusion_matrix": {"tp": 3}}}},
        tags={"decision": "promoted"},
    )
    assert "run_id" in res, res
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    run = mlflow.get_run(res["run_id"])
    assert run.data.metrics["test.hybrid_high_risk.average_precision"] == 0.7 and run.data.params["policy.fusion_weight_lr"] == "1.0"
