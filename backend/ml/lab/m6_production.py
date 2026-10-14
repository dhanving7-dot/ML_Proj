"""M6: training-serving skew, point-in-time correctness, packaging (pickle/joblib/ONNX) and drift sandbox."""
from __future__ import annotations

import copy
import io
import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import contract as C
from ..data import clean_for_training
from ..drift import drift_report
from ..estimators import make_classifier, optional_available
from ..features import SECONDS_PER_DAY, build_pipeline, coerce_features, hour_of_cycle
from ..fraud_model import FraudModel
from .common import LabData, r

RecordValidator = Callable[[dict[str, Any]], dict[str, Any]]


def _json_roundtrip(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialise rows the way an HTTP client would send them."""
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return json.loads(json.dumps(records, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def _skew_sample(data: LabData, n_legit: int = 3000) -> pd.DataFrame:
    return data.enriched("validation", n_legit, seed=3)


def training_serving_skew(data: LabData, model: FraudModel, metadata: dict[str, Any], validate_record: RecordValidator) -> dict[str, Any]:
    feats = list(metadata["features"])
    va = _skew_sample(data)
    y = va[C.TARGET].to_numpy()

    # Golden end-to-end test: raw CSV text -> training path vs JSON -> API validation -> serving path.
    raw_csv = va[[*feats, C.TARGET]].to_csv(index=False)
    train_path, _ = clean_for_training(pd.read_csv(io.StringIO(raw_csv)), drop_duplicates=False)
    serve_records = [validate_record(rec) for rec in _json_roundtrip(va[feats])]
    serve_path = coerce_features(serve_records, feats)
    v_train, v_serve = model.transform(train_path), model.transform(serve_path)
    s_train, s_serve = model.score_transformed(v_train), model.score_transformed(v_serve)
    golden = {
        "rows": int(len(va)),
        "max_feature_vector_diff": float(np.max(np.abs(v_train - v_serve))),
        "max_hybrid_score_diff": float(np.max(np.abs(s_train["hybrid_score"] - s_serve["hybrid_score"]))),
        "decision_mismatches": int(np.sum(s_train["decision"] != s_serve["decision"])),
        "passed": bool(np.max(np.abs(v_train - v_serve)) < 1e-9),
    }

    base = model.score(va[feats])

    def contract_catches(df: pd.DataFrame) -> bool:
        for rec in _json_roundtrip(df.head(300)):
            try:
                validate_record(rec)
            except Exception:
                return True
        return False

    def summarise(sid: str, title: str, cause: str, X_bad: pd.DataFrame | None, bad: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, Any]:
        m = np.ones(len(va), dtype=bool) if mask is None else mask
        flips = base["decision"][m] != bad["decision"][m]
        both = len(np.unique(y[m])) == 2
        oor = model.out_of_range(X_bad[m]) if X_bad is not None else [[] for _ in range(int(m.sum()))]
        return {
            "id": sid,
            "title": title,
            "cause": cause,
            "rows_in_batch": int(m.sum()),
            "frauds_in_batch": int(y[m].sum()),
            "mean_abs_hybrid_change": r(np.mean(np.abs(base["hybrid_score"][m] - bad["hybrid_score"][m]))),
            "decision_change_rate": r(flips.mean()),
            "fraud_flagged_correct": int(np.sum((base["decision"][m] >= 1) & (y[m] == 1))),
            "fraud_flagged_skewed": int(np.sum((bad["decision"][m] >= 1) & (y[m] == 1))),
            "average_precision_correct": r(average_precision_score(y[m], base["hybrid_score"][m])) if both else None,
            "average_precision_skewed": r(average_precision_score(y[m], bad["hybrid_score"][m])) if both else None,
            "caught_by_contract_validation": contract_catches(X_bad[m]) if X_bad is not None else False,
            "rows_with_out_of_range_warning": r(np.mean([len(o) > 0 for o in oor])) if len(oor) else 0.0,
        }

    scenarios = []
    # 1. Scaler re-fitted in production on a serving batch (the highest-amount transactions).
    mask = (va["Amount"] >= va["Amount"].quantile(0.8)).to_numpy()
    bad_model = copy.deepcopy(model)
    pre = bad_model.preprocessor
    engineered = pre.named_steps["engineer"].transform(va[feats][mask])
    pre.steps[-1] = ("scale", StandardScaler().fit(engineered))
    bad = {k: np.array(v) for k, v in base.items()}
    sb = bad_model.score(va[feats][mask])
    for k in bad:
        bad[k][mask] = sb[k]
    scenarios.append(summarise("scaler_refit", "StandardScaler re-fitted on a serving batch", "Production re-fits the scaler on a batch of large transactions instead of loading the training scaler.", None, bad, mask))

    # 2. Amount sent in cents.
    X_bad = va[feats].copy()
    X_bad["Amount"] = X_bad["Amount"] * 100
    scenarios.append(summarise("amount_cents", "Amount sent in cents", "An upstream system sends Amount x 100. Still non-negative, so the contract accepts it.", X_bad, model.score(X_bad)))

    # 3. Time sent in hours instead of seconds.
    X_bad = va[feats].copy()
    X_bad["Time"] = X_bad["Time"] / 3600.0
    scenarios.append(summarise("time_hours", "Time sent in hours", "Time arrives as hours since the first transaction instead of seconds, so the hour-of-cycle encoding is wrong.", X_bad, model.score(X_bad)))

    # 4. Two components swapped upstream.
    X_bad = va[feats].copy()
    X_bad[["V14", "V17"]] = X_bad[["V17", "V14"]].to_numpy()
    scenarios.append(summarise("column_swap", "Two columns swapped", "A refactor maps V14 to V17 and vice versa. Names and types are valid, so nothing is rejected.", X_bad, model.score(X_bad)))

    # 5. The client applies log1p to Amount before sending it.
    X_bad = va[feats].copy()
    X_bad["Amount"] = np.log1p(X_bad["Amount"])
    scenarios.append(summarise("double_log", "Amount log-transformed twice", "The client sends log(1 + Amount); the pipeline applies log(1 + x) again.", X_bad, model.score(X_bad)))
    return {
        "split": data.split_info(0, va, "Uses the active production model on every validation fraud + 3,000 legitimate validation rows."),
        "model_version": metadata.get("version"),
        "golden_test": golden,
        "scenarios": scenarios,
        "prevention": [
            "One shared feature package (ml.contract / ml.features) is imported by training and serving.",
            "The fitted preprocessing (scaler statistics) is persisted inside the artifact and never re-fitted in production.",
            "The API rejects unknown fields, missing fields, non-finite numbers and negative Time / Amount (data contract).",
            "Values outside the training range are flagged on every prediction, which surfaces unit mistakes such as cents.",
            "Every prediction stores the exact feature vector the model saw, and golden examples are re-checked whenever a model is loaded.",
        ],
    }


def point_in_time(data: LabData, model: str = "logistic_regression", time_shift_hours: int = 6) -> dict[str, Any]:
    """Random vs time-ordered evaluation, and a Time origin that differs between training and serving."""
    if model not in ("logistic_regression", "hist_gradient_boosting"):
        raise ValueError("model must be logistic_regression or hist_gradient_boosting")
    feats = list(C.FEATURE_NAMES)
    dev = data.development.sort_values("Time").reset_index(drop=True)
    params = {"class_weight": "balanced"} if model == "logistic_regression" else {}

    def fit_score(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, Any]:
        pipe = build_pipeline(make_classifier(model, seed=0, **params)).fit(tr[feats], tr[C.TARGET])
        p = pipe.predict_proba(te[feats])[:, 1]
        return {
            "train_rows": int(len(tr)),
            "train_frauds": int(tr[C.TARGET].sum()),
            "eval_rows": int(len(te)),
            "eval_frauds": int(te[C.TARGET].sum()),
            "average_precision": r(average_precision_score(te[C.TARGET], p)),
            "roc_auc": r(roc_auc_score(te[C.TARGET], p)),
        }

    cut = int(len(dev) * 0.8)
    time_ordered = fit_score(dev.iloc[:cut], dev.iloc[cut:])
    time_ordered["train_time_range_h"] = [r(dev["Time"].iloc[0] / 3600, 2), r(dev["Time"].iloc[cut - 1] / 3600, 2)]
    time_ordered["eval_time_range_h"] = [r(dev["Time"].iloc[cut] / 3600, 2), r(dev["Time"].iloc[-1] / 3600, 2)]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(dev))
    random_split = fit_score(dev.iloc[perm[:cut]], dev.iloc[perm[cut:]])

    # Same name, different meaning: the serving system computes Time from a different origin.
    tr, va = data.train, data.validation
    pipe = build_pipeline(make_classifier(model, seed=0, **params)).fit(tr[feats], tr[C.TARGET])
    served = va[feats].copy()
    served["Time"] = served["Time"] + time_shift_hours * 3600
    p_ok = pipe.predict_proba(va[feats])[:, 1]
    p_shift = pipe.predict_proba(served)[:, 1]
    hour_ok = hour_of_cycle(va["Time"])
    hour_shift = hour_of_cycle(served["Time"])
    return {
        "split": data.split_info(tr, va, "Uses the training and validation rows only (the test split stays untouched)."),
        "model": model,
        "random_vs_time": {
            "random_80_20": random_split,
            "time_ordered_80_20": time_ordered,
            "difference_time_minus_random": r(time_ordered["average_precision"] - random_split["average_precision"]),
        },
        "explanation": (
            "A random split lets the model train on transactions that happened after the ones it is evaluated on. "
            "A time-ordered split (train on the earlier 80 %, evaluate on the later 20 %) mimics production, where only the past is available. "
            "The production model uses a random stratified split as the brief requires; this experiment measures how much that choice matters here. "
            f"Each evaluation set holds only about {time_ordered['eval_frauds']} fraud rows, so a difference of a few hundredths in either direction is within sampling noise, "
            "and two days of data cannot show slow drift."
        ),
        "semantic_mismatch": {
            "feature": "Time",
            "training_definition": "seconds since the first transaction in the dataset",
            "serving_definition": f"the same clock, but with an origin {time_shift_hours} h earlier",
            "hour_of_cycle_shift_h": r(float(np.mean((hour_shift - hour_ok) % 24)), 2),
            "validation_ap_training_definition": r(average_precision_score(va[C.TARGET], p_ok)),
            "validation_ap_serving_definition": r(average_precision_score(va[C.TARGET], p_shift)),
            "mean_abs_probability_change": r(np.mean(np.abs(p_ok - p_shift)), 6),
            "explanation": "Same name and type, different meaning: the hour-of-cycle features move by the offset, which changes predictions without any validation error.",
        },
        "seconds_per_day": SECONDS_PER_DAY,
    }


def _portable_pipeline(model: str) -> Pipeline:
    est = make_classifier(model, seed=0, **({"n_estimators": 100} if model == "random_forest" else {}), **({"class_weight": "balanced"} if model == "logistic_regression" else {}))
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def onnx_export(data: LabData, model: str = "logistic_regression", eval_rows: int = 5000) -> dict[str, Any]:
    if model not in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        raise ValueError("model must be logistic_regression, random_forest or hist_gradient_boosting")
    if not (optional_available("skl2onnx") and optional_available("onnxruntime")):
        return {"available": False, "error": "skl2onnx / onnxruntime are not installed"}
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    feats = list(C.FEATURE_NAMES)
    tr = data.train if model == "logistic_regression" else data.enriched("train", 20000)
    va = data.sample("validation", eval_rows)
    pipe = _portable_pipeline(model).fit(tr[feats], tr[C.TARGET])
    t = time.perf_counter()
    try:
        onx = to_onnx(pipe, initial_types=[("X", FloatTensorType([None, len(feats)]))], options={id(pipe.named_steps["model"]): {"zipmap": False}}, target_opset=17)
    except Exception as exc:
        return {"available": True, "converted": False, "model": model, "error": f"{type(exc).__name__}: {exc}"}
    convert_ms = (time.perf_counter() - t) * 1000
    blob = onx.SerializeToString()
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    X = va[feats].to_numpy(np.float32)
    t = time.perf_counter()
    outputs = sess.run(None, {"X": X})
    onnx_ms = (time.perf_counter() - t) * 1000
    p_onnx = np.asarray(outputs[1])[:, 1]
    t = time.perf_counter()
    p_sk = pipe.predict_proba(va[feats])[:, 1]
    sk_ms = (time.perf_counter() - t) * 1000
    buf = io.BytesIO()
    joblib.dump(pipe, buf, compress=3)
    return {
        "available": True,
        "converted": True,
        "split": data.split_info(tr, va, "Portable variant fitted on training data; parity measured on validation rows."),
        "model": model,
        "variant_note": "The ONNX variant uses the 30 raw features with a StandardScaler: the custom hour-of-cycle and log-amount step has no ONNX converter, so parity is measured between this variant's scikit-learn and ONNX versions.",
        "opset": 17,
        "inputs": len(feats),
        "graph_nodes": len(onx.graph.node),
        "onnx_bytes": len(blob),
        "joblib_bytes": buf.getbuffer().nbytes,
        "convert_ms": r(convert_ms, 1),
        "parity": {
            "rows": int(len(X)),
            "max_abs_probability_diff": float(np.max(np.abs(p_onnx - p_sk))),
            "mean_abs_probability_diff": float(np.mean(np.abs(p_onnx - p_sk))),
            "label_agreement": r(np.mean((p_onnx >= 0.5) == (p_sk >= 0.5))),
            "note": (
                "ONNX Runtime computes in float32, so differences around 1e-6 are expected."
                if model == "logistic_regression"
                else "Tree ensembles compare inputs with split thresholds in float32 inside ONNX Runtime. Rows whose values sit very close to a threshold can follow a different branch, which explains rare but larger probability differences."
            ),
        },
        "latency_ms": {"sklearn_batch": r(sk_ms, 2), "onnxruntime_batch": r(onnx_ms, 2)},
    }


def packaging(model: Any, metadata: dict[str, Any], artifact_path: str | None) -> dict[str, Any]:
    results = {}
    t = time.perf_counter()
    pkl = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    results["pickle"] = {"bytes": len(pkl), "dump_ms": r((time.perf_counter() - t) * 1000, 2)}
    t = time.perf_counter()
    pickle.loads(pkl)
    results["pickle"]["load_ms"] = r((time.perf_counter() - t) * 1000, 2)
    with tempfile.TemporaryDirectory() as tmp:
        for level in (0, 3, 9):
            path = Path(tmp) / f"m{level}.joblib"
            t = time.perf_counter()
            joblib.dump(model, path, compress=level)
            dump_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            joblib.load(path)
            results[f"joblib_compress_{level}"] = {"bytes": path.stat().st_size, "dump_ms": r(dump_ms, 2), "load_ms": r((time.perf_counter() - t) * 1000, 2)}
    return {
        "model_version": metadata.get("version"),
        "artifact": {
            "path_name": Path(artifact_path).name if artifact_path else None,
            "sha256": metadata.get("artifact_sha256"),
            "library_versions": metadata.get("library_versions"),
            "contents": [
                "model.joblib: FraudModel = preprocessing + Logistic Regression + Isolation Forest + both rarity references + decision policy",
                "metadata.json: schema, config, policy, metrics, curves, reference profile, golden examples",
            ],
        },
        "serialization": results,
        "security": "pickle and joblib can execute arbitrary code when loading. The API only loads artifacts whose SHA-256 matches the registry record written at training time.",
        "grpc": "Conceptual: gRPC would expose the same predict contract as a Protocol Buffers service for low-latency service-to-service calls. This project keeps FastAPI/JSON as the single serving interface.",
    }


SEGMENTS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "random": ("Random 30 % of the validation split (no shift expected)", lambda d: pd.Series(np.random.default_rng(0).random(len(d)) < 0.3, index=d.index)),
    "high_amount": ("Only the top 5 % of amounts", lambda d: d["Amount"] >= d["Amount"].quantile(0.95)),
    "night": ("Only hours 0-5 of the cycle", lambda d: d["hour"] < 6),
    "second_day": ("Only the second day (Time >= 24 h)", lambda d: d["Time"] >= SECONDS_PER_DAY),
    "fraud_only": ("Only fraud rows (a shift in the label mix)", lambda d: d[C.TARGET] == 1),
}


def drift_sandbox(data: LabData, model: FraudModel, metadata: dict[str, Any], segment: str = "high_amount") -> dict[str, Any]:
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {list(SEGMENTS)}")
    feats = list(metadata["features"])
    va = data.validation
    title, selector = SEGMENTS[segment]
    seg = va[selector(va).to_numpy()]
    s_all = model.score(va[feats])
    s_seg = model.score(seg[feats])
    report = drift_report(
        metadata["reference_profile"],
        seg[feats],
        {"lr_probability": s_seg["lr_probability"], "anomaly_percentile": s_seg["anomaly_percentile"], "hybrid_score": s_seg["hybrid_score"]},
    )

    def perf(y: np.ndarray, s: dict[str, np.ndarray]) -> dict[str, Any]:
        flagged = s["decision"] >= 1
        return {
            "rows": int(len(y)),
            "fraud_rows": int(y.sum()),
            "fraud_rate": r(y.mean(), 5),
            "flagged_rate": r(flagged.mean(), 5),
            "recall_review_or_high": r(np.sum(flagged & (y == 1)) / y.sum()) if y.sum() else None,
            "average_precision": r(average_precision_score(y, s["hybrid_score"])) if len(np.unique(y)) == 2 else None,
        }

    ranked = sorted(
        ({"feature": k, "psi": v["psi"], "kl_divergence": v["kl_divergence"], "status": v["status"]} for k, v in report["features"].items()),
        key=lambda d: -(d["psi"] or 0),
    )
    return {
        "split": data.split_info(0, seg, "Segments are filtered validation rows; feature reference = the model's training split, score reference = its validation split."),
        "segment": segment,
        "title": title,
        "segments": {k: v[0] for k, v in SEGMENTS.items()},
        "feature_drift": ranked,
        "score_drift": {k: {m: v[m] for m in ("psi", "kl_divergence", "status")} for k, v in report["scores"].items()},
        "detail": {k: report["features"][k] for k in [d["feature"] for d in ranked[:4]]},
        "performance": {"overall_validation": perf(va[C.TARGET].to_numpy(), s_all), "segment": perf(seg[C.TARGET].to_numpy(), s_seg)},
        "thresholds": report["thresholds"],
        "explanation": "Input drift (PSI on features) and score drift do not by themselves mean detection got worse. Performance can only be measured once confirmed outcomes are known.",
    }
