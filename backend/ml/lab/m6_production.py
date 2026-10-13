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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .. import contract as C
from ..data import clean_for_training
from ..drift import drift_report
from ..estimators import make_classifier, optional_available
from ..features import build_pipeline, coerce_features
from ..training import base_pipeline
from .common import LabData, r

RecordValidator = Callable[[dict[str, Any]], dict[str, Any]]


def _json_roundtrip(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialise rows the way an HTTP client would send them (NaN -> null)."""
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return json.loads(json.dumps(records, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def training_serving_skew(
    data: LabData, model: Any, metadata: dict[str, Any], validate_record: RecordValidator, sample_rows: int = 1500
) -> dict[str, Any]:
    feats = list(metadata["features"])
    pipe = base_pipeline(model)
    threshold = float(metadata["threshold"])
    va = data.sample("validation", sample_rows)
    y = va[C.TARGET].to_numpy()

    # Golden end-to-end test: raw CSV text -> training path vs JSON -> API validation -> serving path.
    raw_csv = va[[C.ID_COLUMN, *feats, C.TARGET]].to_csv(index=False)
    train_path = clean_for_training(pd.read_csv(io.StringIO(raw_csv)))[feats]
    serve_records = [validate_record(rec) for rec in _json_roundtrip(va[feats])]
    serve_path = coerce_features(serve_records, feats)
    v_train = np.asarray(pipe[:-1].transform(train_path), float)
    v_serve = np.asarray(pipe[:-1].transform(serve_path), float)
    p_train = model.predict_proba(train_path)[:, 1]
    p_serve = model.predict_proba(serve_path)[:, 1]
    golden = {
        "rows": int(len(va)),
        "max_feature_vector_diff": float(np.max(np.abs(v_train - v_serve))),
        "max_probability_diff": float(np.max(np.abs(p_train - p_serve))),
        "prediction_mismatches": int(np.sum((p_train >= threshold) != (p_serve >= threshold))),
        "passed": bool(np.max(np.abs(v_train - v_serve)) < 1e-9),
    }

    base_p = model.predict_proba(va[feats])[:, 1]
    base_vec = np.asarray(pipe[:-1].transform(va[feats]), float)

    def contract_catches(df: pd.DataFrame) -> bool:
        for rec in _json_roundtrip(df.head(200)):
            try:
                validate_record(rec)
            except Exception:
                return True
        return False

    def summarise(name: str, title: str, cause: str, X_bad: pd.DataFrame | None, p_bad: np.ndarray, vec_bad: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
        m = np.ones(len(va), dtype=bool) if mask is None else mask
        flips = (base_p[m] >= threshold) != (p_bad[m] >= threshold)
        out = {
            "id": name,
            "title": title,
            "cause": cause,
            "rows_in_batch": int(m.sum()),
            "max_feature_vector_diff": r(np.max(np.abs(base_vec[m] - vec_bad[m])), 6),
            "mean_abs_probability_change": r(np.mean(np.abs(base_p[m] - p_bad[m]))),
            "prediction_flip_rate": r(flips.mean()),
            "roc_auc_correct": r(roc_auc_score(y[m], base_p[m])) if len(np.unique(y[m])) == 2 else None,
            "roc_auc_skewed": r(roc_auc_score(y[m], p_bad[m])) if len(np.unique(y[m])) == 2 else None,
            "caught_by_contract_validation": contract_catches(X_bad[m] if X_bad is not None else va[feats][m]) if X_bad is not None else False,
        }
        return out

    scenarios = []
    # 1. Scaler re-fitted in production on a serving batch (a Tier3-only batch).
    mask = (va["CollegeTier"] == "Tier3").to_numpy()
    bad_pipe = copy.deepcopy(pipe)
    pre = bad_pipe.named_steps["preprocess"]
    num_cols = pre.transformers_[0][2]
    engineered = bad_pipe.named_steps["engineer"].transform(va[feats][mask]) if "engineer" in bad_pipe.named_steps else va[feats][mask]
    imputed = pre.named_transformers_["num"].named_steps["impute"].transform(engineered[num_cols])
    pre.named_transformers_["num"].named_steps["scale"].fit(imputed)
    p_bad = np.array(base_p)
    p_bad[mask] = bad_pipe.predict_proba(va[feats][mask])[:, 1]
    vec_bad = np.array(base_vec)
    vec_bad[mask] = bad_pipe[:-1].transform(va[feats][mask])
    scenarios.append(summarise("scaler_refit", "StandardScaler re-fitted on a serving batch", "Production re-fits the scaler on a Tier3-only batch instead of loading the training scaler.", None, p_bad, vec_bad, mask))

    # 2. Missing values filled with 0 instead of the learned training median.
    X_bad = va[feats].copy()
    nullable = [f.name for f in C.NUMERIC_FEATURES if f.nullable and f.name in feats]
    mask = X_bad[nullable].isna().any(axis=1).to_numpy()
    X_bad[nullable] = X_bad[nullable].fillna(0)
    scenarios.append(
        summarise("missing_zero", "Missing values filled with 0", "Serving code fills nulls with 0 before the model instead of letting the pipeline impute the training median.", X_bad, model.predict_proba(X_bad)[:, 1], np.asarray(pipe[:-1].transform(X_bad), float), mask)
    )

    # 3. Categorical vocabulary mismatch upstream.
    X_bad = va[feats].copy()
    X_bad["City"] = X_bad["City"].replace({"Bangalore": "Bengaluru"})
    X_bad["Stream"] = X_bad["Stream"].replace({"CS": "CSE"})
    mask = ((va["City"] == "Bangalore") | (va["Stream"] == "CS")).to_numpy()
    scenarios.append(
        summarise("category_mismatch", "Category spelling changed upstream", "An upstream system sends 'Bengaluru' and 'CSE' instead of 'Bangalore' and 'CS'.", X_bad, model.predict_proba(X_bad)[:, 1], np.asarray(pipe[:-1].transform(X_bad), float), mask)
    )

    # 4. Same name, different unit: attendance as a fraction.
    X_bad = va[feats].copy()
    X_bad["AttendancePercent"] = X_bad["AttendancePercent"] / 100.0
    scenarios.append(
        summarise("unit_mismatch", "Same name, different unit", "AttendancePercent is sent as a fraction (0.85) instead of a percentage (85). Both are inside the 0-100 contract range.", X_bad, model.predict_proba(X_bad)[:, 1], np.asarray(pipe[:-1].transform(X_bad), float))
    )
    return {
        "split": data.split_info(0, len(va), "Uses the active production model on a validation sample."),
        "model_version": metadata.get("version"),
        "golden_test": golden,
        "scenarios": scenarios,
        "prevention": [
            "One shared feature package (ml.contract / ml.features) is imported by training and serving.",
            "The fitted Pipeline (imputer medians, scaler parameters, encoder vocabularies) is persisted and loaded, never re-fitted in production.",
            "The API rejects unknown fields and out-of-vocabulary categories (data contract).",
            "Every prediction stores the exact feature vector the model saw, and golden examples are re-checked whenever a model is loaded.",
        ],
    }


def point_in_time(data: LabData, model: str = "lightgbm", sample_size: int = 15000) -> dict[str, Any]:
    if model == "lightgbm" and not optional_available("lightgbm"):
        model = "hist_gradient_boosting"
    df = data.sample("train", sample_size)
    y = df[C.TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    sets = []
    for name, feats in C.FEATURE_SETS.items():
        pipe = build_pipeline(make_classifier(model, seed=0), list(feats), engineered=True)
        scores = cross_val_score(pipe, df[list(feats)], y, cv=cv, scoring="roc_auc", n_jobs=1)
        sets.append(
            {
                "feature_set": name,
                "n_features": len(feats),
                "excluded": [f for f in C.FEATURE_NAMES if f not in feats],
                "cv_roc_auc_mean": r(scores.mean()),
                "cv_roc_auc_std": r(scores.std()),
            }
        )

    # "Same name, different meaning": a model trained with the dataset's CGPA_Tier (documented bins
    # 6.5 / 8.0) is served by a team that computes the tier with different grade bands (6.0 / 7.5).
    tr, va = data.sample("train", sample_size), data.validation
    numeric, nominal, ordinal = C.split_feature_kinds(list(C.FEATURE_NAMES))
    feats = [*C.FEATURE_NAMES, "CGPA_Tier"]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("ord", OrdinalEncoder(categories=[["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]], handle_unknown="use_encoded_value", unknown_value=-1), [*ordinal, "CGPA_Tier"]),
        ]
    )
    pipe = Pipeline([("pre", pre), ("model", make_classifier(model, seed=0))]).fit(tr[feats], tr[C.TARGET])
    served = va[feats].copy()
    served["CGPA_Tier"] = np.select([served["CGPA"] < 6.0, served["CGPA"] < 7.5], ["Low", "Mid"], "High")
    p_train_def = pipe.predict_proba(va[feats])[:, 1]
    p_serve_def = pipe.predict_proba(served)[:, 1]
    auc_train_def = roc_auc_score(va[C.TARGET], p_train_def)
    auc_serve_def = roc_auc_score(va[C.TARGET], p_serve_def)
    agreement = float((served["CGPA_Tier"] == va["CGPA_Tier"]).mean())
    delta = auc_train_def - auc_serve_def
    return {
        "split": data.split_info(len(df), len(va), "5-fold CV on a training sample; the semantic test scores on validation."),
        "model": model,
        "feature_sets": sets,
        "explanation": "Placement drives usually start before the final semesters are graded. If predictions are made then, SGPA_Sem7, SGPA_Sem8 and final CGPA do not exist yet; training on them leaks future information and inflates offline scores.",
        "semantic_mismatch": {
            "feature": "CGPA_Tier",
            "training_definition": "dataset tiers (CGPA < 6.5 Low, < 8.0 Mid, else High for most rows)",
            "serving_definition": "CGPA < 6.0 Low, < 7.5 Mid, else High",
            "tier_agreement_between_definitions": r(agreement),
            "validation_roc_auc_training_definition": r(auc_train_def),
            "validation_roc_auc_serving_definition": r(auc_serve_def),
            "mean_abs_probability_change": r(np.mean(np.abs(p_train_def - p_serve_def))),
            "explanation": (
                "Same name and type, different meaning. "
                + (
                    "The AUC change is small here because CGPA and the SGPA columns carry the same information, so the model leans on them rather than on the tier."
                    if abs(delta) < 0.005
                    else "The change in AUC shows how much the model relied on the tier's original meaning."
                )
            ),
        },
    }


def _portable_pipeline(model: str, feats: list[str]) -> Pipeline:
    numeric, nominal, ordinal = C.split_feature_kinds(feats)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [*nominal, *ordinal]),
        ]
    )
    est = make_classifier(model, seed=0, **({"n_estimators": 100} if model == "random_forest" else {}))
    return Pipeline([("preprocess", pre), ("model", est)])


def onnx_export(data: LabData, model: str = "logistic_regression", train_rows: int = 15000, eval_rows: int = 2000) -> dict[str, Any]:
    if model not in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        raise ValueError("model must be logistic_regression, random_forest or hist_gradient_boosting")
    if not (optional_available("skl2onnx") and optional_available("onnxruntime")):
        return {"available": False, "error": "skl2onnx / onnxruntime are not installed"}
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    feats = list(C.FEATURE_NAMES)
    numeric, nominal, ordinal = C.split_feature_kinds(feats)
    tr, va = data.sample("train", train_rows), data.sample("validation", eval_rows)
    pipe = _portable_pipeline(model, feats).fit(tr[feats], tr[C.TARGET])
    initial_types = [(c, FloatTensorType([None, 1])) for c in numeric] + [(c, StringTensorType([None, 1])) for c in [*nominal, *ordinal]]
    t = time.perf_counter()
    try:
        onx = to_onnx(pipe, initial_types=initial_types, options={id(pipe.named_steps["model"]): {"zipmap": False}}, target_opset=17)
    except Exception as exc:
        return {"available": True, "converted": False, "model": model, "error": f"{type(exc).__name__}: {exc}"}
    convert_ms = (time.perf_counter() - t) * 1000
    blob = onx.SerializeToString()
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    X = va[feats]
    feeds = {c: X[c].to_numpy(np.float32).reshape(-1, 1) for c in numeric}
    feeds.update({c: X[c].astype(str).to_numpy().reshape(-1, 1) for c in [*nominal, *ordinal]})
    t = time.perf_counter()
    outputs = sess.run(None, feeds)
    onnx_ms = (time.perf_counter() - t) * 1000
    p_onnx = np.asarray(outputs[1])[:, 1]
    t = time.perf_counter()
    p_sk = pipe.predict_proba(X)[:, 1]
    sk_ms = (time.perf_counter() - t) * 1000
    buf = io.BytesIO()
    joblib.dump(pipe, buf, compress=3)
    return {
        "available": True,
        "converted": True,
        "split": data.split_info(len(tr), len(va), "Portable variant fitted on training data; parity measured on validation rows."),
        "model": model,
        "variant_note": "The ONNX variant omits the custom SGPA feature-engineering step and missing-value indicator columns, which have no ONNX converters; parity is measured between this variant's scikit-learn and ONNX versions.",
        "opset": 17,
        "inputs": len(initial_types),
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
            "contents": ["model.joblib (feature engineering + preprocessing + estimator)", "metadata.json (schema, config, metrics, reference profile, golden examples)"],
        },
        "serialization": results,
        "security": "pickle and joblib can execute arbitrary code when loading. The API only loads artifacts whose SHA-256 matches the registry record written at training time.",
        "grpc": "Conceptual: gRPC would expose the same predict contract as a Protocol Buffers service for low-latency service-to-service calls. This project keeps FastAPI/JSON as the single serving interface.",
    }


SEGMENTS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "random": ("Random validation sample (no shift expected)", lambda d: pd.Series(np.random.default_rng(0).random(len(d)) < 0.3, index=d.index)),
    "tier3": ("Only Tier3 colleges", lambda d: d["CollegeTier"] == "Tier3"),
    "backlogs": ("Only students with a history of backlogs", lambda d: d["HistoryOfBacklogs"] == "Yes"),
    "high_cgpa": ("Only students with CGPA >= 8.5", lambda d: d["CGPA"] >= 8.5),
    "low_attendance": ("Only students with attendance < 65%", lambda d: d["AttendancePercent"] < 65),
}


def drift_sandbox(data: LabData, model: Any, metadata: dict[str, Any], segment: str = "tier3") -> dict[str, Any]:
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {list(SEGMENTS)}")
    feats = list(metadata["features"])
    threshold = float(metadata["threshold"])
    va = data.validation
    title, selector = SEGMENTS[segment]
    seg = va[selector(va).to_numpy()]
    p_all = model.predict_proba(va[feats])[:, 1]
    p_seg = model.predict_proba(seg[feats])[:, 1]
    report = drift_report(metadata["reference_profile"], seg[feats], p_seg)

    def perf(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        return {
            "rows": int(len(y)),
            "placed_rate": r(y.mean()),
            "accuracy": r(np.mean((p >= threshold) == y)),
            "roc_auc": r(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }

    ranked = sorted(
        ({"feature": k, "psi": v["psi"], "kl_divergence": v["kl_divergence"], "status": v["status"]} for k, v in report["features"].items()),
        key=lambda d: -(d["psi"] or 0),
    )
    return {
        "split": data.split_info(0, len(seg), "Segments are filtered validation rows; reference = the model's training split."),
        "segment": segment,
        "title": title,
        "segments": {k: v[0] for k, v in SEGMENTS.items()},
        "feature_drift": ranked,
        "prediction_drift": {k: report["prediction"][k] for k in ("psi", "kl_divergence", "status")},
        "detail": {k: report["features"][k] for k in [d["feature"] for d in ranked[:4]]},
        "performance": {"overall_validation": perf(va[C.TARGET].to_numpy(), p_all), "segment": perf(seg[C.TARGET].to_numpy(), p_seg)},
        "thresholds": report["thresholds"],
        "explanation": "Input drift (PSI on features) and prediction drift do not by themselves mean the model got worse. Performance can only be measured once ground-truth outcomes are known.",
    }
