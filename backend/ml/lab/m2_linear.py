"""M2: linear regression, regularisation, logistic regression and preprocessing experiments.

Regression needs a continuous target. The only continuous column besides ``Time`` is ``Amount``,
so the regression experiments predict ``log(1 + Amount)`` from ``V1``-``V28`` (the components are
partly informative about the amount, e.g. ``V2``). Classification predicts ``Class``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, lasso_path
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from .. import contract as C
from .. import metrics as M
from ..features import build_preprocessor, transformed_feature_names
from .common import PCA, LabData, r, scaled_matrix, scatter_sample

REG_FEATURES = PCA
REG_TARGET = "amount_log"


def _amount_design(data: LabData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """log(1 + Amount) from V1-V28; standardisation fitted on the training split only."""
    tr, va = data.train, data.validation
    X_tr, X_va = scaled_matrix(tr, REG_FEATURES, va)
    info = data.split_info(tr, va, "Target = log(1 + Amount); inputs = V1-V28 standardised with training statistics.")
    return X_tr, tr[REG_TARGET].to_numpy(float), X_va, va[REG_TARGET].to_numpy(float), REG_FEATURES, info


def linear_regression(data: LabData, learning_rate: float = 0.05, iterations: int = 400) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _amount_design(data)
    n = len(y_tr)
    Xb = np.hstack([np.ones((n, 1)), X_tr])
    Xb_va = np.hstack([np.ones((len(y_va), 1)), X_va])

    t = time.perf_counter()
    theta_ne = np.linalg.solve(Xb.T @ Xb, Xb.T @ y_tr)  # normal equation: (X^T X) theta = X^T y
    ne_ms = (time.perf_counter() - t) * 1000

    sk = LinearRegression().fit(X_tr, y_tr)
    theta_sk = np.concatenate([[sk.intercept_], sk.coef_])

    theta = np.zeros(Xb.shape[1])
    history = []
    t = time.perf_counter()
    for i in range(iterations):
        resid = Xb @ theta - y_tr
        theta -= learning_rate * (2.0 / n) * (Xb.T @ resid)
        if not np.all(np.isfinite(theta)):
            history.append({"iteration": i + 1, "train_mse": None})
            break
        if i < 20 or (i + 1) % max(1, iterations // 80) == 0 or i == iterations - 1:
            history.append({"iteration": i + 1, "train_mse": r(np.mean((Xb @ theta - y_tr) ** 2))})
    gd_ms = (time.perf_counter() - t) * 1000
    diverged = not np.all(np.isfinite(theta))

    ne_mse = float(np.mean((Xb @ theta_ne - y_tr) ** 2))
    eig = np.linalg.eigvalsh(X_tr.T @ X_tr / n)
    idx = scatter_sample(len(y_va), 400)
    pred_va = Xb_va @ theta_ne
    cond = float(eig.max() / max(eig.min(), 1e-12))
    return {
        "split": info,
        "target": "log(1 + Amount)",
        "coefficients": [
            {"feature": f, "normal_equation": r(theta_ne[i + 1]), "sklearn": r(theta_sk[i + 1]), "gradient_descent": None if diverged else r(theta[i + 1])}
            for i, f in enumerate(names)
        ],
        "intercept": {"normal_equation": r(theta_ne[0]), "sklearn": r(theta_sk[0]), "gradient_descent": None if diverged else r(theta[0])},
        "agreement": {
            "max_abs_diff_normal_eq_vs_sklearn": r(np.max(np.abs(theta_ne - theta_sk)), 10),
            "max_abs_diff_gd_vs_normal_eq": None if diverged else r(np.max(np.abs(theta - theta_ne)), 6),
            "train_mse_normal_eq": r(ne_mse),
            "train_mse_gd": None if diverged else r(np.mean((Xb @ theta - y_tr) ** 2)),
        },
        "gradient_descent": {"learning_rate": learning_rate, "iterations": iterations, "diverged": diverged, "loss_history": history, "duration_ms": r(gd_ms, 2)},
        "normal_equation_ms": r(ne_ms, 3),
        "conditioning": {
            "condition_number": r(cond, 2),
            "smallest_eigenvalue": r(eig.min(), 6),
            "note": (
                "V1-V28 are uncorrelated PCA components, so after standardisation X^T X / n is almost the identity "
                f"(condition number {cond:.2f}) and gradient descent converges in few iterations."
            ),
        },
        "validation_metrics": M.regression_metrics(y_va, pred_va),
        "train_metrics": M.regression_metrics(y_tr, Xb @ theta_ne),
        "predictions_sample": [{"actual": r(y_va[i], 3), "predicted": r(pred_va[i], 3)} for i in idx],
    }


def regularization(data: LabData, ridge_alpha: float = 1000.0, lasso_alpha: float = 0.01, enet_alpha: float = 0.01, l1_ratio: float = 0.5) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _amount_design(data)
    models = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=ridge_alpha),
        "lasso": Lasso(alpha=lasso_alpha, max_iter=20000),
        "elastic_net": ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio, max_iter=20000),
    }
    summary, coefs = [], {}
    for key, m in models.items():
        m.fit(X_tr, y_tr)
        coefs[key] = m.coef_
        summary.append(
            {
                "model": key,
                "zero_coefficients": int(np.sum(np.abs(m.coef_) < 1e-8)),
                "l2_norm": r(np.linalg.norm(m.coef_)),
                "l1_norm": r(np.sum(np.abs(m.coef_))),
                "train": M.regression_metrics(y_tr, m.predict(X_tr)),
                "validation": M.regression_metrics(y_va, m.predict(X_va)),
            }
        )
    # Regularisation paths. Features are standardised on train (mean 0), so centring y lets the
    # intercept be recovered as mean(y). Ridge uses the closed form via SVD; Lasso uses the
    # warm-started coordinate-descent path.
    ridge_alphas = np.logspace(0, 7, 22)
    lasso_alphas = np.logspace(0, -4, 22)  # lasso_path expects decreasing alphas
    y_mean = y_tr.mean()
    U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
    Uty = U.T @ (y_tr - y_mean)
    _, lasso_coefs, _ = lasso_path(X_tr, y_tr - y_mean, alphas=lasso_alphas, max_iter=20000)
    path = []
    for i, ra in enumerate(ridge_alphas):
        ridge_coef = Vt.T @ (s / (s**2 + ra) * Uty)
        j = len(lasso_alphas) - 1 - i  # pair weakest ridge penalty with weakest lasso penalty
        lasso_coef = lasso_coefs[:, j]
        path.append(
            {
                "ridge_alpha": r(ra, 5),
                "lasso_alpha": r(lasso_alphas[j], 6),
                "ridge_val_rmse": M.regression_metrics(y_va, X_va @ ridge_coef + y_mean)["rmse"],
                "lasso_val_rmse": M.regression_metrics(y_va, X_va @ lasso_coef + y_mean)["rmse"],
                "lasso_nonzero": int(np.sum(np.abs(lasso_coef) > 1e-8)),
                "ridge": {f: r(c) for f, c in zip(names, ridge_coef)},
                "lasso": {f: r(c) for f, c in zip(names, lasso_coef)},
            }
        )
    return {
        "split": info,
        "params": {"ridge_alpha": ridge_alpha, "lasso_alpha": lasso_alpha, "enet_alpha": enet_alpha, "l1_ratio": l1_ratio},
        "alpha_note": "scikit-learn scales the Ridge penalty by the raw sum of squares (about 200,000 rows here) but the Lasso / Elastic Net penalty by 1/(2n), so useful alpha ranges differ by orders of magnitude.",
        "summary": summary,
        "coefficients": [{"feature": f, **{k: r(coefs[k][i]) for k in models}} for i, f in enumerate(names)],
        "path": path,
    }


def _logistic(C_value: float, regularization: str, class_weight: str) -> LogisticRegression:
    cw = "balanced" if class_weight == "balanced" else None
    if regularization == "none":
        return LogisticRegression(C=np.inf, max_iter=5000, class_weight=cw)
    if regularization == "l1":
        return LogisticRegression(C=C_value, l1_ratio=1.0, solver="saga", max_iter=5000, class_weight=cw)
    return LogisticRegression(C=C_value, max_iter=5000, class_weight=cw)


def logistic(data: LabData, C_value: float = 1.0, regularization: str = "l2", threshold: float = 0.5, class_weight: str = "balanced") -> dict[str, Any]:
    tr, va = data.train, data.validation
    feats = list(C.FEATURE_NAMES)
    pre = build_preprocessor()
    X_tr = pre.fit_transform(tr[feats])
    X_va = pre.transform(va[feats])
    y_tr, y_va = tr[C.TARGET].to_numpy(), va[C.TARGET].to_numpy()
    clf = _logistic(C_value, regularization, class_weight).fit(X_tr, y_tr)
    names = transformed_feature_names(pre)
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]

    def bce(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    z = clf.decision_function(X_va)
    idx = scatter_sample(len(z), 450, seed=1, must_include=np.flatnonzero(y_va == 1))
    curve = [{"z": r(v, 3), "sigmoid": r(1 / (1 + np.exp(-v)))} for v in np.linspace(-12, 12, 97)]
    coef_rows = sorted(({"feature": n, "coefficient": r(c), "odds_ratio": r(np.exp(c))} for n, c in zip(names, clf.coef_[0])), key=lambda d: -abs(d["coefficient"]))
    return {
        "split": data.split_info(tr, va, "Preprocessing (hour-of-cycle, log amount, scaling) fitted on the training split only."),
        "params": {"C": C_value, "regularization": regularization, "threshold": threshold, "class_weight": class_weight},
        "intercept": r(clf.intercept_[0]),
        "coefficients": coef_rows,
        "zero_coefficients": int(np.sum(np.abs(clf.coef_) < 1e-8)),
        "sigmoid_curve": curve,
        "sigmoid_points": [{"z": r(z[i], 3), "probability": r(p_va[i]), "label": int(y_va[i])} for i in idx],
        "sigmoid_points_note": "All validation fraud rows plus a random sample of legitimate rows.",
        "binary_cross_entropy": {
            "train_manual": r(bce(y_tr, p_tr), 6),
            "train_sklearn": r(log_loss(y_tr, p_tr), 6),
            "validation_manual": r(bce(y_va, p_va), 6),
            "validation_sklearn": r(log_loss(y_va, p_va), 6),
        },
        "class_weight_note": (
            "With balanced class weights each fraud row counts about 600 times as much as a legitimate row, so predicted probabilities are inflated "
            "and a 0.5 threshold produces many false positives. The production model corrects the intercept for this (prior correction)."
            if class_weight == "balanced"
            else "Without class weights the model sees 0.17 % positives; probabilities stay close to the base rate and a 0.5 threshold flags few rows."
        ),
        "validation_metrics": M.classification_metrics(y_va, p_va, threshold),
        "threshold_sweep": M.threshold_sweep(y_va, p_va),
        "multinomial": _multinomial_amount_band(data),
    }


def _multinomial_amount_band(data: LabData, train_rows: int = 40000) -> dict[str, Any]:
    """Multinomial logistic regression: amount band (tertiles learned on train) from V1-V28."""
    tr = data.sample("train", train_rows)
    va = data.validation
    edges = np.quantile(tr["Amount"], [1 / 3, 2 / 3])
    labels = ["Low amount", "Middle amount", "High amount"]
    y_tr = np.digitize(tr["Amount"], edges)
    y_va = np.digitize(va["Amount"], edges)
    X_tr, X_va = scaled_matrix(tr, PCA, va)
    clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    majority = np.bincount(y_tr).argmax()
    top = {}
    for k, lab in enumerate(labels):
        order = np.argsort(-np.abs(clf.coef_[k]))[:6]
        top[lab] = [{"feature": PCA[i], "coefficient": r(clf.coef_[k][i])} for i in order]
    return {
        "target": "Amount band (tertile edges learned on the training split), predicted from V1-V28",
        "band_edges": r(edges, 2),
        "classes": labels,
        "validation_accuracy": r(accuracy_score(y_va, pred)),
        "validation_macro_f1": r(f1_score(y_va, pred, average="macro")),
        "majority_baseline_accuracy": r(np.mean(y_va == majority)),
        "confusion_matrix": confusion_matrix(y_va, pred, labels=[0, 1, 2]).tolist(),
        "top_coefficients": top,
    }


def preprocessing(data: LabData, feature: str = "Amount", missing_rate: float = 0.1) -> dict[str, Any]:
    if feature not in C.FEATURE_NAMES:
        raise ValueError(f"'{feature}' is not a contract feature")
    tr, va = data.train, data.validation
    y_tr, y_va = tr[C.TARGET].to_numpy(), va[C.TARGET].to_numpy()

    # ---------------- scaling (fitted on train only)
    col_tr, col_va = tr[[feature]], va[[feature]]
    ss = StandardScaler().fit(col_tr)
    mm = MinMaxScaler().fit(col_tr)

    def hist(v: np.ndarray, bins: int = 24) -> list[dict]:
        counts, edges = np.histogram(v, bins=bins)
        return [{"x": r((edges[i] + edges[i + 1]) / 2, 3), "count": int(counts[i])} for i in range(bins)]

    s_va, m_va = ss.transform(col_va).ravel(), mm.transform(col_va).ravel()
    histograms = {"raw": hist(col_tr.to_numpy().ravel()), "standard": hist(ss.transform(col_tr).ravel()), "minmax": hist(mm.transform(col_tr).ravel())}
    if feature == "Amount":
        histograms["log1p"] = hist(np.log1p(col_tr.to_numpy().ravel()))
    scaling = {
        "feature": feature,
        "standard": {"mean_learned": r(ss.mean_[0]), "std_learned": r(ss.scale_[0]), "validation_mean_after": r(s_va.mean()), "validation_std_after": r(s_va.std())},
        "minmax": {"min_learned": r(mm.data_min_[0]), "max_learned": r(mm.data_max_[0]), "validation_min_after": r(m_va.min()), "validation_max_after": r(m_va.max())},
        "histograms": histograms,
        "note": "Validation statistics are not exactly 0/1 (or inside [0, 1]) because the scalers reuse training parameters instead of re-fitting.",
    }

    # ---------------- encoding (categoricals derived from the data; the dataset itself has none)
    hours = tr[["hour"]].astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(hours)
    band_edges = np.quantile(tr["Amount"], [1 / 3, 2 / 3])
    band_names = np.array(["Low", "Mid", "High"])
    band_tr = pd.DataFrame({"amount_band": band_names[np.digitize(tr["Amount"], band_edges)]})
    ordc = OrdinalEncoder(categories=[["Low", "Mid", "High"]]).fit(band_tr)
    naive_hour = tr.groupby("hour")[C.TARGET].mean()
    te_hour = TargetEncoder(target_type="binary", cv=5, random_state=0).fit(hours, y_tr)
    target_hour = [
        {"hour": int(h), "rows": int((tr["hour"] == int(h)).sum()), "naive_fraud_rate": r(naive_hour.get(int(h)), 5), "cross_fitted_full_fit": r(v, 5)}
        for h, v in sorted(zip(te_hour.categories_[0], te_hour.encodings_[0]), key=lambda kv: int(kv[0]))
    ]
    # Leakage demo with an identifier column. The dataset has no transaction ID, so each row gets a
    # unique generated ID; naive target encoding then memorises every training label.
    ids_tr = pd.DataFrame({"row_id": [f"t{i}" for i in range(len(tr))]})
    ids_va = pd.DataFrame({"row_id": [f"v{i}" for i in range(len(va))]})
    global_mean = float(y_tr.mean())
    naive_tr = y_tr.astype(float)  # every training ID maps to its own label
    naive_va = np.full(len(va), global_mean)  # unseen IDs fall back to the global mean
    te = TargetEncoder(target_type="binary", cv=5, random_state=0)
    cf_tr = te.fit_transform(ids_tr, y_tr).ravel()  # cross-fitted: each row encoded by out-of-fold statistics
    cf_va = te.transform(ids_va).ravel()

    def auc(y: np.ndarray, s: np.ndarray) -> float | None:
        return r(roc_auc_score(y, s)) if np.std(s) > 0 else 0.5

    encoding = {
        "one_hot": {"source": "hour of cycle (0-23, from Time)", "output_columns": [str(c) for c in ohe.get_feature_names_out()]},
        "ordinal": {
            "source": "amount band (tertiles learned on the training split)",
            "edges": r(band_edges, 2),
            "mapping": {c: int(i) for i, c in enumerate(ordc.categories_[0])},
            "note": "Low < Mid < High is a real ordering.",
        },
        "target_encoding_hour": target_hour,
        "leakage_demo": {
            "column": "row_id (a generated unique identifier; the dataset has no transaction ID)",
            "naive": {"train_roc_auc": auc(y_tr, naive_tr), "validation_roc_auc": auc(y_va, naive_va)},
            "cross_fitted": {"train_roc_auc": auc(y_tr, cf_tr), "validation_roc_auc": auc(y_va, cf_va)},
            "explanation": "Naively encoding a unique identifier with its own label looks perfect on training data and collapses on unseen rows; cross-fitting exposes this immediately.",
        },
    }

    # ---------------- imputation on SIMULATED missingness (the dataset has no missing values)
    rng = np.random.default_rng(0)
    cols = ["V14", "V17", "V12", "V10", "Amount"]
    sample = data.sample("train", 60000)
    numeric = list(C.FEATURE_NAMES)

    def inject(df: pd.DataFrame) -> pd.DataFrame:
        out = df[numeric].copy()
        for c in cols:
            mask = rng.random(len(out)) < missing_rate
            out.loc[mask, c] = np.nan
        return out

    X_s, X_v = inject(sample), inject(va)
    y_s = sample[C.TARGET].to_numpy()
    missing = []
    for c in cols:
        m_tr = X_s[c].isna().to_numpy()
        missing.append(
            {
                "feature": c,
                "missing_rate_train": r(m_tr.mean()),
                "train_mean": r(X_s[c].mean()),
                "train_median": r(X_s[c].median()),
                "fraud_rate_when_missing": r(y_s[m_tr].mean(), 5),
                "fraud_rate_when_present": r(y_s[~m_tr].mean(), 5),
            }
        )
    strategies = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "median + MissingIndicator": SimpleImputer(strategy="median", add_indicator=True),
        "iterative (model-based) + MissingIndicator": IterativeImputer(max_iter=5, random_state=0, add_indicator=True),
    }
    comparison = []
    for name, imp in strategies.items():
        t = time.perf_counter()
        pipe = Pipeline([("impute", imp), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        pipe.fit(X_s, y_s)
        p = pipe.predict_proba(X_v)[:, 1]
        comparison.append({"strategy": name, "validation_average_precision": r(average_precision_score(y_va, p)), "validation_roc_auc": r(roc_auc_score(y_va, p)), "fit_ms": r((time.perf_counter() - t) * 1000, 1)})
    ind = MissingIndicator().fit(X_s[cols])
    return {
        "split": data.split_info(tr, va, "All transforms are fitted on training data and only applied to validation data."),
        "scaling": scaling,
        "encoding": encoding,
        "imputation": {
            "simulated": True,
            "simulation": f"creditcard.csv has no missing values, so {missing_rate:.0%} of {', '.join(cols)} were blanked completely at random (MCAR) in a 60,000-row training sample and in the validation split.",
            "missingness": missing,
            "missing_indicator_columns": [cols[i] for i in ind.features_],
            "downstream_comparison": comparison,
            "comparison_model": "Class-weighted logistic regression on the 30 raw features",
            "note": "Because the missingness is random (MCAR), it carries no information about fraud, so the indicator columns should not help; with real, informative missingness they can.",
        },
    }
