"""M2: linear regression, regularisation, logistic regression and preprocessing experiments."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, lasso_path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from .. import contract as C
from .. import metrics as M
from ..features import build_preprocessor
from .common import CORE_NUMERIC, LabData, r, scatter_sample

TIER_ORDER = list(C.FEATURES_BY_NAME["CollegeTier"].categories)  # Tier3 < Tier2 < Tier1
REG_FEATURES = [*C.SEMESTERS, *CORE_NUMERIC, "CollegeTier_ordinal"]


def _salary_design(data: LabData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Salary regression among placed students; preprocessing is fitted on the training split only."""

    def frame(df: pd.DataFrame) -> pd.DataFrame:
        d = df[(df[C.TARGET] == 1) & (df[C.REGRESSION_TARGET] > 0)].copy()
        d["CollegeTier_ordinal"] = d["CollegeTier"].map({t: i for i, t in enumerate(TIER_ORDER)}).astype(float)
        return d

    tr, va = frame(data.train), frame(data.validation)
    pre = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X_tr = pre.fit_transform(tr[REG_FEATURES])
    X_va = pre.transform(va[REG_FEATURES])
    info = data.split_info(len(tr), len(va), "Placed students with a positive salary; target = Salary Package (LPA).")
    return X_tr, tr[C.REGRESSION_TARGET].to_numpy(float), X_va, va[C.REGRESSION_TARGET].to_numpy(float), REG_FEATURES, info


def linear_regression(data: LabData, learning_rate: float = 0.05, iterations: int = 400) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _salary_design(data)
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
    return {
        "split": info,
        "target": "Salary Package (LPA)",
        "coefficients": [
            {
                "feature": f,
                "normal_equation": r(theta_ne[i + 1]),
                "sklearn": r(theta_sk[i + 1]),
                "gradient_descent": None if diverged else r(theta[i + 1]),
            }
            for i, f in enumerate(names)
        ],
        "intercept": {"normal_equation": r(theta_ne[0]), "sklearn": r(theta_sk[0]), "gradient_descent": None if diverged else r(theta[0])},
        "agreement": {
            "max_abs_diff_normal_eq_vs_sklearn": r(np.max(np.abs(theta_ne - theta_sk)), 10),
            "max_abs_diff_gd_vs_normal_eq": None if diverged else r(np.max(np.abs(theta - theta_ne)), 6),
            "train_mse_normal_eq": r(ne_mse),
            "train_mse_gd": None if diverged else r(np.mean((Xb @ theta - y_tr) ** 2)),
        },
        "gradient_descent": {
            "learning_rate": learning_rate,
            "iterations": iterations,
            "diverged": diverged,
            "loss_history": history,
            "duration_ms": r(gd_ms, 2),
        },
        "normal_equation_ms": r(ne_ms, 3),
        "conditioning": {
            "condition_number": r(float(eig.max() / max(eig.min(), 1e-12)), 2),
            "smallest_eigenvalue": r(eig.min(), 6),
            "note": "A large condition number (correlated SGPA columns) makes gradient descent converge slowly along some directions.",
        },
        "validation_metrics": M.regression_metrics(y_va, pred_va),
        "train_metrics": M.regression_metrics(y_tr, Xb @ theta_ne),
        "predictions_sample": [{"actual": r(y_va[i], 3), "predicted": r(pred_va[i], 3)} for i in idx],
    }


def regularization(
    data: LabData, ridge_alpha: float = 100.0, lasso_alpha: float = 0.05, enet_alpha: float = 0.05, l1_ratio: float = 0.5
) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _salary_design(data)
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
    ridge_alphas = np.logspace(-2, 5, 22)
    lasso_alphas = np.logspace(0.5, -4, 22)  # lasso_path expects decreasing alphas
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
        "alpha_note": "scikit-learn scales the Ridge penalty by the raw sum of squares but the Lasso/Elastic Net penalty by 1/(2n), so useful alpha ranges differ.",
        "summary": summary,
        "coefficients": [{"feature": f, **{k: r(coefs[k][i]) for k in models}} for i, f in enumerate(names)],
        "path": path,
    }


def _logistic(C_value: float, regularization: str) -> LogisticRegression:
    if regularization == "none":
        return LogisticRegression(C=np.inf, max_iter=5000)
    if regularization == "l1":
        return LogisticRegression(C=C_value, l1_ratio=1.0, solver="saga", max_iter=5000)
    return LogisticRegression(C=C_value, max_iter=5000)


def logistic(data: LabData, C_value: float = 1.0, regularization: str = "l2", threshold: float = 0.5, train_rows: int = 12000) -> dict[str, Any]:
    tr = data.sample("train", train_rows)
    va = data.validation
    feats = list(C.FEATURE_NAMES)
    pre = build_preprocessor(feats, engineered=False)
    X_tr = pre.fit_transform(tr[feats])
    X_va = pre.transform(va[feats])
    y_tr, y_va = tr[C.TARGET].to_numpy(), va[C.TARGET].to_numpy()
    clf = _logistic(C_value, regularization).fit(X_tr, y_tr)
    names = [str(n) for n in pre.get_feature_names_out()]
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]

    def bce(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    z = clf.decision_function(X_va)
    idx = scatter_sample(len(z), 350, seed=1)
    curve = [{"z": r(v, 3), "sigmoid": r(1 / (1 + np.exp(-v)))} for v in np.linspace(-10, 10, 81)]
    coef_rows = sorted(
        ({"feature": n, "coefficient": r(c), "odds_ratio": r(np.exp(c))} for n, c in zip(names, clf.coef_[0])),
        key=lambda d: -abs(d["coefficient"]),
    )
    return {
        "split": data.split_info(len(tr), len(va), "Preprocessing (impute, scale, one-hot, ordinal) fitted on the training sample only."),
        "params": {"C": C_value, "regularization": regularization, "threshold": threshold},
        "intercept": r(clf.intercept_[0]),
        "coefficients": coef_rows,
        "zero_coefficients": int(np.sum(np.abs(clf.coef_) < 1e-8)),
        "sigmoid_curve": curve,
        "sigmoid_points": [{"z": r(z[i], 3), "probability": r(p_va[i]), "label": int(y_va[i])} for i in idx],
        "binary_cross_entropy": {
            "train_manual": r(bce(y_tr, p_tr)),
            "train_sklearn": r(log_loss(y_tr, p_tr)),
            "validation_manual": r(bce(y_va, p_va)),
            "validation_sklearn": r(log_loss(y_va, p_va)),
        },
        "validation_metrics": M.classification_metrics(y_va, p_va, threshold),
        "threshold_sweep": M.threshold_sweep(y_va, p_va),
        "multinomial": _multinomial_salary_band(data),
    }


def _multinomial_salary_band(data: LabData) -> dict[str, Any]:
    """Multinomial logistic regression: salary band (tertiles learned on train) for placed students."""
    feats = list(C.FEATURE_NAMES)
    tr = data.train[(data.train[C.TARGET] == 1) & (data.train[C.REGRESSION_TARGET] > 0)]
    va = data.validation[(data.validation[C.TARGET] == 1) & (data.validation[C.REGRESSION_TARGET] > 0)]
    edges = np.quantile(tr[C.REGRESSION_TARGET], [1 / 3, 2 / 3])
    labels = ["Lower band", "Middle band", "Upper band"]
    y_tr = np.digitize(tr[C.REGRESSION_TARGET], edges)
    y_va = np.digitize(va[C.REGRESSION_TARGET], edges)
    pre = build_preprocessor(feats, engineered=False)
    X_tr, X_va = pre.fit_transform(tr[feats]), pre.transform(va[feats])
    clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    names = [str(n) for n in pre.get_feature_names_out()]
    majority = np.bincount(y_tr).argmax()
    top = {}
    for k, lab in enumerate(labels):
        order = np.argsort(-np.abs(clf.coef_[k]))[:6]
        top[lab] = [{"feature": names[i], "coefficient": r(clf.coef_[k][i])} for i in order]
    return {
        "target": "Salary band among placed students (tertile edges learned on the training split)",
        "band_edges_lpa": r(edges, 2),
        "classes": labels,
        "validation_accuracy": r(accuracy_score(y_va, pred)),
        "validation_macro_f1": r(f1_score(y_va, pred, average="macro")),
        "majority_baseline_accuracy": r(np.mean(y_va == majority)),
        "confusion_matrix": confusion_matrix(y_va, pred, labels=[0, 1, 2]).tolist(),
        "top_coefficients": top,
    }


def preprocessing(data: LabData, feature: str = "CodingTestScore") -> dict[str, Any]:
    if feature not in [f.name for f in C.NUMERIC_FEATURES]:
        raise ValueError(f"'{feature}' is not a numeric contract feature")
    tr, va = data.train, data.validation

    # ---------------- scaling (fitted on train only)
    col_tr = tr[[feature]].dropna()
    col_va = va[[feature]].dropna()
    ss = StandardScaler().fit(col_tr)
    mm = MinMaxScaler().fit(col_tr)

    def hist(v: np.ndarray, bins: int = 24) -> list[dict]:
        counts, edges = np.histogram(v, bins=bins)
        return [{"x": r((edges[i] + edges[i + 1]) / 2, 3), "count": int(counts[i])} for i in range(bins)]

    s_va, m_va = ss.transform(col_va).ravel(), mm.transform(col_va).ravel()
    scaling = {
        "feature": feature,
        "standard": {"mean_learned": r(ss.mean_[0]), "std_learned": r(ss.scale_[0]), "validation_mean_after": r(s_va.mean()), "validation_std_after": r(s_va.std())},
        "minmax": {"min_learned": r(mm.data_min_[0]), "max_learned": r(mm.data_max_[0]), "validation_min_after": r(m_va.min()), "validation_max_after": r(m_va.max())},
        "histograms": {
            "raw": hist(col_tr.to_numpy().ravel()),
            "standard": hist(ss.transform(col_tr).ravel()),
            "minmax": hist(mm.transform(col_tr).ravel()),
        },
        "note": "Validation statistics are not exactly 0/1 because the scaler reuses training parameters instead of re-fitting.",
    }

    # ---------------- encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[["City"]])
    ordc = OrdinalEncoder(categories=[TIER_ORDER]).fit(tr[["CollegeTier"]])
    y_tr = tr[C.TARGET].to_numpy()
    y_va = va[C.TARGET].to_numpy()
    naive_city = tr.groupby("City")[C.TARGET].mean()
    te_city = TargetEncoder(target_type="binary", cv=5).fit(tr[["City"]], y_tr)
    target_city = [
        {"city": c, "naive_mean": r(naive_city.get(c)), "cross_fitted_full_fit": r(v)}
        for c, v in zip(te_city.categories_[0], te_city.encodings_[0])
    ]
    # Leakage demo with an identifier column: naive target encoding memorises each row's label.
    ids_tr = tr[[C.ID_COLUMN]].astype(str)
    ids_va = va[[C.ID_COLUMN]].astype(str)
    naive_map = pd.Series(y_tr, index=ids_tr[C.ID_COLUMN].to_numpy())
    global_mean = float(y_tr.mean())
    naive_tr = ids_tr[C.ID_COLUMN].map(naive_map).to_numpy(float)
    naive_va = ids_va[C.ID_COLUMN].map(naive_map).fillna(global_mean).to_numpy(float)
    te = TargetEncoder(target_type="binary", cv=5)
    cf_tr = te.fit_transform(ids_tr, y_tr).ravel()  # cross-fitted: each row encoded by out-of-fold statistics
    cf_va = te.transform(ids_va).ravel()

    def auc(y: np.ndarray, s: np.ndarray) -> float | None:
        return r(roc_auc_score(y, s)) if np.std(s) > 0 else 0.5

    encoding = {
        "one_hot": {"source": "City", "output_columns": [str(c) for c in ohe.get_feature_names_out()]},
        "ordinal": {"source": "CollegeTier", "mapping": {c: int(i) for i, c in enumerate(ordc.categories_[0])}, "note": "Tier3 < Tier2 < Tier1 is a real ordering."},
        "target_encoding_city": target_city,
        "leakage_demo": {
            "column": C.ID_COLUMN,
            "naive": {"train_roc_auc": auc(y_tr, naive_tr), "validation_roc_auc": auc(y_va, naive_va)},
            "cross_fitted": {"train_roc_auc": auc(y_tr, cf_tr), "validation_roc_auc": auc(y_va, cf_va)},
            "explanation": "Naively encoding a unique identifier with its own label looks perfect on training data and collapses on unseen rows; cross-fitting exposes this immediately.",
        },
    }

    # ---------------- imputation
    nullable = [f.name for f in C.NUMERIC_FEATURES if f.nullable]
    numeric = [f.name for f in C.NUMERIC_FEATURES]
    missing = []
    for c in nullable:
        m_tr = tr[c].isna()
        missing.append(
            {
                "feature": c,
                "missing_rate_train": r(m_tr.mean()),
                "train_mean": r(tr[c].mean()),
                "train_median": r(tr[c].median()),
                "placed_rate_when_missing": r(tr.loc[m_tr, C.TARGET].mean()),
                "placed_rate_when_present": r(tr.loc[~m_tr, C.TARGET].mean()),
            }
        )
    strategies = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "median + MissingIndicator": SimpleImputer(strategy="median", add_indicator=True),
        "iterative (model-based) + MissingIndicator": IterativeImputer(max_iter=5, random_state=0, add_indicator=True),
    }
    sample = data.sample("train", 15000)
    comparison = []
    for name, imp in strategies.items():
        t = time.perf_counter()
        pipe = Pipeline([("impute", imp), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
        pipe.fit(sample[numeric], sample[C.TARGET])
        p = pipe.predict_proba(va[numeric])[:, 1]
        comparison.append(
            {
                "strategy": name,
                "validation_roc_auc": r(roc_auc_score(y_va, p)),
                "validation_log_loss": r(log_loss(y_va, p)),
                "fit_ms": r((time.perf_counter() - t) * 1000, 1),
            }
        )
    ind = MissingIndicator().fit(tr[nullable])
    return {
        "split": data.split_info(len(tr), len(va), "All transforms are fitted on training data and only applied to validation data."),
        "scaling": scaling,
        "encoding": encoding,
        "imputation": {
            "missingness": missing,
            "missing_indicator_columns": [nullable[i] for i in ind.features_],
            "downstream_comparison": comparison,
            "comparison_model": "Logistic regression on numeric features (15,000-row training sample)",
        },
    }
