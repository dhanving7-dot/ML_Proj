"""M3: decision trees, pruning, ensembles (bagging, boosting) and feature importance.

Accuracy is useless on this data (predicting "legitimate" for everything scores 99.8 %), so trees
are compared with average precision (PR-AUC) and ROC-AUC on the untouched validation split.
Tree experiments train on every training fraud plus an undersampled set of legitimate rows so
they run in seconds; the validation split keeps its natural 0.17 % fraud rate.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import contract as C
from .. import metrics as M
from ..estimators import CLASSIFIER_LABELS, make_classifier, optional_available
from ..features import build_pipeline, build_preprocessor, transformed_feature_names
from .common import PCA, LabData, r
from .m2_linear import REG_FEATURES, _amount_design

ENRICHED_LEGIT = 20000


def _design(data: LabData, n_legit: int = ENRICHED_LEGIT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    tr = data.enriched("train", n_legit)
    va = data.validation
    feats = list(C.FEATURE_NAMES)
    pre = build_preprocessor()
    X_tr, X_va = pre.fit_transform(tr[feats]), pre.transform(va[feats])
    info = data.split_info(tr, va, f"Training sample = every training fraud + {n_legit:,} legitimate rows (fraud rate {tr[C.TARGET].mean():.2%}); validation keeps the real rate.")
    return X_tr, tr[C.TARGET].to_numpy(), X_va, va[C.TARGET].to_numpy(), transformed_feature_names(pre), info


def _rank(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    return {"average_precision": r(average_precision_score(y, p)), "roc_auc": r(roc_auc_score(y, p))}


def _gini(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(1 - np.sum(p**2))


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _node_counts(tree: Any, node: int) -> np.ndarray:
    # tree_.value stores class fractions for classifiers; multiply by the node's weighted size.
    return np.asarray(tree.value[node][0] * tree.weighted_n_node_samples[node])


def decision_tree(data: LabData, criterion: str = "gini", max_depth: int | None = 4, min_samples_leaf: int = 20, ccp_alpha: float = 0.0) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _design(data)
    t = time.perf_counter()
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, random_state=0).fit(X_tr, y_tr)
    fit_ms = (time.perf_counter() - t) * 1000
    tree = clf.tree_
    nodes = []
    frontier = [(0, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        counts = _node_counts(tree, node)
        leaf = tree.children_left[node] == -1
        nodes.append(
            {
                "id": int(node),
                "depth": depth,
                "feature": None if leaf else names[tree.feature[node]],
                "threshold": None if leaf else r(tree.threshold[node]),
                "impurity": r(tree.impurity[node]),
                "samples": int(tree.n_node_samples[node]),
                "counts": [int(round(c)) for c in counts],
                "fraud_rate": r(counts[1] / counts.sum()),
                "left": None if leaf else int(tree.children_left[node]),
                "right": None if leaf else int(tree.children_right[node]),
            }
        )
        if not leaf and depth < 3:
            frontier += [(int(tree.children_left[node]), depth + 1), (int(tree.children_right[node]), depth + 1)]

    root_split = None
    if tree.children_left[0] != -1:
        L, R = int(tree.children_left[0]), int(tree.children_right[0])
        pc, lc, rc = _node_counts(tree, 0), _node_counts(tree, L), _node_counts(tree, R)
        wl, wr = lc.sum() / pc.sum(), rc.sum() / pc.sum()
        root_split = {
            "feature": names[tree.feature[0]],
            "threshold": r(tree.threshold[0]),
            "parent_counts": [int(round(c)) for c in pc],
            "left_counts": [int(round(c)) for c in lc],
            "right_counts": [int(round(c)) for c in rc],
            "gini": {
                "parent": r(_gini(pc)),
                "left": r(_gini(lc)),
                "right": r(_gini(rc)),
                "weighted_children": r(wl * _gini(lc) + wr * _gini(rc)),
                "decrease": r(_gini(pc) - wl * _gini(lc) - wr * _gini(rc)),
            },
            "entropy": {
                "parent": r(_entropy(pc)),
                "left": r(_entropy(lc)),
                "right": r(_entropy(rc)),
                "weighted_children": r(wl * _entropy(lc) + wr * _entropy(rc)),
                "information_gain": r(_entropy(pc) - wl * _entropy(lc) - wr * _entropy(rc)),
            },
            "sklearn_parent_impurity": r(tree.impurity[0]),
        }
    p_tr, p_va = clf.predict_proba(X_tr)[:, 1], clf.predict_proba(X_va)[:, 1]
    imp = sorted(({"feature": n, "importance": r(v)} for n, v in zip(names, clf.feature_importances_)), key=lambda d: -d["importance"])
    return {
        "split": info,
        "params": {"criterion": criterion, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "ccp_alpha": ccp_alpha},
        "depth": int(clf.get_depth()),
        "leaves": int(clf.get_n_leaves()),
        "fit_ms": r(fit_ms, 1),
        "train": _rank(y_tr, p_tr),
        "validation": {**_rank(y_va, p_va), "at_0_5": M.classification_metrics(y_va, p_va, 0.5, probabilistic=False)},
        "nodes": nodes,
        "root_split": root_split,
        "feature_importance": [d for d in imp if d["importance"] > 0][:15],
    }


def pruning(data: LabData, criterion: str = "gini") -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _design(data, 10000)
    depth_curve = []
    for d in list(range(1, 21)) + [None]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=0).fit(X_tr, y_tr)
        depth_curve.append(
            {
                "max_depth": d if d is not None else "none",
                "train_average_precision": r(average_precision_score(y_tr, clf.predict_proba(X_tr)[:, 1])),
                "validation_average_precision": r(average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])),
                "leaves": int(clf.get_n_leaves()),
            }
        )
    path = DecisionTreeClassifier(criterion=criterion, random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
    alphas = np.unique(path.ccp_alphas)
    picks = np.unique(np.quantile(alphas, np.linspace(0, 1, 30)))
    ccp_curve = []
    for a in picks:
        clf = DecisionTreeClassifier(criterion=criterion, ccp_alpha=float(a), random_state=0).fit(X_tr, y_tr)
        ccp_curve.append(
            {
                "ccp_alpha": float(f"{a:.6g}"),
                "train_average_precision": r(average_precision_score(y_tr, clf.predict_proba(X_tr)[:, 1])),
                "validation_average_precision": r(average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])),
                "leaves": int(clf.get_n_leaves()),
                "depth": int(clf.get_depth()),
            }
        )
    best_depth = max(depth_curve, key=lambda d: d["validation_average_precision"])
    best_ccp = max(ccp_curve, key=lambda d: (d["validation_average_precision"], -d["leaves"]))
    return {
        "split": info,
        "pre_pruning": {"curve": depth_curve, "best": best_depth},
        "post_pruning": {"alphas_in_path": int(len(alphas)), "curve": ccp_curve, "best": best_ccp},
        "note": "Hyperparameters are compared on the validation split; an unbiased estimate would need a separate test set.",
    }


def regression_tree(data: LabData, max_depth: int = 4, train_rows: int = 60000) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _amount_design(data)
    rows = np.random.default_rng(0).choice(len(y_tr), min(train_rows, len(y_tr)), replace=False)
    X_tr, y_tr = X_tr[rows], y_tr[rows]
    info = {**info, "train_rows_used": int(len(rows)), "train_fraud_rows": None, "note": info["note"] + f" Trees are fitted on a random {len(rows):,}-row training sample for speed."}
    reg = DecisionTreeRegressor(criterion="squared_error", max_depth=max_depth, min_samples_leaf=20, random_state=0).fit(X_tr, y_tr)
    tree = reg.tree_
    L, R = int(tree.children_left[0]), int(tree.children_right[0])
    n0, nl, nr = tree.n_node_samples[0], tree.n_node_samples[L], tree.n_node_samples[R]
    root = {
        "feature": names[tree.feature[0]],
        "threshold_scaled": r(tree.threshold[0]),
        "parent_mse": r(tree.impurity[0]),
        "left": {"samples": int(nl), "mse": r(tree.impurity[L]), "mean_target": r(tree.value[L][0][0])},
        "right": {"samples": int(nr), "mse": r(tree.impurity[R]), "mean_target": r(tree.value[R][0][0])},
        "weighted_children_mse": r((nl * tree.impurity[L] + nr * tree.impurity[R]) / n0),
        "mse_reduction": r(tree.impurity[0] - (nl * tree.impurity[L] + nr * tree.impurity[R]) / n0),
    }
    curve = []
    for d in range(1, 16):
        m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=0).fit(X_tr, y_tr)
        curve.append({"max_depth": d, "train_rmse": M.regression_metrics(y_tr, m.predict(X_tr))["rmse"], "validation_rmse": M.regression_metrics(y_va, m.predict(X_va))["rmse"]})
    return {
        "split": info,
        "target": "log(1 + Amount)",
        "features": REG_FEATURES,
        "root_split": root,
        "validation_metrics": M.regression_metrics(y_va, reg.predict(X_va)),
        "depth_curve": curve,
    }


def ensembles(data: LabData, n_estimators: int = 200, max_features: str = "sqrt", learning_rate: float = 0.1) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, names, info = _design(data)
    mf: Any = None if max_features == "all" else max_features
    models: dict[str, Any] = {
        "decision_tree": DecisionTreeClassifier(random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=n_estimators, max_features=mf, bootstrap=True, oob_score=True, n_jobs=-1, random_state=0),
        "adaboost": AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=3, random_state=0),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=learning_rate, early_stopping=False, random_state=0),
    }
    if optional_available("xgboost"):
        models["xgboost"] = make_classifier("xgboost", seed=0, n_estimators=n_estimators, learning_rate=learning_rate)
    if optional_available("lightgbm"):
        models["lightgbm"] = make_classifier("lightgbm", seed=0, n_estimators=n_estimators, learning_rate=learning_rate)
    rows = []
    fitted: dict[str, Any] = {}
    for key, m in models.items():
        t = time.perf_counter()
        m.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t
        fitted[key] = m
        p_tr, p_va = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]
        ap_tr, ap_va = average_precision_score(y_tr, p_tr), average_precision_score(y_va, p_va)
        rows.append(
            {
                "model": key,
                "label": CLASSIFIER_LABELS[key],
                "family": "single tree" if key == "decision_tree" else ("bagging" if key == "random_forest" else "boosting"),
                "train_average_precision": r(ap_tr),
                "validation_average_precision": r(ap_va),
                "validation_roc_auc": r(roc_auc_score(y_va, p_va)),
                "validation_log_loss": r(log_loss(y_va, np.clip(p_va, 1e-7, 1 - 1e-7))),
                "overfit_gap": r(ap_tr - ap_va),
                "fit_seconds": r(fit_s, 3),
            }
        )
    rf = fitted["random_forest"]
    oob_curve = []
    rf_ws = RandomForestClassifier(n_estimators=10, max_features=mf, oob_score=True, warm_start=True, n_jobs=-1, random_state=0)
    for n in [10, 25, 50, 75, 100, 150, 200, 300, 400]:
        if n > max(n_estimators, 50):
            break
        rf_ws.set_params(n_estimators=n)
        rf_ws.fit(X_tr, y_tr)
        oob = rf_ws.oob_decision_function_[:, 1]
        ok = np.isfinite(oob)
        oob_curve.append(
            {
                "n_estimators": n,
                "oob_average_precision": r(average_precision_score(y_tr[ok], oob[ok])),
                "validation_average_precision": r(average_precision_score(y_va, rf_ws.predict_proba(X_va)[:, 1])),
            }
        )
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y_tr), len(y_tr))
    gb = fitted["gradient_boosting"]
    staged = []
    for i, (pt, pv) in enumerate(zip(gb.staged_predict_proba(X_tr), gb.staged_predict_proba(X_va)), 1):
        if i <= 10 or i % max(1, n_estimators // 40) == 0 or i == n_estimators:
            staged.append({"stage": i, "train_log_loss": r(log_loss(y_tr, pt[:, 1])), "validation_log_loss": r(log_loss(y_va, pv[:, 1]))})
    ada = fitted["adaboost"]
    oob_p = rf.oob_decision_function_[:, 1]
    ok = np.isfinite(oob_p)
    return {
        "split": info,
        "params": {"n_estimators": n_estimators, "max_features": max_features, "learning_rate": learning_rate},
        "protocol": "Every model is fitted on the same training sample and scored on the same validation split.",
        "comparison": sorted(rows, key=lambda d: -d["validation_average_precision"]),
        "random_forest": {
            "oob_average_precision": r(average_precision_score(y_tr[ok], oob_p[ok])),
            "oob_curve": oob_curve,
            "bootstrap_unique_fraction": r(np.unique(boot).size / len(boot)),
            "expected_unique_fraction": r(1 - np.exp(-1)),
            "features_per_split": "sqrt(p)" if max_features == "sqrt" else ("log2(p)" if max_features == "log2" else "all p"),
            "n_features": len(names),
        },
        "gradient_boosting_stages": staged,
        "adaboost": {"first_estimator_weights": [r(w) for w in ada.estimator_weights_[:25]], "first_estimator_errors": [r(e) for e in ada.estimator_errors_[:25]]},
        "note": "Validation has only about 70 fraud rows, so PR-AUC differences of a few hundredths are within sampling noise; see M5 for a paired bootstrap.",
    }


def importance(data: LabData, model: str = "random_forest", explain_index: int = 0) -> dict[str, Any]:
    if model not in ("random_forest", "gradient_boosting", "xgboost", "lightgbm"):
        raise ValueError("model must be random_forest, gradient_boosting, xgboost or lightgbm")
    tr = data.enriched("train", ENRICHED_LEGIT)
    va = data.enriched("validation", 4000)
    feats = list(C.FEATURE_NAMES)
    extra = {"n_estimators": 150} if model == "random_forest" else {}
    pipe = build_pipeline(make_classifier(model, seed=0, **extra))
    pipe.fit(tr[feats], tr[C.TARGET])
    names = transformed_feature_names(pipe[:-1])
    est = pipe.named_steps["model"]

    def to_raw(name: str) -> str:
        return {"amount_log": "Amount", "hour_sin": "Time", "hour_cos": "Time"}.get(name, name)

    impurity_raw: dict[str, float] = {}
    for n, v in zip(names, est.feature_importances_):
        impurity_raw[to_raw(n)] = impurity_raw.get(to_raw(n), 0.0) + float(v)
    total = sum(impurity_raw.values()) or 1.0
    perm = permutation_importance(pipe, va[feats], va[C.TARGET], scoring="average_precision", n_repeats=5, random_state=0, n_jobs=1)
    rows = [
        {"feature": f, "impurity": r(impurity_raw.get(f, 0.0) / total), "permutation": r(perm.importances_mean[i]), "permutation_std": r(perm.importances_std[i])}
        for i, f in enumerate(feats)
    ]
    rows.sort(key=lambda d: -d["permutation"])

    shap_out: dict[str, Any] = {"available": False}
    if optional_available("shap"):
        try:
            import shap

            n_explain = 150 if model == "random_forest" else 400
            Xt = pipe[:-1].transform(va[feats].iloc[:n_explain])
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            base = np.asarray(explainer.expected_value).ravel()
            base_value = float(base[-1])
            mean_abs: dict[str, float] = {}
            for j, n in enumerate(names):
                mean_abs[n] = mean_abs.get(n, 0.0) + float(np.abs(sv[:, j]).mean())
            i = int(np.clip(explain_index, 0, len(sv) - 1))
            contrib = sorted(({"feature": names[j], "shap": r(sv[i, j]), "value": r(Xt[i, j])} for j in range(len(names))), key=lambda d: -abs(d["shap"]))
            raw_row = va[feats].iloc[i]
            shap_out = {
                "available": True,
                "output_space": "log-odds" if model in ("xgboost", "lightgbm", "gradient_boosting") else "probability",
                "rows_explained": int(len(sv)),
                "mean_abs_shap": sorted(({"feature": k, "value": r(v)} for k, v in mean_abs.items()), key=lambda d: -d["value"]),
                "example": {
                    "index": i,
                    "actual": int(va[C.TARGET].iloc[i]),
                    "base_value": r(base_value),
                    "output_value": r(base_value + float(sv[i].sum())),
                    "top_contributions": contrib[:10],
                    "raw_inputs": {k: r(v, 4) for k, v in raw_row.items()},
                },
            }
        except Exception as exc:  # SHAP is an optional, introductory extra
            shap_out = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    corr = data.train[PCA].corr().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    with_amount = data.train[PCA].corrwith(data.train["amount_log"]).abs().sort_values(ascending=False)
    return {
        "split": data.split_info(tr, va, "Permutation importance (average precision) uses every validation fraud + 4,000 legitimate validation rows."),
        "model": model,
        "importances": rows,
        "shap": shap_out,
        "limitations": {
            "max_abs_correlation_between_pca_components": r(np.abs(corr).max()),
            "strongest_correlation_with_log_amount": {"feature": str(with_amount.index[0]), "abs_correlation": r(with_amount.iloc[0])},
            "notes": [
                "V1-V28 are PCA components and therefore uncorrelated with each other, so permutation importance among them is not diluted by correlated copies.",
                "Amount is partly predictable from some components, which can share importance with it.",
                "Impurity importance is computed on training data and favours continuous, high-cardinality features.",
                "Neither measure is causal, and the PCA components have no published meaning, so importance cannot be translated into a business reason.",
            ],
        },
    }
