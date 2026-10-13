"""Estimator factory. Optional libraries (XGBoost, LightGBM) are used only when installed."""
from __future__ import annotations

import importlib.util
from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

N_JOBS = -1


def optional_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def available_classifiers() -> list[str]:
    names = [k for k in CLASSIFIER_LABELS if k not in ("xgboost", "lightgbm")]
    if optional_available("xgboost"):
        names.append("xgboost")
    if optional_available("lightgbm"):
        names.append("lightgbm")
    return names


def make_classifier(name: str, seed: int = 42, **params: Any) -> Any:
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, **{"C": 1.0, **params})
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **{"max_depth": 8, "min_samples_leaf": 20, **params})
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=N_JOBS, **{"n_estimators": 300, "min_samples_leaf": 2, **params}
        )
    if name == "adaboost":
        return AdaBoostClassifier(random_state=seed, **{"n_estimators": 200, "learning_rate": 0.5, **params})
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=seed, **{"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, **params}
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=seed, **{"max_iter": 300, "learning_rate": 0.06, "early_stopping": False, **params}
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            n_jobs=N_JOBS,
            eval_metric="logloss",
            **{"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.8, **params},
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=-1,
            **{"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, **params},
        )
    raise ValueError(f"Unknown classifier '{name}'. Available: {available_classifiers()}")
