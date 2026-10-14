"""Laboratory experiment registry: typed parameters, result cache, concurrency limit and run log."""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ml import contract as C
from ml.lab import m2_linear, m3_trees, m4_unsupervised, m5_evaluation, m6_production
from ml.lab.common import LabData, get_lab_data

from ..config import Settings
from ..db import utcnow
from ..errors import InvalidInput, ModelUnavailable, NotFound
from ..orm import ExperimentRun
from .dataset_service import default_dataset
from .model_service import model_service, validate_record


class P(BaseModel):
    model_config = ConfigDict(extra="forbid")


FeatureName = Literal[tuple(C.FEATURE_NAMES)]  # type: ignore[valid-type]


class LinRegP(P):
    learning_rate: float = Field(0.05, gt=0, le=1.0)
    iterations: int = Field(400, ge=10, le=5000)


class RegP(P):
    ridge_alpha: float = Field(1000.0, ge=1e-4, le=1e8)
    lasso_alpha: float = Field(0.01, ge=1e-5, le=10)
    enet_alpha: float = Field(0.01, ge=1e-5, le=10)
    l1_ratio: float = Field(0.5, ge=0.0, le=1.0)


class LogisticP(P):
    C_value: float = Field(1.0, ge=1e-4, le=1e4)
    regularization: Literal["l2", "l1", "none"] = "l2"
    threshold: float = Field(0.5, ge=0.01, le=0.99)
    class_weight: Literal["balanced", "none"] = "balanced"


class PreprocP(P):
    feature: FeatureName = "Amount"  # type: ignore[valid-type]
    missing_rate: float = Field(0.1, ge=0.01, le=0.5)


class TreeP(P):
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    max_depth: Optional[int] = Field(4, ge=1, le=30)
    min_samples_leaf: int = Field(20, ge=1, le=500)
    ccp_alpha: float = Field(0.0, ge=0.0, le=0.05)


class PruneP(P):
    criterion: Literal["gini", "entropy"] = "gini"


class RegTreeP(P):
    max_depth: int = Field(4, ge=1, le=15)


class EnsembleP(P):
    n_estimators: int = Field(200, ge=10, le=500)
    max_features: Literal["sqrt", "log2", "all"] = "sqrt"
    learning_rate: float = Field(0.1, ge=0.01, le=1.0)


class ImportanceP(P):
    model: Literal["random_forest", "gradient_boosting", "xgboost", "lightgbm"] = "random_forest"
    explain_index: int = Field(0, ge=0, le=149)


class KMeansP(P):
    k: int = Field(4, ge=2, le=10)
    sample_size: int = Field(3000, ge=500, le=6000)


class HierP(P):
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    n_clusters: int = Field(3, ge=2, le=10)
    sample_size: int = Field(500, ge=100, le=1000)


class DbscanP(P):
    eps: float = Field(4.0, gt=0.05, le=20)
    min_samples: int = Field(10, ge=2, le=50)
    sample_size: int = Field(2500, ge=500, le=4000)


class PcaP(P):
    sample_size: int = Field(20000, ge=1000, le=60000)
    standardize: bool = True


class EmbedP(P):
    method: Literal["tsne", "umap"] = "tsne"
    perplexity: float = Field(30.0, ge=5, le=100)
    n_neighbors: int = Field(15, ge=2, le=100)
    sample_size: int = Field(1500, ge=300, le=2500)


class AnomalyP(P):
    contamination: float = Field(0.002, ge=0.0005, le=0.2)
    nu: float = Field(0.002, ge=0.0005, le=0.5)


ModelSmall = Literal["logistic_regression", "decision_tree", "random_forest"]
ModelEval = Literal["logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting", "adaboost"]


class CvP(P):
    k: int = Field(5, ge=2, le=10)
    model: ModelSmall = "logistic_regression"
    sample_size: int = Field(20000, ge=2000, le=80000)


class NestedP(P):
    outer_k: int = Field(5, ge=3, le=10)
    inner_k: int = Field(3, ge=2, le=5)
    n_legit: int = Field(8000, ge=2000, le=20000)


class MetricsP(P):
    model: ModelEval = "logistic_regression"
    threshold: float = Field(0.5, ge=0.01, le=0.99)


class CalibP(P):
    model: ModelEval = "random_forest"


class SearchP(P):
    model: Literal["hist_gradient_boosting", "random_forest"] = "hist_gradient_boosting"
    budget: int = Field(12, ge=4, le=30)


class LearnP(P):
    model: Literal["decision_tree", "logistic_regression", "random_forest"] = "decision_tree"


class CompareP(P):
    model_a: ModelEval = "logistic_regression"
    model_b: ModelEval = "random_forest"


class NoP(P):
    pass


class PitP(P):
    model: Literal["logistic_regression", "hist_gradient_boosting"] = "logistic_regression"
    time_shift_hours: int = Field(6, ge=1, le=23)


class OnnxP(P):
    model: Literal["logistic_regression", "random_forest", "hist_gradient_boosting"] = "logistic_regression"


class DriftP(P):
    segment: Literal["random", "high_amount", "night", "second_day", "fraud_only"] = "high_amount"


@dataclass(frozen=True)
class Experiment:
    module: str
    key: str
    title: str
    topics: tuple[str, ...]
    params: type[P]
    fn: Callable[..., dict[str, Any]]
    needs_model: bool = False


EXPERIMENTS: dict[tuple[str, str], Experiment] = {
    (e.module, e.key): e
    for e in [
        Experiment("m2", "linear-regression", "Linear regression: normal equation vs gradient descent", ("Linear regression", "Normal equation", "Gradient descent", "RMSE / MAE / R²"), LinRegP, m2_linear.linear_regression),
        Experiment("m2", "regularization", "Ridge, Lasso and Elastic Net", ("L2 regularisation", "L1 regularisation", "Elastic Net", "Regularisation path"), RegP, m2_linear.regularization),
        Experiment("m2", "logistic", "Logistic regression, class weights and thresholds", ("Logistic regression", "Sigmoid", "Binary cross-entropy", "Class weights", "Decision threshold", "Multinomial logistic regression"), LogisticP, m2_linear.logistic),
        Experiment("m2", "preprocessing", "Scaling, encoding and imputation", ("StandardScaler", "MinMaxScaler", "log transform", "One-hot", "Ordinal", "Target encoding (cross-fitted)", "Mean / median / model-based imputation", "MissingIndicator"), PreprocP, m2_linear.preprocessing),
        Experiment("m3", "decision-tree", "Decision tree: Gini, entropy and information gain", ("Decision trees", "Gini impurity", "Entropy", "Information gain", "Pre-pruning"), TreeP, m3_trees.decision_tree),
        Experiment("m3", "pruning", "Pre-pruning and cost-complexity post-pruning", ("max_depth", "Cost-complexity pruning", "Overfitting"), PruneP, m3_trees.pruning),
        Experiment("m3", "regression-tree", "Regression tree splitting with MSE", ("Regression trees", "MSE reduction"), RegTreeP, m3_trees.regression_tree),
        Experiment("m3", "ensembles", "Bagging and boosting ensembles", ("Random forest", "Bootstrap", "Feature subsampling", "OOB estimate", "AdaBoost", "Gradient boosting", "XGBoost", "LightGBM"), EnsembleP, m3_trees.ensembles),
        Experiment("m3", "importance", "Impurity, permutation and SHAP importance", ("Impurity importance", "Permutation importance", "SHAP"), ImportanceP, m3_trees.importance),
        Experiment("m4", "kmeans", "K-means, elbow and silhouette", ("K-means", "Elbow method", "Silhouette score"), KMeansP, m4_unsupervised.kmeans),
        Experiment("m4", "hierarchical", "Hierarchical clustering and dendrograms", ("Single / complete / average / Ward linkage", "Dendrogram", "Cophenetic correlation"), HierP, m4_unsupervised.hierarchical),
        Experiment("m4", "dbscan", "DBSCAN density clustering", ("DBSCAN", "Core / border / noise points", "k-distance plot"), DbscanP, m4_unsupervised.dbscan),
        Experiment("m4", "pca", "Principal component analysis", ("PCA", "Explained variance", "Loadings", "Standardisation"), PcaP, m4_unsupervised.pca),
        Experiment("m4", "embedding", "t-SNE and UMAP embeddings", ("t-SNE", "UMAP", "Trustworthiness"), EmbedP, m4_unsupervised.embedding),
        Experiment("m4", "anomaly", "Isolation Forest and One-Class SVM", ("Isolation Forest", "One-Class SVM", "Autoencoder (conceptual)"), AnomalyP, m4_unsupervised.anomaly),
        Experiment("m5", "cross-validation", "K-fold vs stratified K-fold", ("K-fold", "Stratified K-fold", "Rare positives"), CvP, m5_evaluation.cross_validation),
        Experiment("m5", "nested-cv", "Nested cross-validation", ("Nested CV", "Selection bias"), NestedP, m5_evaluation.nested_cv),
        Experiment("m5", "metrics", "Confusion matrix, ROC and precision-recall", ("Precision", "Recall", "F1", "ROC-AUC", "PR curve", "Average precision", "Accuracy paradox"), MetricsP, m5_evaluation.evaluation_metrics),
        Experiment("m5", "calibration", "Probability calibration", ("Platt scaling", "Isotonic regression", "Reliability diagram", "Brier score", "ECE"), CalibP, m5_evaluation.calibration),
        Experiment("m5", "search", "Grid, random and Bayesian search", ("Grid search", "Random search", "Bayesian optimisation (TPE)"), SearchP, m5_evaluation.hyperparameter_search),
        Experiment("m5", "learning-curves", "Learning and validation curves", ("Learning curve", "Validation curve", "Under/overfitting"), LearnP, m5_evaluation.learning_curves),
        Experiment("m5", "statistical-tests", "McNemar's test and paired bootstrap", ("McNemar's test", "Paired bootstrap", "Average precision difference"), CompareP, m5_evaluation.statistical_comparison),
        Experiment("m6", "skew", "Training-serving skew", ("Golden E2E test", "Feature-vector diff", "Re-fitted scaler", "Unit mismatch", "Swapped columns", "Double transform"), NoP, m6_production.training_serving_skew, needs_model=True),
        Experiment("m6", "point-in-time", "Point-in-time correctness and feature semantics", ("Random vs time-ordered split", "Future leakage", "Same name, different meaning"), PitP, m6_production.point_in_time),
        Experiment("m6", "onnx", "ONNX export and parity check", ("ONNX", "skl2onnx", "ONNX Runtime", "Prediction parity"), OnnxP, m6_production.onnx_export),
        Experiment("m6", "packaging", "Pickle vs joblib packaging", ("Pickle", "Joblib", "Compression", "Artifact integrity", "gRPC (conceptual)"), NoP, m6_production.packaging, needs_model=True),
        Experiment("m6", "drift-sandbox", "Drift sandbox: PSI, KL and performance", ("PSI", "KL divergence", "Input drift vs performance"), DriftP, m6_production.drift_sandbox, needs_model=True),
    ]
}


class LabService:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._sem: threading.BoundedSemaphore | None = None

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "module": e.module,
                "key": e.key,
                "title": e.title,
                "topics": list(e.topics),
                "needs_model": e.needs_model,
                "params_schema": e.params.model_json_schema(),
            }
            for e in EXPERIMENTS.values()
        ]

    def _data(self, session: Session) -> tuple[LabData, str]:
        ds = default_dataset(session)
        if ds is None:
            raise NotFound("No dataset is registered for the laboratory")
        return get_lab_data(ds.path, ds.sha256), ds.sha256

    def run(self, session: Session, settings: Settings, module: str, key: str, raw_params: dict[str, Any]) -> dict[str, Any]:
        exp = EXPERIMENTS.get((module, key))
        if exp is None:
            raise NotFound(f"Unknown experiment {module}/{key}")
        try:
            params = exp.params.model_validate(raw_params or {})
        except Exception as exc:
            raise InvalidInput(f"Invalid parameters: {exc}") from None
        am = model_service.active
        if exp.needs_model and am is None:
            raise ModelUnavailable("This experiment needs an active model; train one first")
        data, sha = self._data(session)
        p = params.model_dump()
        cache_key = json.dumps([module, key, p, sha, am.version if (exp.needs_model and am) else None], sort_keys=True)
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return {**self._cache[cache_key], "cached": True}
        if self._sem is None:
            self._sem = threading.BoundedSemaphore(settings.lab_max_concurrency)
        with self._sem:
            t = time.perf_counter()
            status, error = "succeeded", None
            try:
                if key == "skew":
                    result = exp.fn(data, am.model, {**am.metadata, "version": am.version}, validate_record)  # type: ignore[union-attr]
                elif key == "packaging":
                    result = exp.fn(am.model, {**am.metadata, "version": am.version}, am.artifact_path)  # type: ignore[union-attr]
                elif exp.needs_model:
                    result = exp.fn(data, am.model, {**am.metadata, "version": am.version}, **p)  # type: ignore[union-attr]
                else:
                    result = exp.fn(data, **p)
            except ValueError as exc:
                status, error = "failed", str(exc)[:300]
                raise InvalidInput(error) from None
            finally:
                duration = (time.perf_counter() - t) * 1000
                session.add(ExperimentRun(module=module, experiment=key, params=p, status=status, duration_ms=round(duration, 1), error=error))
        payload = {
            "module": module,
            "experiment": key,
            "title": exp.title,
            "params": p,
            "ran_at": utcnow(),
            "duration_ms": round(duration, 1),
            "dataset_sha256": sha,
            "model_version": am.version if (exp.needs_model and am) else None,
            "result": result,
            "cached": False,
        }
        with self._cache_lock:
            self._cache[cache_key] = payload
            while len(self._cache) > settings.lab_cache_size:
                self._cache.popitem(last=False)
        return payload


lab_service = LabService()
