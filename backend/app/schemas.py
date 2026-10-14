"""Request/response schemas. ``TransactionFeatures`` is generated from the feature contract, so the API
schema, the training data contract and the frontend form can never disagree on names or ranges."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from ml import contract as C

# The first row of creditcard.csv (Class = 0), used only as the OpenAPI example.
EXAMPLE_TRANSACTION: dict[str, Any] = {
    "Time": 0.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
    "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.5516, "V12": -0.617801, "V13": -0.99139,
    "V14": -0.311169, "V15": 1.468177, "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
    "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, "V25": 0.128539,
    "V26": -0.189115, "V27": 0.133558, "V28": -0.021053, "Amount": 149.62,
}

TransactionId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:\-]+$")]


def _build_transaction_model() -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for f in C.ALL_FEATURES:
        ann = Annotated[float, Field(ge=f.minimum, le=f.maximum, allow_inf_nan=False, description=f.description)]
        fields[f.name] = (ann, Field(...))
    fields[C.TRANSACTION_ID] = (
        Optional[TransactionId],
        Field(default=None, description="Optional client reference stored with the prediction. Never used as a model input."),
    )
    return create_model(  # type: ignore[call-overload]
        "TransactionFeatures",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_TRANSACTION}),
        **fields,
    )


TransactionFeatures = _build_transaction_model()

DISCLAIMER = (
    "Risk scores are statistical estimates from models trained on a public, historical, anonymised dataset. "
    "They support, but do not replace, investigation by a person."
)

Decision = Literal["LOW_RISK", "REVIEW", "HIGH_RISK"]


class Scores(BaseModel):
    lr_probability: float = Field(description="Logistic Regression fraud probability (prior-corrected).")
    anomaly_score: float = Field(description="Isolation Forest anomaly score (-score_samples); not a probability.")
    anomaly_percentile: float = Field(description="Share of training-reference transactions with a lower or equal anomaly score.")
    lr_rarity: float
    if_rarity: float
    hybrid_score: float = Field(description="w * rarity_LR + (1 - w) * rarity_IF, in [0, 1].")


class Flags(BaseModel):
    logistic_regression: bool
    isolation_forest: bool


class PredictionResponse(BaseModel):
    request_id: str
    created_at: datetime
    model_version: str
    model_type: str
    transaction_id: Optional[str]
    decision: Decision
    scores: Scores
    flags: Flags
    policy: dict[str, Any]
    explanation: dict[str, Any]
    out_of_training_range: list[str]
    latency_ms: float
    timings: dict[str, float]
    disclaimer: str = DISCLAIMER


class BatchRequest(BaseModel):
    records: list[TransactionFeatures] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchItem(BaseModel):
    index: int
    request_id: str
    transaction_id: Optional[str]
    decision: Decision
    hybrid_score: float
    lr_probability: float
    anomaly_percentile: float


class BatchError(BaseModel):
    index: int
    errors: list[dict[str, Any]]


class BatchResponse(BaseModel):
    batch_id: str
    model_version: str
    count: int
    decision_counts: dict[str, int]
    latency_ms: float
    results: list[BatchItem]
    errors: list[BatchError] = []
    disclaimer: str = DISCLAIMER


class OutcomeRequest(BaseModel):
    actual_outcome: Literal[0, 1] = Field(description="Confirmed outcome once known: 1 = fraud, 0 = legitimate.")


class TrainingConfigIn(BaseModel):
    """The subset of ``ml.training.TrainConfig`` that the API accepts (validated twice)."""

    model_config = ConfigDict(extra="forbid")

    drop_duplicates: bool = True
    sample_rows: Optional[int] = Field(default=None, ge=5000)
    time_encoding: Literal["cyclical_hour", "raw", "none"] = "cyclical_hour"
    amount_transform: Literal["log1p", "raw"] = "log1p"
    lr_imbalance: Literal["class_weight", "undersample", "none"] = "class_weight"
    lr_undersample_ratio: float = Field(default=20.0, ge=1, le=1000)
    lr_c_grid: list[float] = Field(default=[0.01, 0.1, 1.0], min_length=1, max_length=6)
    lr_calibration: Literal["prior_correction", "none"] = "prior_correction"
    cv_folds: int = Field(default=5, ge=2, le=10)
    if_n_estimators: int = Field(default=300, ge=50, le=1000)
    if_max_samples: int = Field(default=256, ge=64, le=4096)
    if_reference: Literal["all_train", "legitimate_train"] = "all_train"
    if_flag_rate: float = Field(default=0.01, gt=0, lt=0.5)
    fusion_weight: Optional[float] = Field(default=None, ge=0, le=1)
    high_risk_strategy: Literal["f1", "precision_target"] = "f1"
    high_risk_precision_target: float = Field(default=0.8, gt=0, le=1)
    review_recall_target: float = Field(default=0.9, gt=0, le=1)
    max_review_rate: float = Field(default=0.01, gt=0, lt=0.5)
    if_review_percentile: Optional[float] = Field(default=None, ge=0.5, lt=1)
    min_validation_average_precision: float = Field(default=0.3, ge=0, le=1)
    auto_promote: bool = True


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Optional[int] = None
    config: TrainingConfigIn = TrainingConfigIn()


class ExploreRequest(BaseModel):
    """What-if decision policy evaluated on the validation split of the active model (never the test split)."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(ge=0, le=1)
    review_threshold: float = Field(ge=0, le=1)
    high_risk_threshold: float = Field(ge=0, le=1)
    lr_threshold: float = Field(gt=0, lt=1)
    if_flag_rate: float = Field(gt=0, lt=0.5)
    if_review_percentile: Optional[float] = Field(default=None, ge=0.5, lt=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    time: datetime
    uptime_s: float
    database: str
    model_loaded: bool
    model_version: Optional[str]
    model_type: Optional[str]
    policy: Optional[dict[str, Any]] = None
    consistency_check: Optional[dict[str, Any]]
    message: Optional[str] = None
