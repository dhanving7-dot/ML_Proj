"""Request/response schemas. ``StudentFeatures`` is generated from the feature contract, so the API
schema, the training data contract and the frontend form can never disagree on names, ranges or enums."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from ml import contract as C
from ml.estimators import CLASSIFIER_LABELS

# A real row (StudentID 4) from the dataset, used only as the OpenAPI example.
EXAMPLE_STUDENT: dict[str, Any] = {
    "Gender": "Male", "City": "Jaipur", "Stream": "CS", "Specialisation": "AI", "Hostel": "No",
    "HistoryOfBacklogs": "No", "CollegeTier": "Tier1",
    "SGPA_Sem1": 7.67, "SGPA_Sem2": 8.03, "SGPA_Sem3": 8.08, "SGPA_Sem4": 7.64,
    "SGPA_Sem5": 6.86, "SGPA_Sem6": 7.1, "SGPA_Sem7": 7.56, "SGPA_Sem8": 7.39, "CGPA": 7.51,
    "AttendancePercent": 68.8, "Internships": 1, "Projects": 2, "Workshops": 1, "Certifications": 2,
    "Publications": 0, "AptitudeTestScore": 69.8, "SoftSkillsRating": 2.7, "CodingTestScore": 66.9,
    "MockInterviewScore": 48.0, "ExtraCurricular": 0,
}


def _build_student_model() -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for f in C.ALL_FEATURES:
        if isinstance(f, C.NumericFeature):
            base = int if f.integer else float
            ann = Annotated[base, Field(ge=f.minimum, le=f.maximum)]
            if f.nullable:
                fields[f.name] = (Optional[ann], Field(default=None, description=f"{f.description} Optional: imputed with the training median when null."))
            else:
                fields[f.name] = (ann, Field(..., description=f.description))
        else:
            fields[f.name] = (Literal[tuple(f.categories)], Field(..., description=f.description))  # type: ignore[valid-type]
    return create_model(  # type: ignore[call-overload]
        "StudentFeatures",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_STUDENT}),
        **fields,
    )


StudentFeatures = _build_student_model()

DISCLAIMER = (
    "This is a statistical estimate from a model trained on historical data, not a guarantee. "
    "Placement outcomes depend on many factors the model does not observe."
)


class PredictionResponse(BaseModel):
    request_id: str
    created_at: datetime
    model_version: str
    model_type: str
    prediction: int
    label: str
    probability: float = Field(description="Model-estimated probability of being placed.")
    threshold: float
    latency_ms: float
    timings: dict[str, float]
    imputed_fields: list[str]
    disclaimer: str = DISCLAIMER


class BatchRequest(BaseModel):
    records: list[StudentFeatures] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchItem(BaseModel):
    index: int
    request_id: str
    prediction: int
    label: str
    probability: float


class BatchError(BaseModel):
    index: int
    errors: list[dict[str, Any]]


class BatchResponse(BaseModel):
    batch_id: str
    model_version: str
    count: int
    placed_count: int
    latency_ms: float
    results: list[BatchItem]
    errors: list[BatchError] = []
    disclaimer: str = DISCLAIMER


class OutcomeRequest(BaseModel):
    actual_outcome: Literal[0, 1] = Field(description="Ground-truth outcome once known: 1 = placed, 0 = not placed.")


CandidateName = Literal[tuple(CLASSIFIER_LABELS)]  # type: ignore[valid-type]


class TrainingConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: Optional[list[CandidateName]] = Field(default=None, min_length=1)  # type: ignore[valid-type]
    feature_set: Literal["full", "pre_final_year"] = "full"
    engineered_features: bool = True
    cv_folds: int = Field(default=5, ge=2, le=10)
    selection_metric: Literal["roc_auc", "f1", "accuracy", "neg_log_loss"] = "roc_auc"
    calibration: Literal["none", "sigmoid", "isotonic"] = "none"
    threshold_strategy: Literal["f1", "fixed"] = "f1"
    fixed_threshold: float = Field(default=0.5, gt=0, lt=1)
    sample_rows: Optional[int] = Field(default=None, ge=1000)
    min_validation_roc_auc: float = Field(default=0.75, ge=0, le=1)


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Optional[int] = None
    config: TrainingConfigIn = TrainingConfigIn()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    time: datetime
    uptime_s: float
    database: str
    model_loaded: bool
    model_version: Optional[str]
    model_type: Optional[str]
    consistency_check: Optional[dict[str, Any]]
    message: Optional[str] = None
