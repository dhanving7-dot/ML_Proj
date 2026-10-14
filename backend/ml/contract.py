"""Feature contract: the single source of truth for column names, types, ranges and labels.

Both the training pipeline and the serving API validate data against this contract (the
handout's "data contract" defence against training-serving skew).

Dataset: the European credit-card transactions dataset (``creditcard.csv``): ``Time``,
``V1``-``V28``, ``Amount`` and the label ``Class``. ``V1``-``V28`` are anonymised PCA
components; their original meaning is not published, so this project never assigns one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TARGET = "Class"
CLASS_LABELS = {0: "Legitimate", 1: "Fraud"}
POSITIVE_LABEL = 1

# Final risk decisions produced by the hybrid system, from lowest to highest risk.
DECISIONS: tuple[str, ...] = ("LOW_RISK", "REVIEW", "HIGH_RISK")

# Optional request metadata that is stored with a prediction but never used as a model input.
TRANSACTION_ID = "transaction_id"

FeatureGroup = Literal["time", "amount", "pca"]


@dataclass(frozen=True)
class NumericFeature:
    name: str
    group: FeatureGroup
    description: str
    minimum: float | None = None  # None = unbounded (the value must still be finite)
    maximum: float | None = None
    unit: str | None = None

    kind: str = field(default="numeric", init=False)


PCA_FEATURES: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))

NUMERIC_FEATURES: tuple[NumericFeature, ...] = (
    NumericFeature(
        "Time",
        "time",
        "Seconds elapsed between this transaction and the first transaction in the dataset.",
        minimum=0.0,
        unit="s",
    ),
    *(
        NumericFeature(name, "pca", "Anonymised principal component published with the dataset; its meaning is not disclosed.")
        for name in PCA_FEATURES
    ),
    NumericFeature("Amount", "amount", "Transaction amount (currency not disclosed).", minimum=0.0),
)

ALL_FEATURES: tuple[NumericFeature, ...] = NUMERIC_FEATURES
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in ALL_FEATURES)  # dataset column order
FEATURES_BY_NAME = {f.name: f for f in ALL_FEATURES}

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, TARGET)

# Columns that may appear in uploaded files but are never model inputs, with the reason.
# The published dataset has none; an identifier column supplied with a batch is kept as metadata.
EXCLUDED_COLUMNS: dict[str, str] = {
    TRANSACTION_ID: "Optional client identifier. Stored with the prediction for search, never used as a model input.",
}

# Minimum number of fraud rows required to train and evaluate (validation + test need positives).
MIN_POSITIVE_ROWS = 20


def feature_schema() -> list[dict]:
    """JSON-serialisable description of the features, used by the API and the frontend form."""
    return [
        {
            "name": f.name,
            "kind": "numeric",
            "group": f.group,
            "description": f.description,
            "minimum": f.minimum,
            "maximum": f.maximum,
            "unit": f.unit,
            "integer": False,
            "nullable": False,
        }
        for f in ALL_FEATURES
    ]
