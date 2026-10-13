"""Feature contract: the single source of truth for column names, types, ranges and enums.

Both the training pipeline and the serving API validate data against this
contract (the handout's "data contract" defence against training-serving skew).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TARGET = "PlacementStatus"
REGRESSION_TARGET = "Salary Package"
ID_COLUMN = "StudentID"
ANOMALY_LABEL = "IsAnomaly"
CLASS_LABELS = {0: "Not Placed", 1: "Placed"}

FeatureGroup = Literal["profile", "academic", "assessment", "experience"]


@dataclass(frozen=True)
class NumericFeature:
    name: str
    minimum: float
    maximum: float
    group: FeatureGroup
    description: str
    integer: bool = False
    nullable: bool = False
    unit: str | None = None

    kind: str = field(default="numeric", init=False)


@dataclass(frozen=True)
class CategoricalFeature:
    name: str
    categories: tuple[str, ...]
    group: FeatureGroup
    description: str
    ordered: bool = False  # True when a real ordering exists (ordinal encoding)
    nullable: bool = False

    kind: str = field(default="categorical", init=False)


SEMESTERS = tuple(f"SGPA_Sem{i}" for i in range(1, 9))

NOMINAL_FEATURES: tuple[CategoricalFeature, ...] = (
    CategoricalFeature("Gender", ("Male", "Female"), "profile", "Student gender as recorded by the institution."),
    CategoricalFeature(
        "City",
        ("Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"),
        "profile",
        "City of the college campus.",
    ),
    CategoricalFeature("Stream", ("CS", "IT", "ECE", "EE", "Mechanical", "Civil"), "profile", "Engineering stream."),
    CategoricalFeature(
        "Specialisation", ("AI", "DataScience", "Embedded", "Networking"), "profile", "Elective specialisation track."
    ),
    CategoricalFeature("Hostel", ("Yes", "No"), "profile", "Whether the student lives in the hostel."),
    CategoricalFeature("HistoryOfBacklogs", ("Yes", "No"), "academic", "Whether the student ever had a backlog."),
)

# Ordered from lowest to highest so OrdinalEncoder preserves the real ordering.
ORDINAL_FEATURES: tuple[CategoricalFeature, ...] = (
    CategoricalFeature(
        "CollegeTier", ("Tier3", "Tier2", "Tier1"), "profile", "Institution tier (Tier1 is the highest).", ordered=True
    ),
)

NUMERIC_FEATURES: tuple[NumericFeature, ...] = (
    *(NumericFeature(s, 0.0, 10.0, "academic", f"Semester {s[-1]} grade point average.") for s in SEMESTERS),
    NumericFeature("CGPA", 0.0, 10.0, "academic", "Cumulative grade point average."),
    NumericFeature("AttendancePercent", 0.0, 100.0, "academic", "Overall attendance.", unit="%"),
    NumericFeature("Internships", 0, 20, "experience", "Number of internships completed.", integer=True),
    NumericFeature("Projects", 0, 30, "experience", "Number of projects completed.", integer=True),
    NumericFeature("Workshops", 0, 30, "experience", "Workshops attended.", integer=True, nullable=True),
    NumericFeature("Certifications", 0, 30, "experience", "Certifications earned.", integer=True),
    NumericFeature("Publications", 0, 20, "experience", "Research publications.", integer=True),
    NumericFeature("AptitudeTestScore", 0.0, 100.0, "assessment", "Aptitude test score.", nullable=True),
    NumericFeature("SoftSkillsRating", 0.0, 5.0, "assessment", "Soft-skills rating (0-5).", nullable=True),
    NumericFeature("CodingTestScore", 0.0, 100.0, "assessment", "Coding test score.", nullable=True),
    NumericFeature("MockInterviewScore", 0.0, 100.0, "assessment", "Mock interview score.", nullable=True),
    NumericFeature("ExtraCurricular", 0, 1, "experience", "1 if active in extra-curricular activities.", integer=True),
)

ALL_FEATURES: tuple[NumericFeature | CategoricalFeature, ...] = (
    *NOMINAL_FEATURES,
    *ORDINAL_FEATURES,
    *NUMERIC_FEATURES,
)
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in ALL_FEATURES)
FEATURES_BY_NAME = {f.name: f for f in ALL_FEATURES}

# Columns present in the raw dataset that must never be model inputs, with the reason.
EXCLUDED_COLUMNS: dict[str, str] = {
    ID_COLUMN: "Row identifier with no predictive meaning.",
    REGRESSION_TARGET: "Post-outcome leakage: a salary only exists after a student is placed.",
    ANOMALY_LABEL: "Dataset-generation annotation that is not available at prediction time.",
    "CGPA_Tier": (
        "Derived duplicate of CGPA (bins <6.5 / 6.5-8.0 / >=8.0) that disagrees with CGPA on a minority of rows; "
        "its definition is undocumented, so it cannot be reproduced reliably at serving time and CGPA already carries the signal."
    ),
}

# Documented binning rule observed in the dataset, used by data validation to detect inconsistent rows.
CGPA_TIER_RULE = {"cutoffs": (6.5, 8.0), "labels": ("Low", "Mid", "High")}

# Feature sets for the point-in-time experiment. "full" assumes all eight semesters are
# complete when a prediction is made; "pre_final_year" only uses information that exists
# before the final-year placement season.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full": FEATURE_NAMES,
    "pre_final_year": tuple(n for n in FEATURE_NAMES if n not in ("SGPA_Sem7", "SGPA_Sem8", "CGPA")),
}

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, TARGET)


def feature_schema(feature_set: str = "full") -> list[dict]:
    """JSON-serialisable description of the features, used by the API and the frontend form."""
    names = FEATURE_SETS[feature_set]
    out: list[dict] = []
    for name in names:
        f = FEATURES_BY_NAME[name]
        if isinstance(f, NumericFeature):
            out.append(
                {
                    "name": f.name,
                    "kind": "numeric",
                    "group": f.group,
                    "description": f.description,
                    "minimum": f.minimum,
                    "maximum": f.maximum,
                    "integer": f.integer,
                    "nullable": f.nullable,
                    "unit": f.unit,
                }
            )
        else:
            out.append(
                {
                    "name": f.name,
                    "kind": "ordinal" if f.ordered else "nominal",
                    "group": f.group,
                    "description": f.description,
                    "categories": list(f.categories),
                    "nullable": f.nullable,
                }
            )
    return out


def split_feature_kinds(names: tuple[str, ...] | list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, nominal, ordinal) feature names restricted to ``names`` in contract order."""
    wanted = set(names)
    numeric = [f.name for f in NUMERIC_FEATURES if f.name in wanted]
    nominal = [f.name for f in NOMINAL_FEATURES if f.name in wanted]
    ordinal = [f.name for f in ORDINAL_FEATURES if f.name in wanted]
    return numeric, nominal, ordinal
