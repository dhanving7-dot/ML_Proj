"""Dataset registration, upload validation and statistics."""
from __future__ import annotations

import io
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import contract as C
from ml.data import clean_for_training, column_statistics, file_sha256, load_csv, split_data, validate_dataframe

from ..config import Settings
from ..errors import InvalidInput, NotFound, PayloadTooLarge
from ..orm import Dataset


def _positive_rate(df: pd.DataFrame) -> float | None:
    if C.TARGET not in df.columns:
        return None
    t = pd.to_numeric(df[C.TARGET], errors="coerce")
    return round(float(t.mean()), 4) if t.notna().any() else None


def register_file(session: Session, path: Path, *, name: str, source: str) -> Dataset:
    sha = file_sha256(path)
    existing = session.scalars(select(Dataset).where(Dataset.sha256 == sha, Dataset.source == source)).first()
    if existing is not None:
        return existing
    df = load_csv(path)
    report = validate_dataframe(df)
    ds = Dataset(
        name=name,
        path=str(path),
        sha256=sha,
        source=source,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        positive_rate=_positive_rate(df),
        validation_status=report["status"],
        validation_report=report,
    )
    session.add(ds)
    session.flush()
    return ds


def ensure_default_dataset(session: Session, settings: Settings) -> Dataset | None:
    path = Path(settings.default_dataset)
    if not path.exists():
        return None
    return register_file(session, path, name=path.name, source="default")


def default_dataset(session: Session) -> Dataset | None:
    return session.scalars(select(Dataset).where(Dataset.source == "default").order_by(Dataset.id.desc())).first()


def get_dataset(session: Session, dataset_id: int) -> Dataset:
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        raise NotFound(f"Dataset {dataset_id} not found")
    return ds


async def read_limited(upload: Any, max_bytes: int) -> bytes:
    """Read an UploadFile without ever holding more than ``max_bytes + 1`` bytes."""
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content.strip():
        raise InvalidInput("The uploaded file is empty")
    if b"\x00" in content[:4096]:
        raise InvalidInput("The uploaded file is not a text CSV")
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception:
        raise InvalidInput("The file could not be parsed as CSV") from None
    if df.shape[1] < 2:
        raise InvalidInput("The CSV must contain a header row and multiple columns")
    return df


def save_upload(session: Session, settings: Settings, filename: str, content: bytes) -> Dataset:
    df = parse_csv_bytes(content)
    report = validate_dataframe(df)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = settings.uploads_dir / f"{uuid.uuid4().hex}.csv"  # never trust the client filename on disk
    path.write_bytes(content)
    safe_name = Path(filename or "upload.csv").name[:200]
    ds = Dataset(
        name=safe_name,
        path=str(path),
        sha256=file_sha256(path),
        source="upload",
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        positive_rate=_positive_rate(df),
        validation_status=report["status"],
        validation_report=report,
    )
    session.add(ds)
    session.flush()
    return ds


def dataset_dict(ds: Dataset) -> dict[str, Any]:
    return {
        "id": ds.id,
        "name": ds.name,
        "source": ds.source,
        "sha256": ds.sha256,
        "n_rows": ds.n_rows,
        "n_columns": ds.n_columns,
        "positive_rate": ds.positive_rate,
        "validation_status": ds.validation_status,
        "uploaded_at": ds.uploaded_at,
    }


@lru_cache(maxsize=8)
def _stats_cached(path: str, sha: str) -> dict[str, Any]:
    df = load_csv(path)
    cols = [c for c in df.columns if c != C.ID_COLUMN]
    return {"rows": int(len(df)), "columns": column_statistics(df[cols])}


def dataset_stats(ds: Dataset) -> dict[str, Any]:
    return {"dataset_id": ds.id, **_stats_cached(ds.path, ds.sha256)}


@lru_cache(maxsize=4)
def _test_split(path: str, sha: str) -> pd.DataFrame:
    return split_data(clean_for_training(load_csv(path))).test


def sample_rows(ds: Dataset, n: int = 1, seed: int | None = None) -> list[dict[str, Any]]:
    """Real rows from the held-out test split (never seen in training), for the form's example loader."""
    test = _test_split(ds.path, ds.sha256)
    picked = test.sample(n=min(n, len(test)), random_state=seed)
    out = []
    for _, row in picked.iterrows():
        features = {}
        for name in C.FEATURE_NAMES:
            v = row[name]
            if pd.isna(v):
                features[name] = None
            elif isinstance(C.FEATURES_BY_NAME[name], C.NumericFeature) and C.FEATURES_BY_NAME[name].integer:  # type: ignore[union-attr]
                features[name] = int(v)
            elif hasattr(v, "item"):
                features[name] = v.item()
            else:
                features[name] = v
        out.append({"student_id": int(row[C.ID_COLUMN]), "features": features, "actual_outcome": int(row[C.TARGET])})
    return out
