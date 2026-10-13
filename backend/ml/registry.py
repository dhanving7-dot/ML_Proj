"""Filesystem model packaging: atomic writes, integrity hashes and trusted loading.

Each version lives in ``<models_dir>/<version>/`` with:
* ``model.joblib``   - the full Pipeline (feature engineering + preprocessing + estimator)
* ``metadata.json``  - feature schema, training config, metrics, data hash, library versions
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

import joblib

from .data import file_sha256


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact's hash does not match the registered hash."""


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
    }
    for mod in ("xgboost", "lightgbm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # optional dependency
            pass
    return versions


def save_artifact(models_dir: str | Path, version: str, pipeline: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write the artifact to a temporary directory, then atomically rename it into place.

    A crash mid-write therefore never leaves a half-written version that could be loaded.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    final = models_dir / version
    if final.exists():
        raise FileExistsError(f"Model version {version} already exists")
    tmp = models_dir / f".{version}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        model_path = tmp / "model.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        sha = file_sha256(model_path)
        meta = {**metadata, "version": version, "artifact_sha256": sha, "library_versions": library_versions()}
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "path": str(final),
        "model_path": str(final / "model.joblib"),
        "sha256": sha,
        "size_bytes": (final / "model.joblib").stat().st_size,
        "metadata": meta,
    }


def load_artifact(model_path: str | Path, expected_sha256: str) -> Any:
    """Load a joblib artifact only if its SHA-256 matches the registry record.

    joblib/pickle can execute code on load, so only artifacts produced by this
    pipeline and registered in the database are ever deserialised.
    """
    model_path = Path(model_path)
    actual = file_sha256(model_path)
    if actual != expected_sha256:
        raise ArtifactIntegrityError(f"Hash mismatch for {model_path.name}: artifact was modified or replaced")
    return joblib.load(model_path)


def read_metadata(version_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(version_dir) / "metadata.json").read_text(encoding="utf-8"))
