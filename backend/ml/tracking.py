"""Thin MLflow wrapper. Tracking failures are reported, never hidden and never fatal to training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

EXPERIMENT_NAME = "fraud-detection"


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def mlflow_status(tracking_uri: str) -> dict[str, Any]:
    try:
        import mlflow

        return {"available": True, "version": mlflow.__version__, "tracking_uri": tracking_uri}
    except Exception as exc:  # pragma: no cover - import failure path
        return {"available": False, "error": str(exc)}


def log_training_run(
    *,
    tracking_uri: str,
    artifact_root: Path,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str],
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Log a training run. Returns ``{"run_id": ...}`` or ``{"error": ...}``."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            artifact_root.mkdir(parents=True, exist_ok=True)
            exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.resolve().as_uri())
        else:
            exp_id = exp.experiment_id
        flat: dict[str, float] = {}
        _flatten("", metrics, flat)
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            for k, v in flat.items():
                mlflow.log_metric(k.replace("/", "_"), v)
            mlflow.set_tags(tags)
            if artifact_dir is not None:
                mlflow.log_artifacts(str(artifact_dir), artifact_path="model_package")
        return {"run_id": run.info.run_id, "experiment_id": exp_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
