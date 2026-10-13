"""Train, evaluate and (if it passes every gate) register + activate a model.

Usage (from backend/):
    python -m scripts.train                       # uses ../params.yaml
    python -m scripts.train --sample-rows 8000    # quicker run on a stratified sample

Writes ``reports/metrics.json`` and ``reports/training_run.json`` (tracked as DVC metrics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.orm import TrainingRun  # noqa: E402
from app.services.dataset_service import ensure_default_dataset  # noqa: E402
from app.services.model_service import model_service  # noqa: E402
from app.services.training_service import run_dict, run_job_blocking  # noqa: E402
from ml.training import TrainConfig  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=str(settings.project_root / "params.yaml"))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--candidates", default=None, help="comma-separated, e.g. logistic_regression,xgboost")
    parser.add_argument("--trigger", default="cli", choices=["cli", "dvc"])
    args = parser.parse_args()

    params = {}
    if Path(args.params).exists():
        params = (yaml.safe_load(Path(args.params).read_text(encoding="utf-8")) or {}).get("train", {})
    if args.sample_rows:
        params["sample_rows"] = args.sample_rows
    if args.candidates:
        params["candidates"] = [c.strip() for c in args.candidates.split(",") if c.strip()]
    config = TrainConfig.from_dict(params)

    db.configure(settings.resolved_database_url)
    db.init_db()
    with db.session_scope() as s:
        ds = ensure_default_dataset(s, settings)
        if ds is None:
            print(f"Default dataset not found at {settings.default_dataset}", file=sys.stderr)
            return 1
    try:
        with db.session_scope() as s:
            model_service.load_active(s)
    except Exception as exc:
        print(f"Warning: current active model could not be loaded ({exc}); continuing", file=sys.stderr)

    print(f"Training with config: {json.dumps(config.to_dict())}")
    job_id = run_job_blocking(settings, config, trigger=args.trigger)
    with db.session_scope() as s:
        from sqlalchemy import select

        run = s.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).one()
        info = run_dict(run)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "training_run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    metrics = (info.get("candidate_metrics") or {})
    flat = {
        "status": info["status"],
        "decision": info["decision"],
        "model_type": info["candidate_model_type"],
        "version": info["new_version"],
        "validation": metrics.get("validation"),
        "test": metrics.get("test"),
    }
    (settings.reports_dir / "metrics.json").write_text(json.dumps(flat, indent=2, default=str), encoding="utf-8")

    for entry in info["log"]:
        print(f"  [{entry['stage']}] {entry['message']}")
    print(f"\nStatus: {info['status']}  Decision: {info['decision']}  Version: {info['new_version']}")
    print(f"Reason: {info['reason']}")
    if metrics.get("test"):
        t = metrics["test"]
        print(f"Held-out test: accuracy={t['accuracy']} f1={t['f1']} roc_auc={t['roc_auc']} (n={t['support']})")
    if info.get("mlflow_error"):
        print(f"MLflow: {info['mlflow_error']}")
    elif info.get("mlflow_run_id"):
        print(f"MLflow run: {info['mlflow_run_id']}")
    return 0 if info["status"] in ("succeeded", "rejected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
