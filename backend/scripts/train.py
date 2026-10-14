"""Train the hybrid fraud model, evaluate it and (if it passes every gate) register + activate it.

Usage (from backend/):
    python -m scripts.train                        # uses ../params.yaml (full dataset, ~20-40 s)
    python -m scripts.train --sample-rows 60000    # quicker run on a stratified sample
    python -m scripts.train --no-promote           # register as a candidate; promote later in the UI/API

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
    parser.add_argument("--no-promote", action="store_true", help="register the model as a candidate instead of activating it")
    parser.add_argument("--trigger", default="cli", choices=["cli", "dvc"])
    args = parser.parse_args()

    params = {}
    if Path(args.params).exists():
        params = (yaml.safe_load(Path(args.params).read_text(encoding="utf-8")) or {}).get("train", {})
    if args.sample_rows:
        params["sample_rows"] = args.sample_rows
    if args.no_promote:
        params["auto_promote"] = False
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
    metrics = info.get("candidate_metrics") or {}
    flat = {
        "status": info["status"],
        "decision": info["decision"],
        "model_type": info["candidate_model_type"],
        "version": info["new_version"],
        "policy": metrics.get("policy"),
        "validation": {k: v for k, v in (metrics.get("validation") or {}).items() if k != "decisions"},
        "test": {k: v for k, v in (metrics.get("test") or {}).items() if k != "decisions"},
    }
    (settings.reports_dir / "metrics.json").write_text(json.dumps(flat, indent=2, default=str), encoding="utf-8")

    for entry in info["log"]:
        print(f"  [{entry['stage']}] {entry['message']}")
    print()
    print(f"Status: {info['status']}  Decision: {info['decision']}  Version: {info['new_version']}")
    print(f"Reason: {info['reason']}")
    if metrics.get("policy"):
        print(f"Policy: {json.dumps(metrics['policy'])}")
    for name, m in (metrics.get("test") or {}).items():
        if name == "decisions":
            continue
        print(
            f"Held-out test {name:<22} precision={m.get('precision')} recall={m.get('recall')} f1={m.get('f1')} "
            f"PR-AUC={m.get('average_precision')} ROC-AUC={m.get('roc_auc')} FP={m.get('false_positives')} FN={m.get('false_negatives')}"
        )
    if info.get("mlflow_error"):
        print(f"MLflow: {info['mlflow_error']}")
    elif info.get("mlflow_run_id"):
        print(f"MLflow run: {info['mlflow_run_id']}")
    return 0 if info["status"] in ("succeeded", "rejected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
