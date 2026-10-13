"""Validate a dataset against the feature contract and write a JSON report (DVC stage).

Usage (from backend/):  python -m scripts.validate_data [path/to/data.csv]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from ml.data import file_sha256, load_csv, validate_dataframe  # noqa: E402


def main() -> int:
    settings = get_settings()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.default_dataset
    report = validate_dataframe(load_csv(path))
    report["dataset"] = {"name": path.name, "sha256": file_sha256(path)}
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "data_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for c in report["checks"]:
        print(f"[{c['status'].upper():4}] {c['name']}: {c['message']}")
    print(f"\nOverall: {report['status']}  ->  {out}")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
