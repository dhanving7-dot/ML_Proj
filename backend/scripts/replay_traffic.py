"""Replay real held-out rows through the running API ("replay logs" from the M6 handout).

Every request is sent over HTTP to /api/predict with ``X-Prediction-Source: replay`` so replayed
traffic is stored and shown separately from genuine online traffic. Optionally the dataset's true
outcome is posted back, which enables live-performance (concept drift) monitoring.

Usage (from backend/, with the API running):
    python -m scripts.replay_traffic --n 200 --record-outcomes
    python -m scripts.replay_traffic --n 200 --segment tier3      # a shifted population for drift checks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from ml import contract as C  # noqa: E402
from ml.data import clean_for_training, load_csv, split_data  # noqa: E402

SEGMENTS = {
    "all": lambda d: d,
    "tier3": lambda d: d[d["CollegeTier"] == "Tier3"],
    "backlogs": lambda d: d[d["HistoryOfBacklogs"] == "Yes"],
    "high_cgpa": lambda d: d[d["CGPA"] >= 8.5],
}


def to_record(row: pd.Series) -> dict:
    rec = {}
    for name in C.FEATURE_NAMES:
        v = row[name]
        f = C.FEATURES_BY_NAME[name]
        if pd.isna(v):
            rec[name] = None
        elif isinstance(f, C.NumericFeature):
            rec[name] = int(v) if f.integer else float(v)
        else:
            rec[name] = str(v)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--segment", choices=list(SEGMENTS), default="all")
    parser.add_argument("--record-outcomes", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    test = split_data(clean_for_training(load_csv(get_settings().default_dataset))).test
    pool = SEGMENTS[args.segment](test)
    rows = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)
    ok = correct = failed = 0
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        for _, row in rows.iterrows():
            r = client.post("/api/predict", json=to_record(row), headers={"X-Prediction-Source": "replay"})
            if r.status_code != 200:
                failed += 1
                continue
            body = r.json()
            ok += 1
            actual = int(row[C.TARGET])
            correct += int(body["prediction"] == actual)
            if args.record_outcomes:
                client.post(f"/api/predictions/{body['request_id']}/outcome", json={"actual_outcome": actual})
    print(f"Replayed {ok} rows from the held-out test split (segment={args.segment}); {failed} failed.")
    if ok:
        print(f"Accuracy on replayed rows: {correct / ok:.4f}" + (" (outcomes recorded)" if args.record_outcomes else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
