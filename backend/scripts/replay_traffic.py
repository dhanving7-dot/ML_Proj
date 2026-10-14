"""Replay real held-out transactions through the running API ("replay logs" from the M6 handout).

Every request is sent over HTTP to /api/predict with ``X-Prediction-Source: replay`` so replayed
traffic is stored and shown separately from genuine online traffic. Rows come from the held-out
test split of the active model, so they were never used for fitting or tuning. Optionally the
dataset's true Class is posted back as feedback, which enables live-performance monitoring.

Usage (from backend/, with the API running):
    python -m scripts.replay_traffic --n 300 --record-outcomes
    python -m scripts.replay_traffic --n 200 --segment high_amount     # a shifted population for drift checks
    python -m scripts.replay_traffic --n 60 --segment fraud --record-outcomes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from ml import contract as C  # noqa: E402
from ml.data import load_csv, prepare_splits  # noqa: E402
from ml.features import hour_of_cycle  # noqa: E402

SEGMENTS = {
    "all": lambda d: d,
    "fraud": lambda d: d[d[C.TARGET] == 1],
    "high_amount": lambda d: d[d["Amount"] >= d["Amount"].quantile(0.95)],
    "night": lambda d: d[hour_of_cycle(d["Time"]) < 6],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--segment", choices=list(SEGMENTS), default="all")
    parser.add_argument("--record-outcomes", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        health = client.get("/api/health").json()
        if not health.get("model_loaded"):
            print("The API has no active model; train one first.", file=sys.stderr)
            return 1
        cfg = client.get(f"/api/models/{health['model_version']}").json()["metadata"]["config"]
        test = prepare_splits(
            load_csv(get_settings().default_dataset),
            seed=cfg["seed"],
            validation=cfg["validation_size"],
            test=cfg["test_size"],
            drop_duplicates=cfg["drop_duplicates"],
            sample_rows=cfg.get("sample_rows"),
        ).test
        pool = SEGMENTS[args.segment](test)
        rows = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)
        ok = failed = 0
        counts = {d: 0 for d in C.DECISIONS}
        caught = frauds = 0
        for idx, row in rows.iterrows():
            body = {name: float(row[name]) for name in C.FEATURE_NAMES}
            body[C.TRANSACTION_ID] = f"replay-test-{int(idx)}"
            r = client.post("/api/predict", json=body, headers={"X-Prediction-Source": "replay"})
            if r.status_code != 200:
                failed += 1
                continue
            res = r.json()
            ok += 1
            counts[res["decision"]] += 1
            actual = int(row[C.TARGET])
            frauds += actual
            caught += int(actual == 1 and res["decision"] != "LOW_RISK")
            if args.record_outcomes:
                client.post(f"/api/predictions/{res['request_id']}/outcome", json={"actual_outcome": actual})
    print(f"Replayed {ok} held-out test rows (segment={args.segment}); {failed} failed.")
    print(f"Decisions: {counts}")
    if frauds:
        print(f"Fraud rows in the replay: {frauds}; flagged REVIEW or HIGH_RISK: {caught} ({caught / frauds:.1%})" + (" (outcomes recorded)" if args.record_outcomes else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
