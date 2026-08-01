#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RL and random search runs.")
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for best_path in sorted(args.run_root.glob("*_seed*/best_architecture.json")):
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        run_name = best_path.parent.name
        strategy, seed_text = run_name.rsplit("_seed", 1)
        evaluation = payload["evaluation"]
        rows.append(
            {
                "strategy": strategy,
                "seed": int(seed_text),
                "episode": payload["episode"],
                "map50_95": float(evaluation["map50_95"]),
                "reward": float(evaluation["reward"]),
                "params_m": float(evaluation["params_m"]),
                "flops_g": float(evaluation["flops_g"]),
                "latency_ms": float(evaluation["latency_ms"]),
                "run_dir": str(best_path.parent),
            }
        )

    if not rows:
        raise FileNotFoundError(f"no completed runs found under {args.run_root}")

    csv_path = args.run_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for strategy in sorted({row["strategy"] for row in rows}):
        values = [row["map50_95"] for row in rows if row["strategy"] == strategy]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{strategy}: mAP50-95={mean:.6f} +/- {std:.6f} (n={len(values)})")
    print(f"comparison_csv={csv_path}")


if __name__ == "__main__":
    main()
